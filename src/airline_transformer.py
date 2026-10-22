# %% [markdown]
# # S6E10 小型 Transformer：把表格的一行读成一组 token
#
# 池里 254 个成员已有 GBDT、MLP、TabPFN 等基础模型，但没有「在本数据上训练的 Transformer」。
# 思路参考 Chris Deotte 的讨论（Q-FT / ModernBERT 读表格行，OOF 0.9608），这里做一个不到 100 万参数的小模型（d=96、3 层），
# T4 上可跑完 5 折。
#
# ## 一行如何变成 token
#
# 每行 21 个原始列 → **每列一个 token**，再加 1 个 CLS 和 2 个摘要 token，共 24 个 d 维向量。
# 每个特征 token 是以下几部分的和：
#
# | 部分 | 作用 | 怎么来 |
# |---|---|---|
# | 列嵌入 | 告诉模型「这是哪一列」 | 每列一个可学习向量；没有位置编码，所以列的顺序无关 |
# | 取值嵌入 | 「这一列取了哪个值」 | 查表：类别、评分 0–5、每个年龄、每个航线（精确 `Flight Distance`，出现 ≥3 次）、延误分桶 |
# | 数值嵌入 | 让相近的数值共享信息 | 仅 4 个数值列：标准化后过 sin/cos 周期特征（可学习频率）再线性映射 |
# | 侧特征 | 关于「这个取值」的统计量 | 每列 4 个数：取值频次、同航线下该列均值、辅助任务期望值、实际 − 期望 |
#
# 两个摘要 token 描述整行：行统计（评分均值 / 离散度 / 计数、延误差）与原始数据先验（只用原始数据训练的模型的预测）。
# 侧特征和摘要都不用比赛标签（只用特征，或原始数据集自己的标签），所以只算一次，不必放进 CV 折内。
#
# 模型：3 层 pre-norm Transformer encoder（d=96，4 头），读 CLS 输出做二分类。固定 epoch 数，不在验证折上早停。
#
# | 版本 | epoch | OOF AUC | 备注 |
# |---|---|---|---|
# | v1 | 6 | 0.959915 | 第 3 个 epoch 验证 AUC 见顶（fold 1：0.96074），之后下降到 0.95993 |
# | v2 | 3 | — | cosine 在 3 个 epoch 内走完 |

# %%
import json
import math
import os
import shutil
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")

COMP = "playground-series-s6e10"
TARGET = "satisfaction"
ID = "id"
SEED = 42
N_SPLITS = 5
RUN_MODE = "full"          # "full" | "smoke"
SMOKE_ROWS = 20_000
N_SEEDS = 2
AUX_FOLDS = 3

IS_KAGGLE = Path("/kaggle/input").exists()
OUT_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("outputs/transformer")
OUT_DIR.mkdir(parents=True, exist_ok=True)
HAS_GPU = shutil.which("nvidia-smi") is not None


def resolve_dir() -> Path:
    if not IS_KAGGLE:
        return Path("data")
    for t in Path("/kaggle/input").rglob("train.csv"):
        if COMP in str(t):
            return t.parent
    raise FileNotFoundError("competition train.csv not found")


def find_orig() -> list[Path]:
    root = Path("/kaggle/input") if IS_KAGGLE else Path("data/orig")
    hits = []
    for p in root.rglob("*.csv"):
        if COMP in str(p):
            continue
        try:
            cols = pd.read_csv(p, nrows=0).columns
        except Exception:
            continue
        if TARGET in cols and "Unnamed: 0" in cols:
            hits.append(p)
    return hits


DATA_DIR = resolve_dir()
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
if RUN_MODE == "smoke":
    train = train.sample(SMOKE_ROWS, random_state=SEED).reset_index(drop=True)
    test = test.head(SMOKE_ROWS // 2).reset_index(drop=True)
y = train[TARGET].astype(int).to_numpy()
train_id, test_id = train[ID].to_numpy(), test[ID].to_numpy()
NTR, NTE = len(train), len(test)
full = pd.concat([train.drop(columns=[ID, TARGET]), test.drop(columns=[ID])], ignore_index=True)

orig_paths = find_orig()
orig = pd.concat([pd.read_csv(p) for p in orig_paths], ignore_index=True) if orig_paths else None
if orig is not None:
    orig = orig.drop(columns=[c for c in orig.columns if c.startswith("Unnamed") or c == "id"], errors="ignore")
    orig[TARGET] = orig[TARGET].astype(str).str.strip().str.lower().eq("satisfied").astype(int)
print(f"run_mode={RUN_MODE}  train={NTR:,} test={NTE:,}  orig={0 if orig is None else len(orig):,}  gpu={HAS_GPU}")

# %% [markdown]
# ## 1. 每列的取值编码（取值嵌入的词表）

# %%
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
RATINGS = [c for c in full.columns if c not in CAT_COLS + NUM_COLS]
COLS = list(full.columns)                      # 21 列，固定顺序只用于索引，模型本身不看顺序
DELAY_BINS = [-0.5, 0.5, 5.5, 15.5, 30.5, 60.5, 120.5, np.inf]

codes = np.zeros((len(full), len(COLS)), dtype=np.int64)
vocab = []
for j, c in enumerate(COLS):
    s = full[c]
    if c == "Flight Distance":
        # 航线 ID：出现 ≥3 次的精确距离各占一个编码，其余归入 "rare"
        vc = s.value_counts()
        keep = vc.index[vc >= 3]
        mapping = {v: i + 1 for i, v in enumerate(sorted(keep))}
        k = s.map(mapping).fillna(0).astype(int).to_numpy()
    elif c in ("Departure Delay in Minutes", "Arrival Delay in Minutes"):
        k = pd.cut(s, DELAY_BINS, labels=False).fillna(len(DELAY_BINS) - 1).astype(int).to_numpy()
    else:
        k = pd.factorize(s, sort=True)[0]
    codes[:, j] = k
    vocab.append(int(k.max()) + 1)
print("vocab sizes:", dict(zip(COLS, vocab)))

# 4 个数值列的标准化取值（数值嵌入的输入）
num = np.column_stack([
    full["Age"].to_numpy(float),
    np.log1p(full["Flight Distance"].to_numpy(float)),
    np.log1p(full["Departure Delay in Minutes"].fillna(0).to_numpy(float)),
    np.log1p(full["Arrival Delay in Minutes"].fillna(0).to_numpy(float)),
])
num = ((num - num.mean(0)) / num.std(0)).astype(np.float32)
NUM_IDX = [COLS.index(c) for c in NUM_COLS]

# %% [markdown]
# ## 2. 侧特征（每个 token 4 个数）
#
# 1. 该取值在 train+test 中的频次（log）
# 2. 同一航线（精确距离）下该列的均值（`Flight Distance` 自身改用航线行数）
# 3. 辅助任务期望值：用其余 20 列预测这个评分的期望（只对 13 个评分，其他列为 0）
# 4. 实际值 − 期望值（同上）

# %%
coded = full.copy()
for c in CAT_COLS:
    coded[c] = pd.factorize(coded[c], sort=True)[0]
coded = coded.fillna(0).astype(np.float32)

S = 4
side = np.zeros((len(full), len(COLS), S), dtype=np.float32)
grp = coded.groupby(full["Flight Distance"])
for j, c in enumerate(COLS):
    side[:, j, 0] = np.log1p(full[c].map(full[c].value_counts()).fillna(0).to_numpy(float))
    if c == "Flight Distance":
        side[:, j, 1] = np.log1p(grp[c].transform("size").to_numpy(float))
    else:
        side[:, j, 1] = grp[c].transform("mean").to_numpy(float)

t0 = time.time()
aux_device = "cuda" if HAS_GPU else "cpu"
aux_rounds = 300 if RUN_MODE == "full" else 30
for c in RATINGS:
    j = COLS.index(c)
    yc, uniq = pd.factorize(coded[c], sort=True)
    others = [f for f in COLS if f != c]
    P = np.zeros((len(coded), len(uniq)), dtype=np.float32)
    for a, b in KFold(AUX_FOLDS, shuffle=True, random_state=0).split(coded):
        m = xgb.XGBClassifier(
            n_estimators=aux_rounds, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.7,
            tree_method="hist", device=aux_device, random_state=0, objective="multi:softprob",
        )
        m.fit(coded.iloc[a][others], yc[a])
        P[b] = m.predict_proba(coded.iloc[b][others])
    ev = P @ np.asarray(uniq, dtype=np.float32)
    side[:, j, 2] = ev
    side[:, j, 3] = coded[c].to_numpy() - ev
print(f"aux expected values done ({time.time() - t0:.0f}s, device={aux_device})")

# 每列每个侧特征单独标准化
mu, sd = side.mean(0, keepdims=True), side.std(0, keepdims=True)
side = ((side - mu) / np.where(sd > 0, sd, 1)).astype(np.float32)

# %% [markdown]
# ## 3. 两个摘要 token 的输入
#
# - 行统计：13 个评分的均值 / 标准差 / 最小 / 最大、0 / ≤2 / ≥4 / 5 的个数、到达−出发延误、总延误
# - 原始数据先验：只用原始数据集训练的 LightGBM 对每行的预测（logit）

# %%
r = coded[RATINGS].to_numpy()
dep = full["Departure Delay in Minutes"].fillna(0).to_numpy(float)
arr = full["Arrival Delay in Minutes"].fillna(0).to_numpy(float)
row_stats = np.column_stack([
    r.mean(1), r.std(1), r.min(1), r.max(1),
    (r == 0).sum(1), (r <= 2).sum(1), (r >= 4).sum(1), (r == 5).sum(1),
    np.sign(arr - dep) * np.log1p(np.abs(arr - dep)), np.log1p(arr + dep),
]).astype(np.float32)
row_stats = (row_stats - row_stats.mean(0)) / row_stats.std(0)

if orig is not None:
    o = orig[COLS].copy()
    for c in CAT_COLS:
        o[c] = o[c].astype("category")
    f = full[COLS].copy()
    for c in CAT_COLS:
        f[c] = pd.Categorical(f[c], categories=o[c].cat.categories)
    teacher = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, random_state=0, verbose=-1,
    ).fit(o, orig[TARGET])
    p = np.clip(teacher.predict_proba(f)[:, 1], 1e-6, 1 - 1e-6)
    prior = np.log(p / (1 - p))[:, None].astype(np.float32)
    prior = (prior - prior.mean()) / prior.std()
    print(f"original-data teacher: AUC on train rows {roc_auc_score(y, prior[:NTR, 0]):.4f}")
else:
    prior = np.zeros((len(full), 1), dtype=np.float32)
    print("[warn] original dataset not found; prior token is zeros")

# %% [markdown]
# ## 4. 模型
#
# token = 列嵌入 + 取值嵌入 + 侧特征投影（+ 数值嵌入，仅数值列），RMS 归一化后进入 encoder。

# %%
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D, HEADS, LAYERS, FF, DROP = 96, 4, 3, 192, 0.1
N_FREQ = 16
EPOCHS = 3 if RUN_MODE == "full" else 2   # v1 用 6：验证 AUC 在第 3 个 epoch 见顶后下降（航线 embedding 过拟合）
BATCH = 1024
LR, WD = 1e-3, 1e-2


class RowTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        n_col = len(COLS)
        self.col_emb = nn.Parameter(torch.randn(n_col, D) * 0.02)
        offsets = np.concatenate([[0], np.cumsum(vocab)[:-1]])
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.val_emb = nn.Embedding(int(sum(vocab)), D)
        nn.init.normal_(self.val_emb.weight, std=0.02)
        self.side_w = nn.Parameter(torch.randn(n_col, S, D) * 0.02)       # 每列独立的侧特征投影
        self.freq = nn.Parameter(torch.randn(len(NUM_IDX), N_FREQ) * 1.0)  # 周期数值嵌入的可学习频率
        self.num_w = nn.Parameter(torch.randn(len(NUM_IDX), 2 * N_FREQ, D) * 0.02)
        self.register_buffer("num_idx", torch.tensor(NUM_IDX, dtype=torch.long))
        self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.stats_proj = nn.Linear(row_stats.shape[1], D)
        self.prior_proj = nn.Linear(prior.shape[1], D)
        self.norm = nn.RMSNorm(D)

    def forward(self, codes_b, num_b, side_b, stats_b, prior_b):
        tok = self.col_emb + self.val_emb(codes_b + self.offsets)               # (B, 21, D)
        tok = tok + torch.einsum("bcs,csd->bcd", side_b, self.side_w)
        ang = 2 * math.pi * num_b.unsqueeze(-1) * self.freq                     # (B, 4, F)
        per = torch.cat([torch.sin(ang), torch.cos(ang)], -1)                   # (B, 4, 2F)
        num_tok = torch.einsum("bnf,nfd->bnd", per, self.num_w)
        tok = tok.index_add(1, self.num_idx, num_tok.to(tok.dtype))   # autocast 下 einsum 输出是半精度
        summary = torch.stack([self.stats_proj(stats_b), self.prior_proj(prior_b)], 1)   # (B, 2, D)
        x = torch.cat([self.cls.expand(len(tok), -1, -1), tok, summary], 1)            # (B, 24, D)
        return self.norm(x)


class TabTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = RowTokenizer()
        layer = nn.TransformerEncoderLayer(D, HEADS, FF, DROP, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

    def forward(self, *inputs):
        h = self.encoder(self.tokenizer(*inputs))
        return self.head(h[:, 0]).squeeze(-1)


n_params = sum(p.numel() for p in TabTransformer().parameters())
print(f"model parameters: {n_params / 1e6:.2f}M  device={DEVICE}")

# %% [markdown]
# ## 5. 5 折训练
#
# AdamW + 5% warmup + cosine，固定 epoch 数（每个 epoch 的验证 AUC 只打印、不用于选模型）。
# `N_SEEDS` 个 seed 平均。

# %%
T = {
    "codes": torch.tensor(codes),
    "num": torch.tensor(num),
    "side": torch.tensor(side),
    "stats": torch.tensor(row_stats),
    "prior": torch.tensor(prior),
}
T = {k: v.to(DEVICE) for k, v in T.items()}
KEYS = ["codes", "num", "side", "stats", "prior"]
y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)


def batches(idx, shuffle, gen=None):
    idx = torch.as_tensor(idx, device=DEVICE)
    if shuffle:
        idx = idx[torch.randperm(len(idx), device=DEVICE, generator=gen)]
    for s in range(0, len(idx), BATCH):
        b = idx[s : s + BATCH]
        yield b, [T[k][b] for k in KEYS]


@torch.no_grad()
def predict(model, idx):
    model.eval()
    out = []
    for _, xb in batches(idx, False):
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            out.append(model(*xb).float())
    return torch.cat(out).cpu().numpy()


folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(np.zeros(NTR), y))
test_idx = np.arange(NTR, NTR + NTE)
oof = np.zeros(NTR)
pred = np.zeros(NTE)
t_all = time.time()
for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
    for s_i in range(N_SEEDS):
        torch.manual_seed(SEED + 100 * s_i + fold)
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + 100 * s_i + fold)
        model = TabTransformer().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        steps = EPOCHS * math.ceil(len(tr_idx) / BATCH)
        warm = max(1, int(0.05 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, t / steps)))
        )
        scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
        loss_fn = nn.BCEWithLogitsLoss()
        for ep in range(1, EPOCHS + 1):
            model.train()
            t_ep = time.time()
            for b, xb in batches(tr_idx, True, gen):
                with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                    loss = loss_fn(model(*xb).float(), y_t[b])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            va_auc = roc_auc_score(y[va_idx], predict(model, va_idx))
            print(f"  fold {fold} seed {s_i} epoch {ep}: val auc {va_auc:.6f}  ({time.time() - t_ep:.0f}s)", flush=True)
        oof[va_idx] += predict(model, va_idx) / N_SEEDS
        pred += predict(model, test_idx) / (N_SEEDS * N_SPLITS)
        del model, opt
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[fold {fold}] auc={roc_auc_score(y[va_idx], oof[va_idx]):.6f}  ({time.time() - t_all:.0f}s)", flush=True)

auc = roc_auc_score(y, oof)
print(f"\nOOF AUC = {auc:.6f}")

# %% [markdown]
# ## 6. 结果
#
# 输出是 logit；`submission.csv` 转成概率。OOF / test 以 `oof_transformer.npy` / `test_transformer.npy` 保存，供堆叠使用。

# %%
np.save(OUT_DIR / "oof_transformer.npy", oof.astype(np.float32))
np.save(OUT_DIR / "test_transformer.npy", pred.astype(np.float32))
sub = pd.DataFrame({ID: test_id, TARGET: 1 / (1 + np.exp(-pred))})
assert sub[ID].is_unique and sub[TARGET].between(0, 1).all()
sub.to_csv(OUT_DIR / "submission.csv", index=False)
metrics = {
    "oof_auc": float(auc),
    "n_params": int(n_params),
    "epochs": EPOCHS,
    "n_seeds": N_SEEDS,
    "minutes": round((time.time() - t_all) / 60, 1),
}
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
