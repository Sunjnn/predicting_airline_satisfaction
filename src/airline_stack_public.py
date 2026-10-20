# %% [markdown]
# # S6E10 公开 OOF 堆叠（实验）
#
# 公开 notebook（`cdeotte/stacking-tfms-with-public-oof`）的做法：竞赛里已有多个公开的
# OOF/test 预测库，全部用同一套 `StratifiedKFold(5, shuffle=True, random_state=42)`。
# 把它们的 OOF 当特征训一个二级模型，能拿到比任何单模型都高的分数。
#
# **注意**：这个提交的分数主要来自他人模型的预测，不是自己的建模能力。Playground 常见，
# 但含金量与自建模型不同，是否作为最终提交由使用者决定。
#
# 本 notebook 先确认：各库的格式、行顺序、单模型 OOF 水平，以及堆叠后的 OOF AUC。

# %%
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

COMP = "playground-series-s6e10"
TARGET = "satisfaction"
SEED = 42
N_SPLITS = 5

IS_KAGGLE = Path("/kaggle/input").exists()
OUT = Path("/kaggle/working") if IS_KAGGLE else Path("outputs/stack")
OUT.mkdir(parents=True, exist_ok=True)

comp_dirs = [p for p in Path("/kaggle/input").rglob("train.csv") if COMP in str(p)]
if not comp_dirs:
    raise FileNotFoundError("competition data not mounted")
COMP_DIR = comp_dirs[0].parent

train = pd.read_csv(COMP_DIR / "train.csv", usecols=["id", TARGET])
test = pd.read_csv(COMP_DIR / "test.csv", usecols=["id"])
sample_sub = pd.read_csv(COMP_DIR / "sample_submission.csv")
y = train[TARGET].astype(int).to_numpy()
train_ids = train["id"].to_numpy()
test_ids = test["id"].to_numpy()
print(f"data_dir={COMP_DIR}")
print(f"train {len(train):,}  test {len(test):,}  target rate {y.mean():.4f}")

# %% [markdown]
# ## 1. 扫描公开预测文件

# %%
n_oof = len(sorted(Path("/kaggle/input").rglob("oof_*.npy")))
n_npz = len(sorted(Path("/kaggle/input").rglob("*.npz")))
n_pq = len(sorted(Path("/kaggle/input").rglob("*.parquet")))
print(f"oof_*.npy={n_oof}  *.npz={n_npz}  *.parquet={n_pq}")

for p in sorted(Path("/kaggle/input").rglob("*.parquet")):
    try:
        df = pd.read_parquet(p)
        print(f"  {p.parent.name}/{p.name}: {df.shape}  first_cols={list(df.columns)[:6]}")
    except Exception as exc:
        print(f"  [warn] {p}: {type(exc).__name__}: {exc}")


# %% [markdown]
# ## 2. 装配成员矩阵
#
# 统一处理成 logit：概率在 [0,1] 内则转换，已在 logit 域（含负值）则原样裁剪到 ±30。
# 行对齐优先按 `id` 列，没有 `id` 时假定与 csv 行序一致。

# %%
def to_logit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    if np.nanmin(v) >= 0.0 and np.nanmax(v) <= 1.0:
        p = np.clip(v, 1e-15, 1 - 1e-15)
        return np.log(p / (1 - p))
    return np.clip(v, -30.0, 30.0)


members: dict[str, tuple[np.ndarray, np.ndarray]] = {}

# (a) npy 成对文件
for oof_path in sorted(Path("/kaggle/input").rglob("oof_*.npy")):
    name = oof_path.stem[len("oof_"):]
    test_path = oof_path.with_name(f"test_{name}.npy")
    if not test_path.exists():
        continue
    o, t = np.load(oof_path).ravel(), np.load(test_path).ravel()
    if len(o) != len(y) or len(t) != len(test):
        continue
    members[f"{oof_path.parent.name}:{name}"] = (to_logit(o), to_logit(t))

print(f"members from npy pairs: {len(members)}")

# (b) parquet 宽表：按列拆成成员，test 表按列名匹配
oof_tables, test_tables = [], []
for pq in sorted(Path("/kaggle/input").rglob("*.parquet")):
    try:
        df = pd.read_parquet(pq)
    except Exception:
        continue
    if len(df) == len(y):
        oof_tables.append((pq, df))
    elif len(df) == len(test):
        test_tables.append((pq, df))

print(f"parquet: {len(oof_tables)} train-shaped, {len(test_tables)} test-shaped")

for pq, df_o in oof_tables:
    aligned_o = df_o.set_index("id").reindex(train_ids) if "id" in df_o.columns else df_o
    for pq_t, df_t in test_tables:
        shared = [c for c in aligned_o.columns if c in df_t.columns]
        if len(shared) < max(1, int(0.5 * len(aligned_o.columns))):
            continue
        aligned_t = df_t.set_index("id").reindex(test_ids) if "id" in df_t.columns else df_t
        for col in shared:
            vo = pd.to_numeric(aligned_o[col], errors="coerce").to_numpy()
            vt = pd.to_numeric(aligned_t[col], errors="coerce").to_numpy()
            if not (np.isfinite(vo).all() and np.isfinite(vt).all()):
                continue
            members[f"{pq.stem}:{col}"] = (to_logit(vo), to_logit(vt))
        break

print(f"total members: {len(members)}")

rows = [{"member": n, "oof_auc": roc_auc_score(y, v[0])} for n, v in members.items()]
board = pd.DataFrame(rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
print("\nbest single members:")
print(board.head(12).to_string(index=False))
print(f"\nmember OOF AUC: min={board.oof_auc.min():.6f} median={board.oof_auc.median():.6f} max={board.oof_auc.max():.6f}")

# %% [markdown]
# ## 3. 5 折堆叠（LightGBM 二级模型）
#
# 折与成员模型一致（`seed=42`），因此 OOF 与成员可比。同时打印与「成员平均」的对比。

# %%
names = list(members)
X_oof = np.stack([members[n][0] for n in names], axis=1)
X_test = np.stack([members[n][1] for n in names], axis=1)
print(f"stack matrix: {X_oof.shape} / {X_test.shape}")

mean_oof = X_oof.mean(axis=1)
print(f"simple mean of members: OOF AUC = {roc_auc_score(y, mean_oof):.6f}")

params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=200,
    feature_fraction=0.7,
    bagging_fraction=0.9,
    bagging_freq=1,
    lambda_l2=5.0,
    n_jobs=-1,
    verbosity=-1,
    seed=SEED,
)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof = np.zeros(len(y))
pred = np.zeros(len(test))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_oof, y), start=1):
    model = lgb.train(
        params,
        lgb.Dataset(X_oof[tr_idx], y[tr_idx]),
        num_boost_round=1500,
        valid_sets=[lgb.Dataset(X_oof[va_idx], y[va_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    oof[va_idx] = model.predict(X_oof[va_idx], num_iteration=model.best_iteration)
    pred += model.predict(X_test, num_iteration=model.best_iteration) / N_SPLITS
    print(f"[fold {fold}] auc={roc_auc_score(y[va_idx], oof[va_idx]):.6f}  iter={model.best_iteration}")

stack_auc = roc_auc_score(y, oof)
print(f"\nstack OOF AUC = {stack_auc:.6f}")
print(f"best single   = {board.oof_auc.max():.6f} ({board.iloc[0]['member']})")

# %% [markdown]
# ## 4. 提交文件

# %%
sub = pd.DataFrame({"id": test_ids, TARGET: np.asarray(pred, dtype=float)})
if len(sub) == len(sample_sub):
    sub.to_csv(OUT / "submission.csv", index=False)
    print(f"wrote {OUT / 'submission.csv'}  rows={len(sub)}")
print(sub.head())

metrics = {
    "stack_oof_auc": stack_auc,
    "mean_member_oof_auc": float(roc_auc_score(y, mean_oof)),
    "best_single_oof_auc": float(board.oof_auc.max()),
    "best_single_member": str(board.iloc[0]["member"]),
    "n_members": len(names),
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
