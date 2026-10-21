# %% [markdown]
# # S6E10 公开 OOF 堆叠 v2
#
# v1（77 成员、`LogisticRegression(C=1.0)`）：OOF 0.961961 / LB 0.96167，已饱和。
# 读了榜上几份堆叠 notebook 后，v2 的改动：
#
# | 改动 | 来源 | 公开实测 |
# |---|---|---|
# | 成员池扩到 ~250（`cdeotte` 124 + TFM 11 + 之后公开的各库），近似重复按秩相关 > 0.9999 去掉 | `dariushafshar/s6e10-247-public-oofs-one-stack` | 嵌套 CV 0.96221 |
# | 强 L2：`C=0.003`（~250 个高度相关输入，默认 C=1 偏弱） | `yeashusemwal/secrets-to-satisfaction-it-s-stacking` | +0.00003 |
# | family-wise L2：TabPFN 列 ×3、其他基础模型 ×1.5（等于把它们的 L2 惩罚除以 9 / 2.25） | `goodpjw2008` Part 11 / `ewtherka` | +0.00001~2 |
# | 0.8 × LR + 0.2 × LightGBM 二级模型（各自标准化） | `yeashusemwal/trust-the-cv` | +0.00001 |
# | **排除 stack/blend 级成员**：已在全量标签上拟合过的堆叠结果会让嵌套 CV 虚高 | `ewtherka` 的完全嵌套复核 | — |
# | 自建成员：`jianansun/s6e10-fe-lightgbm`（FE + 航线画像 LightGBM，同一套 5 折） | 本仓库 | — |
#
# 二级模型统一在 `StratifiedKFold(5, shuffle=True, random_state=42)` 上做嵌套 CV（多数成员也用这套折）。
# 非基线配置只有在 5 折里至少 4 折胜过基线时才采用。

# %%
import glob
import json
import os
import re
import time
from collections import deque

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
COMP = "playground-series-s6e10"
TARGET = "satisfaction"
INPUT = "/kaggle/input"
OUT = "/kaggle/working"
N_JOBS = min(5, os.cpu_count() or 1)
SEED = 42


def find_dir(slug: str, root: str = INPUT, max_depth: int = 6) -> str | None:
    """挂载目录里名为 slug 的文件夹（数据集 / notebook 输出 / 比赛数据都适用），找不到返回 None。"""
    queue, hits = deque([(root, 0)]), []
    while queue:
        d, depth = queue.popleft()
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name == slug:
                    hits.append(e.path)
                elif depth < max_depth:
                    queue.append((e.path, depth + 1))
    return sorted(hits, key=len)[0] if hits else None


COMP_DIR = find_dir(COMP)
train = pd.read_csv(f"{COMP_DIR}/train.csv", usecols=["id", TARGET])
train_ids = train["id"].to_numpy()
test_ids = pd.read_csv(f"{COMP_DIR}/test.csv", usecols=["id"])["id"].to_numpy()
y = train[TARGET].astype(int).to_numpy()
NTR = len(y)

MF = np.zeros(NTR, int)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(NTR), y)):
    MF[va] = k
print(f"data_dir={COMP_DIR}  train {NTR:,}  test {len(test_ids):,}  target rate {y.mean():.4f}")

# %% [markdown]
# ## 1. 成员来源
#
# 每个来源一个 loader，产出 `(name, oof, test)`。任何来源缺失或格式变化都只跳过该来源，不中断。

# %%
SOURCES = []


def source(key, slug):
    def register(fn):
        SOURCES.append({"key": key, "slug": slug, "load": fn})
        return fn

    return register


def first(d, name):
    hits = sorted(glob.glob(os.path.join(d, "**", name), recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def by_id(df, ids, col):
    s = df.set_index("id")[col]
    assert s.index.is_unique, "duplicate ids"
    s = s.reindex(ids)
    assert s.notna().all(), f"{int(s.isna().sum()):,} ids missing"
    return s.to_numpy(np.float64)


def npy_pairs(d, prefix, pattern="oof_*.npy"):
    for f in sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True)):
        name = os.path.basename(f)[4:-4]
        yield prefix + name, np.load(f), np.load(os.path.join(os.path.dirname(f), f"test_{name}.npy"))


def npz_pairs(d, prefix, pattern="*.npz", strip=0):
    for f in sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True)):
        with np.load(f) as z:
            if "oof" in z.files and "test" in z.files:
                yield prefix + os.path.basename(f)[strip:-4], z["oof"], z["test"]


# ---- 基础池：Chris Deotte 打包的两个数据集
@source("c124", "s6e10-124-public-oof-and-test-predictions")
def load_c124(d):
    yield from npy_pairs(d, "")


@source("tfm", "s6e10-tfm-oof-and-test-predictions")
def load_tfm(d):
    yield from npy_pairs(d, "tfm_")


# ---- 之后公开的库
@source("golem", "s6e10-golem-oof-library")
def load_golem(d):
    man = pd.read_csv(f"{d}/manifest.csv")
    for r in man.itertuples():
        yield "gl_" + r.wide_column, np.load(f"{d}/{r.oof_file}"), np.load(f"{d}/{r.test_file}")


@source("abdullah", "oof-dataset")
def load_abdullah(d):
    for m in ("tabpfn_catfd10", "tabpfn_raw", "tabpfn_te_cnt", "cat_v3", "xgb_v3"):
        o, t = pd.read_csv(f"{d}/{m}_oof.csv"), pd.read_csv(f"{d}/{m}_test.csv")
        yield "ab_" + m, by_id(o, train_ids, "pred"), by_id(t, test_ids, "pred")


@source("sadam", "ps-s6e10-airline-satisfaction-artifacts")
def load_sadam(d):
    yield from npz_pairs(d, "st_", "preds_*.npz", strip=6)
    for n in ("orig_cat_pred", "orig_model_pred"):
        with np.load(first(d, f"{n}.npz")) as z:
            yield "st_" + n, z["train"], z["test"]


@source("forge", "s6e10-forge-own-pairs")
def load_forge(d):
    yield from npy_pairs(d, "am_", "oof_forge_*.npy")


@source("sachith", "s6e10-stack-oof-predictions")
def load_sachith(d):
    yield from npz_pairs(d, "s7_")


@source("koumeimaki", "s6e10-route-tfm-nn-oof")
def load_koumeimaki(d):
    assert (pd.read_csv(f"{d}/train_ids_folds.csv")["id"].to_numpy() == train_ids).all()
    assert (pd.read_csv(f"{d}/test_ids.csv")["id"].to_numpy() == test_ids).all()
    yield from npy_pairs(d, "km_")


@source("gp_tabpfn", "s6e10-tabpfn-member")
def load_gp_tabpfn(d):
    for f in sorted(glob.glob(f"{d}/oof_*.csv")):
        n = os.path.basename(f)[4:-4]
        yield "gpt_" + n, by_id(pd.read_csv(f), train_ids, "pred"), by_id(pd.read_csv(f"{d}/test_{n}.csv"), test_ids, "pred")


@source("gp_extra", "s6e10-goodpjw2008-extra-members-oof")
def load_gp_extra(d):
    yield from npy_pairs(d, "gpx_")


@source("dmitriy", "s6e10-catboost-all-categorical-oof-and-test")
def load_dmitriy(d):
    yield "dk_cb_cat", np.load(first(d, "oof_cb_cat.npy")), np.load(first(d, "test_cb_cat.npy"))


@source("kagan_abl", "s6e10-ablation-members-oof")
def load_kagan_abl(d):
    yield from npy_pairs(d, "kga_")


@source("megayak", "s6e10-oof-library")
def load_megayak(d):
    mo, mt = pd.read_parquet(f"{d}/oof_predictions.parquet"), pd.read_parquet(f"{d}/test_predictions.parquet")
    assert (mo["id"].to_numpy() == train_ids).all() and (mt["id"].to_numpy() == test_ids).all()
    groups = {}
    for c in mo.columns:
        if c in ("id", "fold", TARGET):
            continue
        groups.setdefault(c.rsplit("_s", 1)[0] if c.startswith("realmlp") else c, []).append(c)
    for g, cols in groups.items():
        yield "mg_" + g, mo[cols].mean(1).to_numpy(), mt[cols].mean(1).to_numpy()


@source("arhan", "s6e10-oof-test-predictions")
def load_arhan(d):
    ao, at = pd.read_parquet(f"{d}/oof_predictions.parquet"), pd.read_parquet(f"{d}/test_predictions.parquet")
    assert (ao["id"].to_numpy() == train_ids).all() and (at["id"].to_numpy() == test_ids).all()
    for c in ("lgb_5f", "xgb_5f", "cat_5f", "realmlp_5f"):
        yield "ar_" + c, ao[c].to_numpy(), at[c].to_numpy()


@source("busyaprime", "s6e10-busyaprime-five-members-oof")
def load_busyaprime(d):
    yield from npz_pairs(d, "", "bp_*.npz")


# ---- notebook 输出
@source("yekenot", "ps-s6-e10-realmlp-pytabkit")
def load_yekenot(d):
    o, t = pd.read_csv(first(d, "oof_preds.csv")), pd.read_csv(first(d, "submission.csv"))
    yield "yk_realmlp", by_id(o, train_ids, TARGET), by_id(t, test_ids, TARGET)


@source("hermengardo", "single-catboost-eda-ps6e10")
def load_hermengardo(d):
    yield "hm_cat", np.load(first(d, "oof_train.npy")), np.load(first(d, "oof_test.npy"))


@source("kratosyan", "s6e10-xgboost-realmlp-route-id-teacher")
def load_kratosyan(d):
    yield from npz_pairs(d, "kr_", "*_route_ogaux.npz")


@source("kagan_enc", "s6e10-encodings-catboost-realmlp-lb-0-961")
def load_kagan_enc(d):
    o, t = pd.read_csv(first(d, "oof_members.csv")), pd.read_csv(first(d, "test_members.csv"))
    for c in [c for c in o.columns if c != "id"]:
        yield "kg_" + c, by_id(o, train_ids, c), by_id(t, test_ids, c)


@source("aman_v5", "s6e10-the-original-prior-stack-v5")
def load_aman_v5(d):
    with np.load(first(d, "oof_artifacts.npz")) as z:
        assert (z["y"] == y).all(), "labels in a different row order"
        for m in ("lgbm_champion", "lgbm_te", "lgbm_te_rows", "xgb_gpu", "catboost_gpu"):
            yield "av5_" + m, z["oof_" + m], z["test_" + m]


@source("tabpfn_nb", "s6e10-tabpfn-route-og")
def load_tabpfn_nb(d):
    for nm, slug in [
        ("pfn_samanyu_route_og", "s6e10-tabpfn-route-og"),
        ("pfn_samanyu_seeds", "s6e10-tabpfn-seeds"),
        ("pfn_hemingweb_raw", "s6e10-exp01-tabpfn35-full-context"),
    ]:
        dd = find_dir(slug)
        if dd is None:
            continue
        yield (
            nm,
            by_id(pd.read_csv(first(dd, "oof_exp01_tabpfn35.csv")), train_ids, "pred"),
            by_id(pd.read_csv(first(dd, "test_exp01_tabpfn35.csv")), test_ids, TARGET),
        )


@source("mitudru", "s6e10-analysis-first-equality-blocks-stack")
def load_mitudru(d):
    mo = pd.read_parquet(first(d, "oof_predictions.parquet")).set_index("id").loc[train_ids]
    mt = pd.read_parquet(first(d, "test_predictions.parquet")).set_index("id").loc[test_ids]
    for c in mo.columns:
        if c in mt.columns:
            yield "mi_" + c, mo[c].to_numpy(), mt[c].to_numpy()


@source("wangxintong", "s6e10-catboost-rating-crosses-cv-0-96080")
def load_wangxintong(d):
    for slug, c in [
        ("s6e10-catboost-rating-crosses-cv-0-96080", "rating_segments"),
        ("s6e10-lightgbm-context-stats-cv-0-96065", "context_numeric"),
    ]:
        dd = find_dir(slug)
        if dd is None:
            continue
        yield (
            "wx_" + c,
            by_id(pd.read_parquet(first(dd, "oof_predictions.parquet")), train_ids, c),
            by_id(pd.read_parquet(first(dd, "test_predictions.parquet")), test_ids, c),
        )


@source("member_csv", "s6e10-xgb-realmlp-catboost-lb-0-96123")
def load_member_csv(d):
    for slug, tag in [("s6e10-xgb-realmlp-catboost-lb-0-96123", "th"), ("s6e10-honest-oof-stack", "mgh")]:
        dd = find_dir(slug)
        if dd is None:
            continue
        o = pd.read_csv(first(dd, "oof_members.csv")).set_index("id").loc[train_ids]
        t = pd.read_csv(first(dd, "test_members.csv")).set_index("id").loc[test_ids]
        for c in t.columns:
            if c in o.columns:
                yield f"{tag}_{c}", o[c].to_numpy(), t[c].to_numpy()


# ---- 自建成员（本仓库 FE LightGBM，同一套 seed-42 5 折）
@source("own_fe", "s6e10-fe-lightgbm")
def load_own_fe(d):
    yield "own_fe_lgbm", np.load(first(d, "fe_oof.npy")), np.load(first(d, "fe_test_pred.npy"))


print(f"{len(SOURCES)} sources registered")

# %% [markdown]
# ## 2. 装载与检查
#
# - 长度、有限值；概率统一转 logit（已在 logit 域的原样保留），裁剪到 ±30
# - **名字含 stack / blend / ensemble 的成员直接排除**：它们在全量标签上拟合过，会让嵌套 CV 虚高
# - OOF AUC 必须在 [0.90, 0.9625]：过低说明行序/标签不对，过高说明不是真 OOF
# - train / test 预测均值漂移 ≤ 0.03
# - 近似重复：OOF 秩相关 > 0.9999 的后来者丢弃

# %%
MIN_AUC, MAX_AUC, DRIFT, DUP_RHO = 0.90, 0.9625, 0.03, 0.9999
STACK_NAME = re.compile(r"stack|blend|ensemble|meta", re.I)


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


rows, OOFL, TSTL = [], [], []
for s in SOURCES:
    d = find_dir(s["slug"])
    if d is None:
        rows.append(dict(source=s["key"], member="(all)", status="not attached"))
        continue
    try:
        got = list(s["load"](d))
    except Exception as e:
        rows.append(dict(source=s["key"], member="(all)", status=f"skipped: {type(e).__name__}: {e}"[:120]))
        continue
    for name, o, t in got:
        o, t = np.asarray(o, np.float64).ravel(), np.asarray(t, np.float64).ravel()
        r = dict(source=s["key"], member=name, status="")
        if STACK_NAME.search(name):
            r["status"] = "stack-level member"
        elif o.shape != y.shape or t.shape != test_ids.shape:
            r["status"] = f"wrong length {o.shape[0]:,} / {t.shape[0]:,}"
        elif not (np.isfinite(o).all() and np.isfinite(t).all()):
            r["status"] = "NaN or inf"
        else:
            is_logit = bool(o.min() < 0 or o.max() > 1 or t.min() < 0 or t.max() > 1)
            p_o, p_t = (1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-t))) if is_logit else (o, t)
            zo, zt = (o, t) if is_logit else (logit(o), logit(t))
            r["drift"] = abs(p_t.mean() - p_o.mean())
            r["oof_auc"] = roc_auc_score(y, zo)
            if r["oof_auc"] < MIN_AUC:
                r["status"] = f"OOF AUC {r['oof_auc']:.3f} too low"
            elif r["oof_auc"] > MAX_AUC:
                r["status"] = f"OOF AUC {r['oof_auc']:.4f} too high"
            elif r["drift"] > DRIFT:
                r["status"] = f"drift {r['drift']:.3f}"
            else:
                OOFL.append(np.clip(zo, -30, 30).astype(np.float32))
                TSTL.append(np.clip(zt, -30, 30).astype(np.float32))
                r["col"] = len(OOFL) - 1
        rows.append(r)

CHK = pd.DataFrame(rows)
CHK["col"] = CHK.get("col", pd.Series(dtype=float)).fillna(-1).astype(int)
Zall, Tall = np.column_stack(OOFL), np.column_stack(TSTL)
del OOFL, TSTL

# 近似重复：按装载顺序贪心保留
Rk = np.column_stack([rankdata(Zall[:, c]) for c in range(Zall.shape[1])])
Rk -= Rk.mean(0)
Rk /= np.sqrt((Rk**2).mean(0))
RHO = np.clip((Rk.T @ Rk) / NTR, -1, 1)
del Rk
kept = []
for j in range(Zall.shape[1]):
    if kept and RHO[j, kept].max() > DUP_RHO:
        i = CHK.index[CHK["col"] == j][0]
        CHK.loc[i, "status"] = f"near-copy (rho {RHO[j, kept].max():.6f})"
    else:
        kept.append(j)
CHK["in_pool"] = CHK["col"].isin(kept)
NAMES = [CHK.loc[CHK["col"] == c, "member"].iloc[0] for c in kept]
SRCS = [CHK.loc[CHK["col"] == c, "source"].iloc[0] for c in kept]
Z, ZT = Zall[:, kept].astype(np.float64), Tall[:, kept].astype(np.float64)
del Zall, Tall, RHO

tab = (
    CHK[CHK["member"] != "(all)"]
    .groupby("source", sort=False)
    .agg(loaded=("member", "size"), kept=("in_pool", "sum"), best_oof_auc=("oof_auc", "max"))
)
print(tab.to_string(formatters={"best_oof_auc": "{:.6f}".format}))
skipped = CHK[CHK["member"] == "(all)"]
if len(skipped):
    print("\nsources not loaded:\n" + skipped[["source", "status"]].to_string(index=False))
reason = CHK.loc[(CHK["member"] != "(all)") & ~CHK["in_pool"], "status"].str.replace(r" \(.*| [\d.]+.*", "", regex=True)
print("\ndropped by reason:", reason.value_counts().to_dict())
print(f"pool: {len(NAMES)} members   [{time.time() - T0:.0f}s]")
CHK.drop(columns=["col"]).to_csv(f"{OUT}/member_check.csv", index=False)

OWN = [i for i, s in enumerate(SRCS) if s.startswith("own_")]
if OWN:
    for i in OWN:
        print(f"own member {NAMES[i]}: OOF AUC {roc_auc_score(y, Z[:, i]):.6f}")
else:
    print("[warn] own member not in pool")

# %% [markdown]
# ## 3. 二级模型
#
# - LR：logit 输入，`C` 很小（强 L2）；family 方案先按族把列放大再用同一个 `C`
# - LightGBM 二级：浅树 + 大叶子样本数，只做补充
# - 融合：两者各自按 OOF 标准化后 0.8 / 0.2 加权

# %%
def family(n):
    if "tabpfn" in n or n.startswith("pfn_"):
        return "tabpfn"
    if n.startswith("tfm_") or "tabicl" in n or "tabm" in n:
        return "tfm"
    if "realmlp" in n or "rmlp" in n or "mlp" in n:
        return "mlp"
    return "gbdt"


FAM = np.array([family(n) for n in NAMES])
SCALE = np.array([{"tabpfn": 3.0, "tfm": 1.5}.get(f, 1.0) for f in FAM])
print({f: int((FAM == f).sum()) for f in ("gbdt", "mlp", "tabpfn", "tfm")})


def fold_auc(o):
    return np.array([roc_auc_score(y[MF == k], o[MF == k]) for k in range(5)])


def _lr(Zm, k, C):
    a, b = MF != k, MF == k
    m = LogisticRegression(C=C, max_iter=5000).fit(Zm[a], y[a])
    return b, m.decision_function(Zm[b])


def nested_lr(Zm, C):
    o = np.zeros(NTR)
    for b, v in Parallel(n_jobs=N_JOBS)(delayed(_lr)(Zm, k, C) for k in range(5)):
        o[b] = v
    return o


GB_P = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=15,
    min_data_in_leaf=2000,
    feature_fraction=0.5,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    num_threads=4,
    seed=0,
)


def fit_gb(A, ya):
    ai = np.random.default_rng(0).permutation(len(A))
    cut = int(len(ai) * 0.9)
    return lgb.train(
        GB_P,
        lgb.Dataset(A[ai[:cut]], ya[ai[:cut]]),
        3000,
        valid_sets=[lgb.Dataset(A[ai[cut:]], ya[ai[cut:]])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def nested_gb(A):
    o = np.zeros(NTR)
    for k in range(5):
        a, b = MF != k, MF == k
        m = fit_gb(A[a], y[a])
        o[b] = m.predict(A[b], raw_score=True, num_iteration=m.best_iteration)
    return o


def zs(v):
    return (v - v.mean()) / v.std()


W_GB = 0.2
X = {"flat": (Z, ZT), "family": (Z * SCALE, ZT * SCALE)}
LR_SETTINGS = [("flat", 0.003), ("flat", 0.002), ("family", 0.003), ("family", 0.005)]
BASELINE = ("lr", "flat", 0.003)

t = time.time()
GB_OOF = nested_gb(Z)
print(f"LightGBM stacker: fold-mean {fold_auc(GB_OOF).mean():.6f}  ({time.time() - t:.0f}s)", flush=True)

LR_OOF, RES = {}, {}
for sch, C in LR_SETTINGS:
    t = time.time()
    LR_OOF[sch, C] = nested_lr(X[sch][0], C)
    RES["lr", sch, C] = fold_auc(LR_OOF[sch, C])
    RES["blend", sch, C] = fold_auc((1 - W_GB) * zs(LR_OOF[sch, C]) + W_GB * zs(GB_OOF))
    print(
        f"{sch:6s} C={C:<6g} LR {RES['lr', sch, C].mean():.6f} | blend {RES['blend', sch, C].mean():.6f}"
        f"  ({time.time() - t:.0f}s)",
        flush=True,
    )

# 自建成员的边际贡献：去掉它们后跑一次基线
own_delta = None
if OWN:
    keep_cols = [i for i in range(Z.shape[1]) if i not in OWN]
    no_own = fold_auc(nested_lr(Z[:, keep_cols], BASELINE[2]))
    own_delta = float((RES[BASELINE] - no_own).mean())
    print(f"baseline without own members: {no_own.mean():.6f}  -> own members add {own_delta * 1e6:+.1f} x 1e-6")

# %% [markdown]
# ## 4. 选择
#
# 基线是 flat C=0.003 的纯 LR。其他配置按 fold-mean 排序，最优者必须在 ≥4/5 折上胜过基线才采用，
# 否则用基线（没有额外调参，最不容易对 OOF 过拟合）。

# %%
base = RES[BASELINE]
board = pd.DataFrame(
    [
        dict(stacker=k[0], scheme=k[1], C=k[2], fold_mean=v.mean(), gain_x1e6=(v - base).mean() * 1e6, folds_up=int((v > base).sum()))
        for k, v in RES.items()
    ]
).sort_values("fold_mean", ascending=False)
print(board.to_string(index=False, formatters={"fold_mean": "{:.6f}".format, "gain_x1e6": "{:+.1f}".format}))

best = max(RES, key=lambda k: RES[k].mean())
FINAL = best if best != BASELINE and (RES[best] > base).sum() >= 4 else BASELINE
print(f"\nbest by fold-mean: {best}  ->  final: {FINAL}")

# %% [markdown]
# ## 5. 全量重拟合与提交

# %%
def sig(v):
    return 1 / (1 + np.exp(-v))


kind, sch, C = FINAL
lr_full = LogisticRegression(C=C, max_iter=5000).fit(X[sch][0], y).decision_function(X[sch][1])
if kind == "lr":
    test_fin, oof_fin = lr_full, LR_OOF[sch, C]
else:
    g = fit_gb(Z, y)
    gb_full = g.predict(ZT, raw_score=True, num_iteration=g.best_iteration)
    lo = LR_OOF[sch, C]
    test_fin = (1 - W_GB) * (lr_full - lo.mean()) / lo.std() + W_GB * (gb_full - GB_OOF.mean()) / GB_OOF.std()
    oof_fin = (1 - W_GB) * zs(lo) + W_GB * zs(GB_OOF)

sub = pd.DataFrame({"id": test_ids, TARGET: sig(test_fin)})
assert len(sub) == len(test_ids) and sub[TARGET].between(0, 1).all()
sub.to_csv(f"{OUT}/submission.csv", index=False)
np.save(f"{OUT}/stack_oof.npy", oof_fin.astype(np.float32))
np.save(f"{OUT}/stack_test.npy", test_fin.astype(np.float32))
board.to_csv(f"{OUT}/stacker_board.csv", index=False)

metrics = {
    "final": list(FINAL),
    "final_fold_mean_auc": float(RES[FINAL].mean()),
    "final_pooled_oof_auc": float(roc_auc_score(y, oof_fin)),
    "baseline_fold_mean_auc": float(base.mean()),
    "n_members": len(NAMES),
    "families": {f: int((FAM == f).sum()) for f in ("gbdt", "mlp", "tabpfn", "tfm")},
    "own_members_gain": own_delta,
    "minutes": round((time.time() - T0) / 60, 1),
}
(open(f"{OUT}/metrics.json", "w")).write(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
