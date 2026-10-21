# %% [markdown]
# # S6E10 特征工程版 LightGBM（对照实验）
#
# 目标是验证公开 notebook 里那套增量特征到底能不能补上 0.002 的差距。相对主线模型的差别：
#
# | 增量 | 来源 | 作者实测增益 |
# |---|---|---|
# | digit 特征（`(x // 10**k) % 10`，k=-4..3） | `evgendvorkin/s6e10-lightgbm-cv-5-folds-0-96111` | OOF +0.00069 |
# | 频次编码 + rarity（`-log(freq)`） | 同上 | OOF +0.00038 |
# | 目标编码 × 3 种平滑（auto/10/100），折内拟合 | 同上 | OOF +0.00038 |
# | 原始数据集训练的模型预测值作为特征 | 同上 | — |
# | 服务评分聚合（均值/离散度/极值/0 与 5 计数） | 本仓库主线模型 | — |
# | 原始数据按精确键元组查 target rate + 支持度 | 同上（升级版） | — |
# | 航线画像：精确距离（× 舱位）上的评分均值与个人偏差 | `kozykappa/s6e10-route-profile-cross-fitted-target` | — |
# | 航线 ID / 评分 × 乘客分段的折内目标编码 | 同上 | — |
# | v4：完整航线画像（同一距离下全部原始列的均值 + 行数） | `goodpjw2008/s6e10-auxiliary-task-features` | 单独即达到航线 TE 水平 |
# | v4：辅助任务特征（用其余 20 列预测每个评分的期望值 + 整行伪对数似然，不用标签） | 同上 | XGB +0.00013 |
# | v4：LightGBM 改为更强正则（127 叶、`min_child=50`、`ff=0.5`） | 同上 | — |
#
# 单一 LightGBM、5 折，不做融合，方便干净地读出「特征」本身的收益。

# %%
import json
import shutil
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings("ignore")

COMP = "playground-series-s6e10"
SEED = 42
RUN_MODE = "full"          # "full" | "smoke"
N_SPLITS = 5
SMOKE_ROWS = 50_000
USE_ORIGINAL = True
USE_AUX = True             # 辅助任务特征（16 个多分类 XGBoost × AUX_FOLDS 折，有 GPU 时用 cuda）
AUX_FOLDS = 3

TARGET = "satisfaction"
ID = "id"

IS_KAGGLE = Path("/kaggle/input").exists()
OUT_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_dir() -> Path:
    if not IS_KAGGLE:
        return Path("data")
    for cand in [Path(f"/kaggle/input/{COMP}"), Path(f"/kaggle/input/competitions/{COMP}")]:
        if (cand / "train.csv").exists():
            return cand
    for t in Path("/kaggle/input").rglob("train.csv"):
        return t.parent
    raise FileNotFoundError("competition train.csv not found")


def find_orig() -> Path | None:
    if not IS_KAGGLE:
        p = Path("data/orig/train.csv")
        return p if p.exists() else None
    for p in Path("/kaggle/input").rglob("train.csv"):
        try:
            cols = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception:
            continue
        if TARGET in cols and "Unnamed: 0" in cols:
            return p
    return None


DATA_DIR = resolve_dir()
ORIG_PATH = find_orig() if USE_ORIGINAL else None
print(f"run_mode={RUN_MODE}  data_dir={DATA_DIR}  orig={ORIG_PATH}")

# %% [markdown]
# ## 1. 基础预处理
#
# 与公开 notebook 保持一致：`Arrival Delay` 缺失填 0（后续 digit 与 freq 都基于此），
# 4 个类别列转 category，`id`/target 单独拆出。

# %%
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
if RUN_MODE == "smoke":
    train = train.sample(SMOKE_ROWS, random_state=SEED).reset_index(drop=True)
    test = test.head(SMOKE_ROWS // 2).reset_index(drop=True)

y = train[TARGET].astype(int)
train_id = train[ID].to_numpy()
test_id = test[ID].to_numpy()
train = train.drop(columns=[TARGET, ID])
test = test.drop(columns=[ID])

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
for df in (train, test):
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)
for col in CAT_COLS:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

RAW_COLS = list(train.columns)
print(f"base train={train.shape} test={test.shape}  target rate={y.mean():.4f}")

# %% [markdown]
# ## 2. digit 特征
#
# 把每个数值列的十进制各位单独拆出来（k=-4..3）。合成数据的数值分布会在低位数字上留下
# 生成器痕迹，作者把这些列排进了特征重要性前列。

# %%
DIGIT_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
DIGITS = range(-4, 4)
for col in DIGIT_COLS:
    for k in DIGITS:
        name = f"{col}_digit{k}"
        for df in (train, test):
            df[name] = (df[col].fillna(0) // (10**k) % 10).astype("int8")
print(f"digit features: {len(DIGIT_COLS) * len(list(DIGITS))}  -> train={train.shape}")

# %% [markdown]
# ## 服务评分聚合
#
# 13 个 0–5 评分是同一组「体验」维度，整体水平与离散度本身有信号（主线模型已验证过）。

# %%
SERVICE_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]
for df in (train, test):
    s = df[SERVICE_COLS]
    df["service_mean"] = s.mean(axis=1).astype("float32")
    df["service_std"] = s.std(axis=1).astype("float32")
    df["service_min"] = s.min(axis=1).astype("int8")
    df["service_max"] = s.max(axis=1).astype("int8")
    df["service_sum"] = s.sum(axis=1).astype("int16")
    df["service_n_zero"] = (s == 0).sum(axis=1).astype("int8")
    df["service_n_five"] = (s == 5).sum(axis=1).astype("int8")
print(f"service aggregates added -> train={train.shape}")

# %% [markdown]
# ## 3. 频次编码 + rarity
#
# 频次用 train+test 合并统计（无标签，不算泄漏）；rarity = `-log(freq)`。

# %%
FREQ_COLS = [c for c in train.columns if c != ID]
for col in FREQ_COLS:
    vc = pd.concat([train[col], test[col]], axis=0).value_counts(normalize=True)
    for df in (train, test):
        freq = df[col].map(vc).astype("float32")
        df[f"{col}_freq"] = freq.to_numpy()
        df[f"{col}_rarity"] = (-np.log(freq.clip(lower=1.0 / len(train)))).to_numpy()
print(f"freq/rarity columns: {2 * len(FREQ_COLS)}  -> train={train.shape}")

# %% [markdown]
# ## 航线画像（route profile）
#
# 公开方案（`kozykappa/s6e10-route-profile-cross-fitted-target`、`kratosyan` 的 route-ID 系列）发现：
# 合成数据里**精确的 `Flight Distance` 近似一条航线 ID**。同一距离（及距离 × 舱位）下，
# 关键评分的均值描述了「这条航线通常的服务水平」，个人评分减去航线均值则是「相对体验」。
# 只用特征、不用标签，train+test 合并统计。

# %%
ROUTE_COLS = [
    "Online boarding",
    "Inflight wifi service",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "service_mean",
]
_all = pd.concat([train, test], axis=0, ignore_index=True)
for key_cols, suffix in [(["Flight Distance"], "route"), (["Flight Distance", "Class"], "route_class")]:
    by = _all.groupby(key_cols, observed=True, dropna=False)
    cnt = by["service_mean"].transform("size").astype("float32").to_numpy()
    feats = {f"{suffix}_count_log": np.log1p(cnt)}
    for col in ROUTE_COLS:
        if suffix == "route_class" and col not in ("Online boarding", "service_mean"):
            continue
        mu = by[col].transform("mean").astype("float32").to_numpy()
        feats[f"{suffix}_{col}_mean"] = mu
        feats[f"{suffix}_{col}_delta"] = _all[col].to_numpy(dtype="float32") - mu
    for name, v in feats.items():
        train[name] = v[: len(train)].astype("float32")
        test[name] = v[len(train):].astype("float32")

# 完整航线画像：同一距离下其余每个原始列的均值（类别列用整数编码），以及该距离的行数
_coded = _all[RAW_COLS].copy()
for col in CAT_COLS:
    _coded[col] = _coded[col].astype("category").cat.codes
grp = _coded.groupby("Flight Distance")
fdm = {f"fdm_{c}": grp[c].transform("mean").to_numpy(dtype="float32") for c in RAW_COLS if c != "Flight Distance"}
fdm["fd_cnt"] = grp["Age"].transform("size").to_numpy(dtype="float32")
for name, v in fdm.items():
    train[name] = v[: len(train)]
    test[name] = v[len(train):]
del _all, grp
print(f"route profile added -> train={train.shape}")

# %% [markdown]
# ## 辅助任务特征
#
# 来自 `goodpjw2008/s6e10-auxiliary-task-features`：对 13 个评分与 3 个类别列，用其余 20 列训练多分类 XGBoost
# （train+test 合并、K 折，每行由没见过它的模型预测），保留：
#
# - `aux_ev_<评分>`：该评分的期望值 Σ k·P(k)，「根据其他回答，这位乘客预期会打几分」
# - `aux_sum_logp`：16 列实际取值概率的对数和，整行的伪对数似然（「这一行有多典型」）
#
# 不用标签，所以和计数、航线画像一样只算一次，不必放进 CV 折内。作者实测期望值单模型 +0.0001；
# 单列实际取值概率单独用反而有害，这里不加。

# %%
if USE_AUX:
    from sklearn.model_selection import KFold

    AUX_DEVICE = "cuda" if shutil.which("nvidia-smi") else "cpu"
    AUX_ROUNDS = 300 if RUN_MODE == "full" else 30
    AUX_TARGETS = SERVICE_COLS + ["Class", "Type of Travel", "Customer Type"]
    _coded = pd.concat([train[RAW_COLS], test[RAW_COLS]], axis=0, ignore_index=True)
    for col in CAT_COLS:
        _coded[col] = _coded[col].astype("category").cat.codes
    aux, logp = {}, np.zeros(len(_coded), dtype=np.float64)
    t0 = time.time()
    for c in AUX_TARGETS:
        yc, uniq = pd.factorize(_coded[c], sort=True)
        k = len(uniq)
        others = [f for f in RAW_COLS if f != c]
        P = np.zeros((len(_coded), k), dtype=np.float32)
        for a, b in KFold(AUX_FOLDS, shuffle=True, random_state=0).split(_coded):
            m = xgb.XGBClassifier(
                n_estimators=AUX_ROUNDS,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.7,
                tree_method="hist",
                device=AUX_DEVICE,
                random_state=0,
                objective="multi:softprob" if k > 2 else "binary:logistic",
            )
            m.fit(_coded.iloc[a][others], yc[a])
            P[b] = m.predict_proba(_coded.iloc[b][others])
        logp += np.log(np.clip(P[np.arange(len(_coded)), yc], 1e-6, 1))
        if c in SERVICE_COLS:
            aux[f"aux_ev_{c}"] = P @ np.asarray(uniq, dtype=np.float32)
        print(f"  aux {c:<34} k={k}  acc={(P.argmax(1) == yc).mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    aux["aux_sum_logp"] = logp.astype(np.float32)
    for name, v in aux.items():
        train[name] = np.asarray(v[: len(train)], dtype="float32")
        test[name] = np.asarray(v[len(train):], dtype="float32")
    del _coded
    print(f"aux features added: {len(aux)}  device={AUX_DEVICE}  -> train={train.shape}")

# %% [markdown]
# ## 4. 目标编码的键
#
# 除了 4 个原始类别列，还构造分箱键（年龄/距离/延误）与交叉键。目标编码本身在每折内拟合，
# 因此这些键的构造可以放心用到全量数据。

# %%
def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_5"] = (out["Age"] // 5).astype("int32")
    out["distance_100"] = (out["Flight Distance"] // 100).astype("int32")
    out["distance_250"] = (out["Flight Distance"] // 250).astype("int32")
    out["distance_500"] = (out["Flight Distance"] // 500).astype("int32")
    out["distance_div10"] = (out["Flight Distance"] // 10).astype("int32")
    out["distance_div100"] = (out["Flight Distance"] // 100).astype("int32")
    out["distance_mod1000"] = (out["Flight Distance"] % 1000).astype("int32")
    out["dep_delay_10"] = (out["Departure Delay in Minutes"] // 10).astype("int32")
    out["arr_delay_10"] = (out["Arrival Delay in Minutes"] // 10).astype("int32")
    out["class_travel"] = out["Class"].astype(str) + "_" + out["Type of Travel"].astype(str)
    out["customer_class"] = out["Customer Type"].astype(str) + "_" + out["Class"].astype(str)
    out["gender_class"] = out["Gender"].astype(str) + "_" + out["Class"].astype(str)
    out["class_board"] = out["Class"].astype(str) + "_" + out["Online boarding"].astype(str)
    out["distance_travel"] = out["distance_500"].astype(str) + "_" + out["Type of Travel"].astype(str)
    # 航线 ID 与「关键评分 × 乘客分段」的条件键（route-profile 方案）
    segment = (
        out["Type of Travel"].astype(str) + "_" + out["Customer Type"].astype(str) + "_" + out["Class"].astype(str)
    )
    out["passenger_segment"] = segment
    out["route_id"] = out["Flight Distance"].astype("int32")
    out["route_class"] = out["Flight Distance"].astype(str) + "_" + out["Class"].astype(str)
    out["route_travel"] = out["Flight Distance"].astype(str) + "_" + out["Type of Travel"].astype(str)
    for col, short in [
        ("Online boarding", "boarding"),
        ("Inflight wifi service", "wifi"),
        ("Inflight entertainment", "entertainment"),
        ("Seat comfort", "seat"),
    ]:
        out[f"{short}_segment"] = out[col].astype(str) + "_" + segment
    out["age_segment"] = out["age_5"].astype(str) + "_" + segment
    return out


te_train = add_keys(train)
te_test = add_keys(test)
TE_COLS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
    "age_5",
    "distance_100",
    "distance_250",
    "distance_500",
    "distance_div10",
    "distance_div100",
    "distance_mod1000",
    "dep_delay_10",
    "arr_delay_10",
    "class_travel",
    "customer_class",
    "gender_class",
    "class_board",
    "distance_travel",
    "passenger_segment",
    "route_id",
    "route_class",
    "route_travel",
    "boarding_segment",
    "wifi_segment",
    "entertainment_segment",
    "seat_segment",
    "age_segment",
]
SMOOTHINGS = [("auto", "auto"), ("s10", 10.0), ("s100", 100.0)]
print(f"TE keys: {len(TE_COLS)} x smoothings {len(SMOOTHINGS)} = {len(TE_COLS) * len(SMOOTHINGS)} features")

# %% [markdown]
# ## 5. 原始数据集：单独训练的模型预测值
#
# 只在 129,880 行原始数据上训一个 LightGBM，把它的预测概率当成 train/test 的一列特征。
# 这是「把原始数据当先验」而不是「当额外样本」，与主线 v2 里直接拼接行不同。

# %%
if ORIG_PATH is not None:
    orig = pd.read_csv(ORIG_PATH)
    orig = orig.drop(columns=[c for c in orig.columns if c.startswith("Unnamed")], errors="ignore")
    orig[TARGET] = orig[TARGET].astype(str).str.strip().str.lower().eq("satisfied").astype(int)
    orig_feat = [c for c in train.columns if c in orig.columns and not c.endswith(("_freq", "_rarity"))]
    orig_feat = [c for c in orig_feat if "digit" not in c]
    orig_X = orig[orig_feat].copy()
    for col in CAT_COLS:
        orig_X[col] = orig_X[col].astype("category")
    orig_model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=5.0,
        reg_alpha=0.05,
        random_state=123,
        n_jobs=-1,
        verbose=-1,
    )
    orig_model.fit(orig_X, orig[TARGET])
    train["orig_pred"] = orig_model.predict_proba(train[orig_feat])[:, 1].astype("float32")
    test["orig_pred"] = orig_model.predict_proba(test[orig_feat])[:, 1].astype("float32")
    print(f"orig-only model AUC on original: {roc_auc_score(orig[TARGET], orig_model.predict_proba(orig_X)[:, 1]):.6f}")
    print(f"orig_pred added; train cols={train.shape[1]}")
else:
    print("[warn] original dataset not found, skipping orig_pred")

# %% [markdown]
# ## 原始数据的精确键查询
#
# 对若干键元组（类别交叉、关键评分组合、13 个评分全组合），在原始数据上统计 target rate 与
# 支持度，再映射回比赛数据。用的是原始标签、不是合成标签，因此不构成对验证折的泄漏；
# rate 做 alpha 平滑，避免「只命中 1 条原始记录」时退化成原始标签本身。

# %%
ORIG_KEY_SPECS = {
    "class_travel": ["Class", "Type of Travel"],
    "customer_class": ["Customer Type", "Class"],
    "travel_gender": ["Type of Travel", "Gender"],
    "board_wifi": ["Online boarding", "Inflight wifi service"],
    "seat_clean": ["Seat comfort", "Cleanliness"],
    "ent_comfort": ["Inflight entertainment", "Seat comfort"],
    "rating13": SERVICE_COLS,
    "rating13_seg": SERVICE_COLS + CAT_COLS,
}
LOOKUP_ALPHA = 20.0


def _frame_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        out = out.str.cat(df[c].astype(str), sep="|")
    return out


if ORIG_PATH is not None:
    prior = float(orig[TARGET].mean())
    added_lookup = []
    for name, cols in ORIG_KEY_SPECS.items():
        g = orig.assign(_k=_frame_key(orig, cols)).groupby("_k")[TARGET].agg(["sum", "count"])
        rate_name = f"orig_{name}_rate"
        cnt_name = f"orig_{name}_cnt"
        rate_map = ((g["sum"] + prior * LOOKUP_ALPHA) / (g["count"] + LOOKUP_ALPHA)).astype("float32")
        cnt_map = np.log1p(g["count"]).astype("float32")
        for df in (train, test):
            k = _frame_key(df, cols)
            df[rate_name] = k.map(rate_map).fillna(prior).astype("float32").to_numpy()
            df[cnt_name] = k.map(cnt_map).fillna(0.0).astype("float32").to_numpy()
        added_lookup += [rate_name, cnt_name]
        print(f"  {name:<16} keys={len(rate_map):>7,}  coverage={train[rate_name].ne(prior).mean():.3f}")
    print(f"orig lookups added: {len(added_lookup)}  -> train={train.shape}")
else:
    print("[warn] original dataset not found, skipping lookups")

# %% [markdown]
# ## 6. 5 折 CV

# %%
BASE_FEATURES = [c for c in train.columns]
LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    learning_rate=0.02,
    num_leaves=127,
    min_child_samples=50,
    feature_fraction=0.5,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    n_jobs=-1,
    verbosity=-1,
)
ROUNDS = 3000 if RUN_MODE == "full" else 200

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
pred = np.zeros(len(test))
gain = pd.Series(0.0, index=BASE_FEATURES)
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), start=1):
    X_tr = train.iloc[tr_idx].reset_index(drop=True)
    X_va = train.iloc[va_idx].reset_index(drop=True)
    X_te = test.reset_index(drop=True)

    # 折内目标编码：3 种平滑各算一套，训练集用 fit_transform（内部 CV 防泄漏）
    for smooth_name, smooth_val in SMOOTHINGS:
        enc = TargetEncoder(cv=5, smooth=smooth_val, shuffle=True, random_state=SEED)
        tr_enc = enc.fit_transform(te_train.iloc[tr_idx][TE_COLS], y.iloc[tr_idx])
        va_enc = enc.transform(te_train.iloc[va_idx][TE_COLS])
        te_enc = enc.transform(te_test[TE_COLS])
        for j, col in enumerate(TE_COLS):
            X_tr[f"TE_{col}_{smooth_name}"] = tr_enc[:, j]
            X_va[f"TE_{col}_{smooth_name}"] = va_enc[:, j]
            X_te[f"TE_{col}_{smooth_name}"] = te_enc[:, j]

    dtrain = lgb.Dataset(X_tr, y.iloc[tr_idx])
    dvalid = lgb.Dataset(X_va, y.iloc[va_idx], reference=dtrain)
    model = lgb.train(
        {**LGB_PARAMS, "seed": SEED, "bagging_seed": SEED, "feature_fraction_seed": SEED},
        dtrain,
        num_boost_round=ROUNDS,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )

    oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    pred += model.predict(X_te, num_iteration=model.best_iteration) / N_SPLITS
    # TE 列只在折内存在，用 add(fill_value=0) 累积，避免按索引对齐时被丢掉
    gain = gain.add(pd.Series(model.feature_importance("gain"), index=X_tr.columns), fill_value=0.0)
    print(
        f"[fold {fold}] best_iter={model.best_iteration:4d}  "
        f"auc={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}  "
        f"n_features={X_tr.shape[1]}  ({time.time() - t0:.0f}s)"
    )

auc = roc_auc_score(y, oof)
print(f"\nOOF AUC = {auc:.6f}")

# %% [markdown]
# ## 7. 结果与提交

# %%
print("top 25 features by gain:")
print(gain.sort_values(ascending=False).head(25).round(1).to_string())

submission = pd.DataFrame({ID: test_id, TARGET: np.asarray(pred, dtype=float)})
if RUN_MODE == "full":
    assert len(submission) == len(sample_sub), (len(submission), len(sample_sub))
assert submission[ID].is_unique and submission[TARGET].between(0, 1).all()
submission.to_csv(OUT_DIR / "submission.csv", index=False)

# 保存 OOF / test 预测，方便后续与主线模型做 stacking
np.save(OUT_DIR / "fe_oof.npy", oof)
np.save(OUT_DIR / "fe_test_pred.npy", pred)

metrics = {
    "oof_auc": auc,
    "n_features": int(gain.size),
    "n_splits": N_SPLITS,
    "n_train": int(len(train)),
    "use_original": bool(ORIG_PATH is not None),
}
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))

# %% [markdown]
# ## 8. 下一步
#
# - 特征这套已经跑通（v1 OOF 0.96030 / LB 0.95986），继续加：10 折、多 seed
# - 与主线 10 折集成（`lgb_a`/`lgb_b`/`nn`）做 stacking——两套特征空间差异大，融合价值高
# - 公开 OOF 库（`sachith7`、`megayak`、`cdeotte`）可做二级 stacking，公开榜同类做法到 0.9618–0.9620
# - 公开 notebook 里 0.96207 那类「对公开 LB 做曲面拟合」的分数在 private 上大概率不成立，不作为参考
