# %% [markdown]
# # Predicting Airline Satisfaction — playground-series-s6e10
#
# 二分类：预测乘客是否满意（`satisfaction`），提交概率，按 ROC AUC 评估。
#
# 流程：
# 1. 读取比赛数据（+ 可选：原始 Invistico 航空公司满意度数据集作为补充训练样本）
# 2. 特征工程：服务评分聚合、延误结构、类别交叉、分箱
# 3. LightGBM / XGBoost 双模型 5 折 CV（同一套折划分；CatBoost 默认关闭，见 `USE_CATBOOST`）
# 4. 按 OOF AUC 在概率单纯形上搜索融合权重
# 5. 输出 `submission.csv` + `metrics.json`
#
# 训练在 Kaggle 上跑；本地只做 smoke 验证（`RUN_MODE = "smoke"`）。

# %%
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- config ----
COMP = "playground-series-s6e10"
SEED = 42
RUN_MODE = "full"          # "full" = 全部数据 + 5 折；"smoke" = 抽样 + 2 折，仅验证流程
# 以下两个开关默认关闭，原因见 README「实验记录」：
# 原始数据集（teejmahal20/airline-passenger-satisfaction）经诊断对合成验证集无增益（+0.00003）
USE_ORIGINAL_DATA = False
# CatBoost 5 折要跑 2.1 小时，且 OOF 比 LightGBM 低 0.0003，融合只 +0.0001
USE_CATBOOST = False
N_SPLITS = 5 if RUN_MODE == "full" else 2
SMOKE_ROWS = 50_000

TARGET = "satisfaction"
ID = "id"

IS_KAGGLE = Path("/kaggle/input").exists()
OUT_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_competition_dir() -> Path:
    """定位比赛数据目录。

    新运行环境挂在 /kaggle/input/competitions/<slug>/，老环境是 /kaggle/input/<slug>/，
    两种都兼容；都找不到时递归扫描并给出诊断信息。
    """
    if not IS_KAGGLE:
        return Path("data")
    candidates = [
        Path(f"/kaggle/input/{COMP}"),
        Path(f"/kaggle/input/competitions/{COMP}"),
    ]
    candidates += sorted(Path("/kaggle/input").glob(f"*/{COMP}"))
    for candidate in candidates:
        if (candidate / "train.csv").exists():
            return candidate
    for train_path in Path("/kaggle/input").rglob("train.csv"):
        print(f"[warn] falling back to {train_path.parent}")
        return train_path.parent
    listing = sorted(str(p.relative_to("/kaggle/input")) for p in Path("/kaggle/input").glob("*"))
    raise FileNotFoundError(
        f"train.csv not found under /kaggle/input. Entries: {listing}. "
        f"Check competition_sources in kernel-metadata.json ('{COMP}') and rule acceptance."
    )


def find_original_train(comp_dir: Path) -> Path | None:
    """原始数据集只有 129,880 行、多一个 'Unnamed: 0' 列，用它跟比赛数据区分开。"""
    if not IS_KAGGLE:
        return None
    for path in Path("/kaggle/input").rglob("train.csv"):
        if path.parent == comp_dir:
            continue
        try:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        if TARGET in cols and "Unnamed: 0" in cols:
            return path
    return None


DATA_DIR = resolve_competition_dir()
ORIG_TRAIN = find_original_train(DATA_DIR) if USE_ORIGINAL_DATA else None

random.seed(SEED)
np.random.seed(SEED)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 60)
print(f"run_mode={RUN_MODE}  n_splits={N_SPLITS}  kaggle={IS_KAGGLE}")
print(f"competition data: {DATA_DIR}")
print(f"original data:    {ORIG_TRAIN if ORIG_TRAIN else '(not used)'}")

# %% [markdown]
# ## 1. 读取数据与轻量 EDA

# %%
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(f"train: {train.shape}   test: {test.shape}   sample_sub: {sample_sub.shape}")
print(f"target rate: {train[TARGET].mean():.4f}")

na = train.isna().sum()
print("\nmissing values (columns with >0):")
print(na[na > 0] if (na > 0).any() else "  none")

if RUN_MODE == "smoke":
    train = train.sample(SMOKE_ROWS, random_state=SEED).reset_index(drop=True)
    test = test.head(SMOKE_ROWS // 2).reset_index(drop=True)
    print(f"smoke sampling -> train {train.shape}, test {test.shape}")

# %% [markdown]
# ## 2. 特征工程
#
# 原始列分三组：
# - **服务评分**（0–5 有序）：13 个。0 表示"未使用/不适用"，所以除均值外还统计 0 / 4 / 5 的个数
# - **数值**：Age、Flight Distance、Departure/Arrival Delay
# - **类别**：Gender、Customer Type、Type of Travel、Class
#
# 工程特征集中在四类：
# 1. 整体体验：均值、标准差、极值、0 与 5 的计数、0 的占比
# 2. 分组体验：数字渠道（WiFi / 在线订票 / 在线登机）与舒适度（座椅 / 腿部空间 / 清洁 / 餐食）
# 3. 延误结构：到达延误缺失、总延误、延误差、是否延误、对数
# 4. 强交互：`Class × Type of Travel`、`Customer Type × Class`、年龄/距离/延误分箱

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
DIGITAL_COLS = ["Inflight wifi service", "Ease of Online booking", "Online boarding"]
COMFORT_COLS = ["Seat comfort", "Leg room service", "Cleanliness", "Food and drink"]
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
CROSS_COLS = ["class_travel", "customer_class", "gender_travel", "age_group", "distance_bin", "delay_bin"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    s = out[SERVICE_COLS]
    out["service_mean"] = s.mean(axis=1)
    out["service_std"] = s.std(axis=1)
    out["service_min"] = s.min(axis=1)
    out["service_max"] = s.max(axis=1)
    out["service_sum"] = s.sum(axis=1)
    out["service_n_zero"] = (s == 0).sum(axis=1)
    out["service_n_four"] = (s == 4).sum(axis=1)
    out["service_n_five"] = (s == 5).sum(axis=1)
    out["service_zero_ratio"] = out["service_n_zero"] / len(SERVICE_COLS)
    out["digital_mean"] = out[DIGITAL_COLS].mean(axis=1)
    out["comfort_mean"] = out[COMFORT_COLS].mean(axis=1)

    dep = out["Departure Delay in Minutes"].astype(float)
    arr = out["Arrival Delay in Minutes"].astype(float)
    out["arrival_delay_missing"] = arr.isna().astype(int)
    arr_filled = arr.fillna(dep)
    out["arrival_delay_filled"] = arr_filled
    out["delay_total"] = dep + arr_filled
    out["delay_diff"] = arr_filled - dep
    out["has_dep_delay"] = (dep > 0).astype(int)
    out["log_dep_delay"] = np.log1p(dep)
    out["log_delay_total"] = np.log1p(out["delay_total"])

    out["log_distance"] = np.log1p(out["Flight Distance"].astype(float))
    out["distance_per_age"] = out["Flight Distance"] / out["Age"].clip(lower=1)

    # 分箱：GBDT 自己能找到切点，但显式分箱能让类别交叉携带更集中的统计量
    out["age_group"] = pd.cut(out["Age"], bins=[0, 25, 35, 45, 55, 65, 200], labels=False).astype(int)
    out["distance_bin"] = pd.qcut(
        out["Flight Distance"].rank(method="first"), 10, labels=False, duplicates="drop"
    ).astype(int)
    out["delay_bin"] = pd.cut(
        out["delay_total"], bins=[-1, 0, 15, 60, 180, np.inf], labels=False
    ).astype(int)

    out["class_travel"] = out["Class"] + "_" + out["Type of Travel"]
    out["customer_class"] = out["Customer Type"] + "_" + out["Class"]
    out["gender_travel"] = out["Gender"] + "_" + out["Type of Travel"]

    return out


train_fe = build_features(train)
test_fe = build_features(test)
train_frames = [train_fe]

orig_fe = None
if ORIG_TRAIN is not None:
    orig_raw = pd.read_csv(ORIG_TRAIN)
    orig_raw = orig_raw.drop(columns=[c for c in orig_raw.columns if c.startswith("Unnamed")], errors="ignore")
    orig_raw[TARGET] = orig_raw[TARGET].astype(str).str.strip().str.lower().eq("satisfied").astype(int)
    if RUN_MODE == "smoke":
        orig_raw = orig_raw.head(SMOKE_ROWS)
    orig_fe = build_features(orig_raw)
    train_frames.append(orig_fe)
    print(f"original data rows: {len(orig_fe)}  target rate: {orig_fe[TARGET].mean():.4f}")

frames = train_frames + [test_fe]  # 仅用于统一 category 取值

FEATURE_COLS = [c for c in train_fe.columns if c not in (ID, TARGET)]
print(f"engineered feature count: {len(FEATURE_COLS)}")

# 训练/推理必须共用同一套 category（原始数据可能带来新的类别取值）
STRING_CROSS = ["class_travel", "customer_class", "gender_travel"]
NUM_CROSS = ["age_group", "distance_bin", "delay_bin"]
for col in CAT_COLS + STRING_CROSS:
    categories = sorted(set().union(*[set(f[col].astype(str)) for f in frames]))
    dtype = pd.CategoricalDtype(categories=categories, ordered=False)
    for f in frames:
        f[col] = f[col].astype(str).astype(dtype)
for col in NUM_CROSS:
    values = sorted(set().union(*[set(f[col].astype(int)) for f in frames]))
    dtype = pd.CategoricalDtype(categories=values, ordered=True)
    for f in frames:
        f[col] = f[col].astype(int).astype(dtype)

n_syn = len(train_fe)
X = pd.concat([f[FEATURE_COLS] for f in train_frames], axis=0, ignore_index=True)
y_full = np.concatenate([f[TARGET].astype(int).to_numpy() for f in train_frames])
y = train_fe[TARGET].astype(int).to_numpy()
X_test = test_fe[FEATURE_COLS]
EXTRA_IDX = np.arange(n_syn, len(X))  # 原始数据行，只参与训练、不参与验证

print(f"X: {X.shape} (synthetic {n_syn} + extra {len(EXTRA_IDX)})  X_test: {X_test.shape}")
print(f"positive rate: {y.mean():.4f}")

# %% [markdown]
# ## 3. 三个模型 × 5 折 CV
#
# 折划分只在比赛训练集上做；补充数据整体并入每个 fold 的训练集，因此 OOF AUC 仍然可比。

# %%
import lightgbm as lgb

LGB_ROUNDS = 5000 if RUN_MODE == "full" else 200


def fit_lgb(tr_idx: np.ndarray, va_idx: np.ndarray):
    params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=100,
        feature_fraction=0.65,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.1,
        lambda_l2=1.0,
        max_bin=255,
        n_jobs=-1,
        verbosity=-1,
        seed=SEED,
    )
    dtrain = lgb.Dataset(X.iloc[tr_idx], y_full[tr_idx], free_raw_data=True)
    dvalid = lgb.Dataset(X.iloc[va_idx], y[va_idx], reference=dtrain, free_raw_data=True)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=LGB_ROUNDS,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(0)],
    )
    gain = pd.Series(model.feature_importance("gain"), index=FEATURE_COLS)
    return (
        model.predict(X.iloc[va_idx], num_iteration=model.best_iteration),
        model.predict(X_test, num_iteration=model.best_iteration),
        gain,
        model.best_iteration,
    )


import xgboost as xgb

XGB_ROUNDS = 5000 if RUN_MODE == "full" else 200


def fit_xgb(tr_idx: np.ndarray, va_idx: np.ndarray):
    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        max_depth=8,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.65,
        min_child_weight=10,
        reg_lambda=1.0,
        reg_alpha=0.1,
        n_jobs=-1,
        seed=SEED,
        enable_categorical=True,
        max_cat_to_onehot=1,
    )
    dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y_full[tr_idx], enable_categorical=True)
    dvalid = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=XGB_ROUNDS,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=250,
        verbose_eval=False,
    )
    rng = (0, model.best_iteration + 1)
    gain = pd.Series(model.get_score(importance_type="total_gain")).reindex(FEATURE_COLS).fillna(0.0)
    return model.predict(dvalid, iteration_range=rng), model.predict(dtest, iteration_range=rng), gain, model.best_iteration


CAT_ROUNDS = 3000 if RUN_MODE == "full" else 200


def fit_catboost(tr_idx: np.ndarray, va_idx: np.ndarray):
    from catboost import CatBoostClassifier, Pool

    cat_features = [c for c in FEATURE_COLS if str(X[c].dtype) == "category"]
    X_cat = X.copy()
    for c in cat_features:
        X_cat[c] = X_cat[c].astype(str)
    X_test_cat = X_test.copy()
    for c in cat_features:
        X_test_cat[c] = X_test_cat[c].astype(str)

    pool_tr = Pool(X_cat.iloc[tr_idx], y_full[tr_idx], cat_features=cat_features)
    pool_va = Pool(X_cat.iloc[va_idx], y[va_idx], cat_features=cat_features)
    model = CatBoostClassifier(
        iterations=CAT_ROUNDS,
        learning_rate=0.06,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        od_type="Iter",
        od_wait=200,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
    gain = pd.Series(model.get_feature_importance(pool_tr, type="LossFunctionChange"), index=FEATURE_COLS)
    return (
        model.predict_proba(X_cat.iloc[va_idx])[:, 1],
        model.predict_proba(X_test_cat)[:, 1],
        gain,
        model.get_best_iteration(),
    )


def run_cv(name: str, fit_fn) -> tuple[np.ndarray, np.ndarray, float, pd.Series]:
    oof = np.zeros(n_syn)
    pred = np.zeros(len(X_test))
    gain = pd.Series(0.0, index=FEATURE_COLS)
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        p_va, p_te, g, best_iter = fit_fn(tr_idx, va_idx)
        oof[va_idx] = p_va
        pred += p_te / len(folds)
        gain += g / len(folds)
        print(
            f"[{name}] fold {fold}: best_iter={best_iter:4d}  "
            f"auc={roc_auc_score(y[va_idx], p_va):.6f}  ({time.time() - t0:.0f}s)"
        )
    auc = roc_auc_score(y, oof)
    print(f"[{name}] OOF AUC = {auc:.6f}\n")
    return oof, pred, auc, gain


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = []
for tr_idx, va_idx in skf.split(np.zeros(n_syn), y):
    folds.append((np.concatenate([tr_idx, EXTRA_IDX]), va_idx))
print(f"{len(folds)} folds, train sizes: {[len(t) for t, _ in folds]}")

results: dict[str, dict] = {}
results["lgb"] = dict(zip(("oof", "pred", "auc", "gain"), run_cv("lgb", fit_lgb)))
results["xgb"] = dict(zip(("oof", "pred", "auc", "gain"), run_cv("xgb", fit_xgb)))

if USE_CATBOOST:
    try:
        results["cat"] = dict(zip(("oof", "pred", "auc", "gain"), run_cv("cat", fit_catboost)))
    except Exception as exc:  # CatBoost 未安装或不兼容时不让整次运行失败
        print(f"[warn] CatBoost skipped: {type(exc).__name__}: {exc}")

# %% [markdown]
# ## 4. 融合
#
# 不同模型的分数尺度不同，先转成 rank，再在概率单纯形上以 0.05 步长搜索权重。
# 单模型已经最优时直接采用单模型。

# %%
from scipy.stats import rankdata

names = list(results)
oof_rank = {k: rankdata(results[k]["oof"]) / n_syn for k in names}
test_rank = {k: rankdata(results[k]["pred"]) / len(X_test) for k in names}

best_weights, best_auc = None, -1.0


def simplex_weights(n: int, step: float = 0.05):
    """生成 n 个非负、和为 1 的权重组合（步长 step）。"""
    levels = np.arange(0.0, 1.0001, step)

    def rec(prefix: list[float], remaining: float):
        if len(prefix) == n - 1:
            yield tuple(prefix + [max(remaining, 0.0)])
            return
        for w in levels:
            if w > remaining + 1e-9:
                break
            yield from rec(prefix + [float(w)], remaining - w)

    yield from rec([], 1.0)


for weights in simplex_weights(len(names)):
    blend = sum(w * oof_rank[k] for w, k in zip(weights, names))
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_weights, best_auc = weights, auc

print("model OOF AUC:", {k: round(results[k]["auc"], 6) for k in names})
print("best blend:", {k: round(w, 2) for k, w in zip(names, best_weights)}, f"-> OOF AUC={best_auc:.6f}")

best_single = max(names, key=lambda k: results[k]["auc"])
if results[best_single]["auc"] >= best_auc - 1e-9:
    print(f"-> using single model: {best_single}")
    final_test = results[best_single]["pred"]
    final_oof = results[best_single]["oof"]
else:
    print("-> using weighted blend")
    final_test = sum(w * test_rank[k] for w, k in zip(best_weights, names))
    final_oof = sum(w * oof_rank[k] for w, k in zip(best_weights, names))

final_auc = roc_auc_score(y, final_oof)
print(f"final OOF AUC = {final_auc:.6f}")

# %%
importance = pd.DataFrame({f"{k}_gain": results[k]["gain"] for k in names})
importance["rank"] = importance[[f"{k}_gain" for k in names]].rank(ascending=False).mean(axis=1)
print("top 20 features (mean rank across models):")
print(importance.sort_values("rank").head(20).round(1).to_string())

# %% [markdown]
# ## 5. 生成提交文件

# %%
submission = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: np.asarray(final_test, dtype=float)})
if RUN_MODE == "full":
    # smoke 模式截断了 test，长度对不上是预期的
    assert len(submission) == len(sample_sub), (len(submission), len(sample_sub))
assert submission[ID].is_unique
assert submission[TARGET].between(0, 1).all()

sub_path = OUT_DIR / "submission.csv"
submission.to_csv(sub_path, index=False)
print(f"wrote {sub_path}  rows={len(submission)}")
print(submission.head())

metrics = {
    "final_oof_auc": final_auc,
    "model_oof_auc": {k: results[k]["auc"] for k in names},
    "blend_weights": dict(zip(names, [float(w) for w in best_weights])),
    "n_features": len(FEATURE_COLS),
    "n_train": int(n_syn),
    "n_extra": int(len(EXTRA_IDX)),
}
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))

# %% [markdown]
# ## 6. 后续可尝试的方向
#
# - 10 折 + 多 seed 平均（OOF 方差下降后，融合权重更可信）
# - 用 OOF 预测做二级 stacking（LogisticRegression / 浅层 LGBM），通常再 +0.0005 左右
# - `Online boarding` 与 `service_n_zero` 的交互、按 `Class` 分层的目标编码
# - 调 `num_leaves` / `min_child_samples` 与 `feature_fraction`，本例对这三者最敏感
# - 若 CV 与 LB 差距持续，检查是否有分布偏移（合成数据的 test 采样方式）
