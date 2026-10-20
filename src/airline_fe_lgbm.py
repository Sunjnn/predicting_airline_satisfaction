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
#
# 单一 LightGBM、5 折，不做融合，方便干净地读出「特征」本身的收益。

# %%
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
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
# ## 6. 5 折 CV

# %%
BASE_FEATURES = [c for c in train.columns]
LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    learning_rate=0.02,
    num_leaves=255,
    max_depth=8,
    min_child_samples=5,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    lambda_l2=5.0,
    lambda_l1=0.05,
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
# - 若这套特征把 OOF 推到 0.960+，把它作为主线 notebook 的特征层，再叠 10 折 + 多模型
# - 还差原作者的两块：原始数据按精确键元组查 target rate、数值列转 category
# - 注意公开 notebook 的高分有相当部分来自对公开 LB 的曲面拟合（见 `rank16` 那篇），
#   那类做法在 private LB 上大概率不成立，不作为参考
