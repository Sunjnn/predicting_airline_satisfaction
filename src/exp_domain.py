"""Diagnostic: is the original Invistico dataset on the same distribution as the
synthetic competition data, and does adding it help?

Three measurements, all with the same small feature set and a short LightGBM:

  A  train on original       -> eval on synthetic holdout
  B  train on synthetic      -> eval on original holdout
  C  train on synthetic      -> eval on synthetic holdout   (reference, what CV reports)
  D  train on both           -> eval on synthetic holdout   (does the extra data help?)

Run: .venv\\Scripts\\python.exe src\\exp_domain.py
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
TARGET = "satisfaction"

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
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    s = out[SERVICE_COLS]
    out["service_mean"] = s.mean(axis=1)
    out["service_std"] = s.std(axis=1)
    out["service_sum"] = s.sum(axis=1)
    out["service_n_zero"] = (s == 0).sum(axis=1)
    out["service_n_five"] = (s == 5).sum(axis=1)
    dep = out["Departure Delay in Minutes"].astype(float)
    arr = out["Arrival Delay in Minutes"].astype(float)
    out["arrival_delay_filled"] = arr.fillna(dep)
    out["delay_total"] = dep + out["arrival_delay_filled"]
    out["log_distance"] = np.log1p(out["Flight Distance"].astype(float))
    out["class_travel"] = out["Class"] + "_" + out["Type of Travel"]
    return out


CANON = [
    *SERVICE_COLS,
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
    "service_mean",
    "service_std",
    "service_sum",
    "service_n_zero",
    "service_n_five",
    "arrival_delay_filled",
    "delay_total",
    "log_distance",
    "class_travel",
    TARGET,
]


def prep(df: pd.DataFrame) -> pd.DataFrame:
    fe = build(df)
    for col in CAT_COLS + ["class_travel"]:
        fe[col] = fe[col].astype("category")
    # 原始数据集多出 'Unnamed: 0' 和 'Inflight service' 两列，统一裁到同一套特征上
    return fe[CANON]


def fit_eval(tr: pd.DataFrame, va: pd.DataFrame, label: str) -> float:
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=96,
        min_child_samples=100,
        feature_fraction=0.7,
        bagging_fraction=0.85,
        bagging_freq=1,
        n_jobs=-1,
        verbosity=-1,
        seed=SEED,
    )
    # 类别字典必须两边一致
    for col in tr.columns:
        if str(tr[col].dtype) == "category":
            cats = sorted(set(tr[col].cat.categories) | set(va[col].cat.categories))
            tr[col] = tr[col].cat.set_categories(cats)
            va[col] = va[col].cat.set_categories(cats)
    y_tr = tr[TARGET].astype(int)
    y_va = va[TARGET].astype(int)
    model = lgb.train(
        params,
        lgb.Dataset(tr.drop(columns=[TARGET]), y_tr),
        num_boost_round=400,
    )
    auc = roc_auc_score(y_va, model.predict(va.drop(columns=[TARGET])))
    print(f"{label:<44} AUC={auc:.6f}  (n_train={len(tr)}, n_eval={len(va)})")
    return auc


def main() -> None:
    syn = pd.read_csv(DATA / "train.csv")
    orig = pd.read_csv(DATA / "orig" / "train.csv")
    orig[TARGET] = orig[TARGET].astype(str).str.strip().str.lower().eq("satisfied").astype(int)
    print(f"synthetic: {syn.shape} target={syn[TARGET].mean():.4f}")
    print(f"original:  {orig.shape} target={orig[TARGET].mean():.4f}")

    # 抽样以控制诊断耗时
    syn_small, syn_eval = train_test_split(syn, test_size=100_000, random_state=SEED, stratify=syn[TARGET])
    orig_fit, orig_eval = train_test_split(orig, test_size=20_000, random_state=SEED, stratify=orig[TARGET])

    X_syn = prep(syn_small)
    X_syn_eval = prep(syn_eval)
    X_orig = prep(orig_fit)
    X_orig_eval = prep(orig_eval)

    print()
    fit_eval(X_orig, X_syn_eval, "A  original -> synthetic")
    fit_eval(X_syn, X_orig_eval, "B  synthetic -> original")
    fit_eval(X_syn, X_syn_eval, "C  synthetic -> synthetic (reference)")
    both = pd.concat([X_orig, X_syn], ignore_index=True)
    fit_eval(both, X_syn_eval, "D  original+synthetic -> synthetic")

    # 分布对比：原始数据里哪些取值在合成数据里没出现
    print("\nvalue ranges (synthetic vs original):")
    for col in ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
        print(
            f"  {col:<30} syn[{syn[col].min():>8.1f},{syn[col].max():>8.1f}] "
            f"orig[{orig[col].min():>8.1f},{orig[col].max():>8.1f}]"
        )
    print("\nservice rating value counts (synthetic share vs original share):")
    for col in ["Online boarding", "Inflight wifi service"]:
        a = syn[col].value_counts(normalize=True).sort_index()
        b = orig[col].value_counts(normalize=True).sort_index()
        print(f"  {col}")
        print(pd.DataFrame({"synthetic": a, "original": b}).round(3).to_string())


if __name__ == "__main__":
    main()
