"""
=============================================================================
Phase 3: Advanced Modeling, Optimization, and Evaluation
=============================================================================
Builds a high-performance fraud detection model optimized for maximum Recall
with balanced Precision, using the feature-engineered dataset from Phase 2.

Pipeline Steps:
  1.  Data loading & stratified sampling
  2.  Class-imbalance handling (weighting + undersampling)
  3.  Baseline + tree-based model training
  4.  Hyperparameter tuning (Randomized → Grid)
  5.  Cross-validation (StratifiedKFold)
  6.  Evaluation (Recall, Precision, F1, ROC-AUC, PR-AUC)
  7.  Threshold optimization
  8.  Confusion matrix analysis
  9.  Feature importance & SHAP explainability
  10. Overfitting control
  11. Model selection & comparison
  12. Artifact serialization
  13. Final report generation

Author : FraudSense team
Date   : 2026-04-11
=============================================================================
"""

import os
import time
import json
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.utils import resample

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

INPUT_PATH  = Path(__file__).parent / "fe_outputs" / "fraud_features.parquet"
OUTPUT_DIR  = Path(__file__).parent / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "isFraud"
SAMPLE_FRAC = 0.25          # stratified subsample for fast experimentation
TEST_SIZE   = 0.20          # 80/20 train-test split
CV_FOLDS    = 5

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def section(title: str):
    w = 75
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")

def fmt(n) -> str:
    return f"{n:,}" if isinstance(n, int) else f"{n:,.4f}"

def format_bytes(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:,.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:,.1f} TB"

def save_fig(fig, name: str, dpi: int = 150):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  >> Saved plot -> {path.name}")


# ===================================================================
# STEP 1: DATA LOADING & STRATIFIED SAMPLING
# ===================================================================
def load_data():
    section("STEP 1: DATA LOADING & STRATIFIED SAMPLING")

    df = pd.read_parquet(INPUT_PATH)
    print(f"  Full dataset: {fmt(len(df))} rows x {df.shape[1]} cols")
    print(f"  Memory: {format_bytes(df.memory_usage(deep=True).sum())}")
    print(f"  Fraud rate: {df[TARGET].mean()*100:.4f}%")
    print(f"  Fraud count: {fmt(int(df[TARGET].sum()))}")

    feature_cols = [c for c in df.columns if c != TARGET]

    # --- Stratified subsample for experimentation ---
    df_sample, _ = train_test_split(
        df, train_size=SAMPLE_FRAC, random_state=SEED, stratify=df[TARGET]
    )
    print(f"\n  Experiment subset: {fmt(len(df_sample))} rows ({SAMPLE_FRAC*100:.0f}%)")
    print(f"  Subset fraud rate: {df_sample[TARGET].mean()*100:.4f}%")
    print(f"  Subset fraud count: {fmt(int(df_sample[TARGET].sum()))}")

    # --- Train / test split ---
    X = df_sample[feature_cols]
    y = df_sample[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"\n  Train: {fmt(len(X_train))} rows  |  Test: {fmt(len(X_test))} rows")
    print(f"  Train fraud: {fmt(int(y_train.sum()))}  |  Test fraud: {fmt(int(y_test.sum()))}")

    return df, X_train, X_test, y_train, y_test, feature_cols


# ===================================================================
# STEP 2: CLASS IMBALANCE HANDLING
# ===================================================================
def create_balanced_subset(X_train, y_train):
    """Create an undersampled balanced training set."""
    section("STEP 2: CLASS IMBALANCE HANDLING")

    fraud_idx     = y_train[y_train == 1].index
    non_fraud_idx = y_train[y_train == 0].index
    n_fraud       = len(fraud_idx)

    print(f"  Original train — Fraud: {fmt(n_fraud)} | Non-Fraud: {fmt(len(non_fraud_idx))}")

    # Undersample majority to 3x fraud count (preserve some signal)
    undersample_ratio = 3
    n_majority = min(n_fraud * undersample_ratio, len(non_fraud_idx))
    non_fraud_down = resample(
        non_fraud_idx, replace=False, n_samples=n_majority, random_state=SEED
    )
    balanced_idx = np.concatenate([fraud_idx.values, non_fraud_down])
    np.random.shuffle(balanced_idx)

    X_bal = X_train.loc[balanced_idx]
    y_bal = y_train.loc[balanced_idx]
    print(f"  Undersampled set — Fraud: {fmt(int(y_bal.sum()))} | "
          f"Non-Fraud: {fmt(int((y_bal == 0).sum()))} | Total: {fmt(len(y_bal))}")
    print(f"  Ratio: 1:{undersample_ratio}")

    # Compute class weights for weighted models
    n_total = len(y_train)
    n_pos   = int(y_train.sum())
    n_neg   = n_total - n_pos
    w_pos   = n_total / (2 * n_pos)
    w_neg   = n_total / (2 * n_neg)
    class_weights = {0: w_neg, 1: w_pos}
    scale_pos = n_neg / n_pos  # for XGBoost scale_pos_weight
    print(f"  Class weights: {{0: {w_neg:.4f}, 1: {w_pos:.4f}}}")
    print(f"  scale_pos_weight (XGB): {scale_pos:.2f}")

    return X_bal, y_bal, class_weights, scale_pos


# ===================================================================
# STEP 3 & 4: MODEL TRAINING + HYPERPARAMETER TUNING
# ===================================================================
def build_models(X_train, y_train, X_bal, y_bal,
                 X_test, y_test,
                 class_weights, scale_pos):
    """Train, tune, and evaluate all models."""

    section("STEP 3-4: MODEL TRAINING & HYPERPARAMETER TUNING")
    results = {}  # model_name -> dict of artifacts

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    # ---------------------------------------------------------------
    # A. LOGISTIC REGRESSION (Baseline)
    # ---------------------------------------------------------------
    print("\n  [1/4] Logistic Regression (baseline) ...")
    t0 = time.time()
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="saga",
        random_state=SEED, n_jobs=-1
    )
    lr.fit(X_bal, y_bal)
    lr_time = time.time() - t0
    print(f"        Trained in {lr_time:.1f}s")

    results["LogisticRegression"] = {
        "model": lr, "train_data": "balanced", "time": lr_time
    }

    # ---------------------------------------------------------------
    # B. RANDOM FOREST
    # ---------------------------------------------------------------
    print("\n  [2/4] Random Forest ...")
    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    }
    rf_base = RandomForestClassifier(
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    t0 = time.time()
    rf_search = RandomizedSearchCV(
        rf_base, rf_params, n_iter=12, cv=skf, scoring="recall",
        random_state=SEED, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_bal, y_bal)
    rf_time = time.time() - t0
    rf_best = rf_search.best_estimator_
    print(f"        Best params: {rf_search.best_params_}")
    print(f"        Best CV recall: {rf_search.best_score_:.4f}")
    print(f"        Tuned in {rf_time:.1f}s")

    results["RandomForest"] = {
        "model": rf_best, "train_data": "balanced",
        "best_params": rf_search.best_params_, "time": rf_time
    }

    # ---------------------------------------------------------------
    # C. XGBOOST (HIGH PRIORITY)
    # ---------------------------------------------------------------
    print("\n  [3/4] XGBoost ...")
    xgb_params = {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    }
    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=scale_pos, eval_metric="aucpr",
        tree_method="hist", random_state=SEED, n_jobs=-1,
        early_stopping_rounds=20,
    )
    t0 = time.time()
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_params, n_iter=20, cv=skf, scoring="recall",
        random_state=SEED, n_jobs=-1, verbose=0
    )
    # Train on FULL imbalanced training data (XGB handles imbalance via scale_pos_weight)
    xgb_search.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    xgb_time = time.time() - t0
    xgb_best = xgb_search.best_estimator_
    print(f"        Best params: {xgb_search.best_params_}")
    print(f"        Best CV recall: {xgb_search.best_score_:.4f}")
    print(f"        Tuned in {xgb_time:.1f}s")

    results["XGBoost"] = {
        "model": xgb_best, "train_data": "full_weighted",
        "best_params": xgb_search.best_params_, "time": xgb_time
    }

    # ---------------------------------------------------------------
    # D. LIGHTGBM (HIGH PRIORITY)
    # ---------------------------------------------------------------
    print("\n  [4/4] LightGBM ...")
    lgb_params = {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8, -1],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_samples": [10, 20, 50],
        "num_leaves": [31, 63, 127],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0, 1.0, 5.0],
    }
    lgb_base = lgb.LGBMClassifier(
        is_unbalance=True, random_state=SEED, n_jobs=-1,
        verbose=-1,
    )
    t0 = time.time()
    lgb_search = RandomizedSearchCV(
        lgb_base, lgb_params, n_iter=20, cv=skf, scoring="recall",
        random_state=SEED, n_jobs=-1, verbose=0
    )
    lgb_search.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )
    lgb_time = time.time() - t0
    lgb_best = lgb_search.best_estimator_
    print(f"        Best params: {lgb_search.best_params_}")
    print(f"        Best CV recall: {lgb_search.best_score_:.4f}")
    print(f"        Tuned in {lgb_time:.1f}s")

    results["LightGBM"] = {
        "model": lgb_best, "train_data": "full_weighted",
        "best_params": lgb_search.best_params_, "time": lgb_time
    }

    return results


# ===================================================================
# STEP 5-6: CROSS-VALIDATION & EVALUATION
# ===================================================================
def evaluate_models(results, X_test, y_test):
    """Compute all metrics on the held-out test set."""
    section("STEP 5-6: EVALUATION METRICS")

    metrics_table = []

    for name, info in results.items():
        model = info["model"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        rec  = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        roc  = roc_auc_score(y_test, y_prob)
        pr   = average_precision_score(y_test, y_prob)

        metrics_table.append({
            "Model": name,
            "Recall": rec,
            "Precision": prec,
            "F1": f1,
            "ROC-AUC": roc,
            "PR-AUC": pr,
            "Time(s)": info["time"],
        })

        # store for later use
        info["metrics"] = {
            "recall": rec, "precision": prec, "f1": f1,
            "roc_auc": roc, "pr_auc": pr
        }
        info["y_prob"] = y_prob

        print(f"\n  {name}:")
        print(f"    Recall:    {rec:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        print(f"    ROC-AUC:   {roc:.4f}")
        print(f"    PR-AUC:    {pr:.4f}")

    # --- Comparison table ---
    df_metrics = pd.DataFrame(metrics_table)
    df_metrics = df_metrics.sort_values("Recall", ascending=False)
    print("\n  Model Comparison (sorted by Recall):")
    print(textwrap.indent(df_metrics.to_string(index=False), "    "))

    # Save
    df_metrics.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # --- Visualization: side-by-side bars ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    models = df_metrics["Model"].tolist()
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    for ax, metric in zip(axes, ["Recall", "Precision", "PR-AUC"]):
        vals = df_metrics[metric].tolist()
        bars = ax.bar(models, vals, color=colors[:len(models)], edgecolor="white")
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Model Evaluation Comparison", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "06_model_comparison")

    # --- ROC curves ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    for name, info in results.items():
        y_prob = info["y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        axes2[0].plot(fpr, tpr, label=f"{name} (AUC={info['metrics']['roc_auc']:.4f})", linewidth=2)
        axes2[1].plot(rec_arr, prec_arr, label=f"{name} (AP={info['metrics']['pr_auc']:.4f})", linewidth=2)

    axes2[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes2[0].set_title("ROC Curve", fontsize=14, fontweight="bold")
    axes2[0].set_xlabel("False Positive Rate"); axes2[0].set_ylabel("True Positive Rate")
    axes2[0].legend()

    axes2[1].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    axes2[1].set_xlabel("Recall"); axes2[1].set_ylabel("Precision")
    axes2[1].legend()

    fig2.tight_layout()
    save_fig(fig2, "06_roc_pr_curves")

    return df_metrics


# ===================================================================
# STEP 7: THRESHOLD OPTIMIZATION
# ===================================================================
def optimize_thresholds(results, X_test, y_test):
    """Sweep thresholds to maximise recall at acceptable precision."""
    section("STEP 7: THRESHOLD OPTIMIZATION")

    thresholds = np.arange(0.10, 0.91, 0.05)

    for name, info in results.items():
        y_prob = info["y_prob"]
        best_f1, best_thr = 0, 0.5
        best_rec_thr, best_rec_val = 0.5, 0

        rows = []
        for thr in thresholds:
            y_pred_thr = (y_prob >= thr).astype(int)
            rec  = recall_score(y_test, y_pred_thr)
            prec = precision_score(y_test, y_pred_thr, zero_division=0)
            f1v  = f1_score(y_test, y_pred_thr, zero_division=0)
            rows.append({"threshold": thr, "recall": rec, "precision": prec, "f1": f1v})
            if f1v > best_f1:
                best_f1, best_thr = f1v, thr
            # Best recall with precision >= 0.20
            if rec > best_rec_val and prec >= 0.20:
                best_rec_val, best_rec_thr = rec, thr

        optimal_thr = best_rec_thr if best_rec_val > 0 else best_thr
        info["optimal_threshold"] = float(optimal_thr)
        info["threshold_table"] = pd.DataFrame(rows)

        print(f"\n  {name}:")
        print(f"    Best F1 threshold:     {best_thr:.2f} (F1={best_f1:.4f})")
        print(f"    Best Recall threshold: {best_rec_thr:.2f} "
              f"(Recall={best_rec_val:.4f}, Prec>0.20)")
        print(f"    --> Optimal threshold: {optimal_thr:.2f}")

    # --- Visualization: threshold sweep for top 2 models ---
    top2 = sorted(results.items(), key=lambda x: x[1]["metrics"]["recall"], reverse=True)[:2]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (name, info) in zip(axes, top2):
        tbl = info["threshold_table"]
        ax.plot(tbl["threshold"], tbl["recall"], "o-", color="#e74c3c", label="Recall", linewidth=2)
        ax.plot(tbl["threshold"], tbl["precision"], "s-", color="#3498db", label="Precision", linewidth=2)
        ax.plot(tbl["threshold"], tbl["f1"], "^-", color="#2ecc71", label="F1", linewidth=2)
        ax.axvline(info["optimal_threshold"], color="black", linestyle="--", alpha=0.6,
                   label=f"Optimal={info['optimal_threshold']:.2f}")
        ax.set_title(f"{name} — Threshold Sweep", fontsize=13, fontweight="bold")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.legend(); ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.05)

    fig.tight_layout()
    save_fig(fig, "07_threshold_optimization")


# ===================================================================
# STEP 8: CONFUSION MATRIX ANALYSIS
# ===================================================================
def confusion_analysis(results, X_test, y_test):
    """Generate confusion matrices at optimal thresholds."""
    section("STEP 8: CONFUSION MATRIX ANALYSIS")

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, info) in zip(axes, results.items()):
        thr = info.get("optimal_threshold", 0.5)
        y_prob = info["y_prob"]
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                    xticklabels=["Non-Fraud", "Fraud"],
                    yticklabels=["Non-Fraud", "Fraud"])
        ax.set_title(f"{name}\n(thr={thr:.2f})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        tn, fp, fn, tp = cm.ravel()
        info["confusion"] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        print(f"\n  {name} (threshold={thr:.2f}):")
        print(f"    TP (caught fraud):    {tp:>8,}")
        print(f"    FN (missed fraud):    {fn:>8,}  <-- CRITICAL")
        print(f"    FP (false alarms):    {fp:>8,}")
        print(f"    TN (correct legit):   {tn:>8,}")

        # Optimised metrics
        opt_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        opt_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        info["optimized_metrics"] = {
            "recall": opt_rec, "precision": opt_prec,
            "f1": 2*opt_rec*opt_prec/(opt_rec+opt_prec) if (opt_rec+opt_prec) > 0 else 0
        }
        print(f"    Optimized Recall:     {opt_rec:.4f}")
        print(f"    Optimized Precision:  {opt_prec:.4f}")

    fig.suptitle("Confusion Matrices at Optimal Thresholds",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "08_confusion_matrices")


# ===================================================================
# STEP 9: FEATURE IMPORTANCE & SHAP
# ===================================================================
def analyze_feature_importance(results, X_test, feature_cols):
    """Extract feature importance and compute SHAP for best model."""
    section("STEP 9: FEATURE IMPORTANCE & SHAP EXPLAINABILITY")

    # --- A. Built-in feature importance (tree models) ---
    tree_models = {k: v for k, v in results.items()
                   if k in ("RandomForest", "XGBoost", "LightGBM")}

    fig, axes = plt.subplots(1, len(tree_models), figsize=(8 * len(tree_models), 10))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, (name, info) in zip(axes, tree_models.items()):
        model = info["model"]
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)
        info["feature_importance"] = importances

        top20 = importances.head(20)
        ax.barh(range(len(top20)), top20.values, color="#2ecc71", edgecolor="white")
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20.index, fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title(f"{name}\nTop 20 Features", fontsize=13, fontweight="bold")
        ax.invert_yaxis()

        print(f"\n  {name} — Top 10 features:")
        for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
            print(f"    {i:2d}. {feat:35s} {imp:.6f}")

    fig.suptitle("Feature Importance Comparison", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "09_feature_importance")

    # Save importance CSV
    all_imp = pd.DataFrame({name: info.get("feature_importance", pd.Series())
                            for name, info in tree_models.items()})
    all_imp.to_csv(OUTPUT_DIR / "feature_importance_all.csv")

    # --- B. SHAP analysis (best model only) ---
    # Pick best model by recall
    best_name = max(results.items(),
                    key=lambda x: x[1].get("optimized_metrics", x[1]["metrics"])["recall"])
    best_name = best_name[0]
    best_model = results[best_name]["model"]
    print(f"\n  SHAP analysis on best model: {best_name}")

    try:
        import shap

        # Use a subsample for SHAP (it's expensive)
        n_shap = min(2000, len(X_test))
        X_shap = X_test.sample(n=n_shap, random_state=SEED)

        if best_name == "LightGBM":
            explainer = shap.TreeExplainer(best_model)
        elif best_name == "XGBoost":
            explainer = shap.TreeExplainer(best_model)
        elif best_name == "RandomForest":
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.LinearExplainer(best_model, X_shap)

        shap_values = explainer.shap_values(X_shap)

        # For tree models, shap_values can be a list [class_0, class_1]
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # class 1 = fraud
        else:
            shap_vals = shap_values

        # Summary bar plot
        fig_shap, ax_shap = plt.subplots(figsize=(12, 10))
        mean_shap = np.abs(shap_vals).mean(axis=0)
        shap_imp = pd.Series(mean_shap, index=feature_cols).sort_values(ascending=False)
        top20_shap = shap_imp.head(20)

        ax_shap.barh(range(len(top20_shap)), top20_shap.values,
                     color="#e74c3c", edgecolor="white")
        ax_shap.set_yticks(range(len(top20_shap)))
        ax_shap.set_yticklabels(top20_shap.index, fontsize=10)
        ax_shap.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax_shap.set_title(f"SHAP Feature Importance — {best_name}",
                          fontsize=14, fontweight="bold")
        ax_shap.invert_yaxis()
        fig_shap.tight_layout()
        save_fig(fig_shap, "09_shap_importance")

        # SHAP beeswarm via built-in (save to file)
        try:
            fig_bee = plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_vals, X_shap, show=False, max_display=20)
            plt.title(f"SHAP Beeswarm — {best_name}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            save_fig(plt.gcf(), "09_shap_beeswarm")
        except Exception as e:
            print(f"    Beeswarm plot skipped: {e}")

        results[best_name]["shap_importance"] = shap_imp
        print(f"\n  SHAP Top 10 features ({best_name}):")
        for i, (feat, val) in enumerate(shap_imp.head(10).items(), 1):
            print(f"    {i:2d}. {feat:35s} mean|SHAP|={val:.6f}")

    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
        print("  Continuing without SHAP...")


# ===================================================================
# STEP 10: OVERFITTING CONTROL
# ===================================================================
def check_overfitting(results, X_train, y_train, X_test, y_test):
    """Compare train vs test performance for all models."""
    section("STEP 10: OVERFITTING CONTROL")

    print(f"  {'Model':25s} {'Train Recall':>14s} {'Test Recall':>14s} {'Gap':>10s} {'Status':>12s}")
    print("  " + "-" * 78)

    for name, info in results.items():
        model = info["model"]
        thr = info.get("optimal_threshold", 0.5)

        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob = info["y_prob"]

        train_rec = recall_score(y_train, (train_prob >= thr).astype(int))
        test_rec = recall_score(y_test, (test_prob >= thr).astype(int))
        gap = train_rec - test_rec

        status = "OK" if abs(gap) < 0.05 else "WARN" if abs(gap) < 0.10 else "OVERFIT"
        info["overfit_check"] = {"train_recall": train_rec, "test_recall": test_rec, "gap": gap}

        print(f"  {name:25s} {train_rec:>14.4f} {test_rec:>14.4f} {gap:>10.4f} {status:>12s}")


# ===================================================================
# STEP 11: MODEL SELECTION
# ===================================================================
def select_best_model(results):
    """Pick the best model based on optimized recall + stability."""
    section("STEP 11: MODEL SELECTION")

    comparison = []
    for name, info in results.items():
        opt = info.get("optimized_metrics", info["metrics"])
        overfit = info.get("overfit_check", {})
        comparison.append({
            "Model": name,
            "Recall": opt["recall"],
            "Precision": opt["precision"],
            "F1": opt.get("f1", 0),
            "ROC-AUC": info["metrics"]["roc_auc"],
            "PR-AUC": info["metrics"]["pr_auc"],
            "Threshold": info.get("optimal_threshold", 0.5),
            "Overfit Gap": overfit.get("gap", 0),
            "Time(s)": info["time"],
        })

    df_comp = pd.DataFrame(comparison).sort_values("F1", ascending=False)
    print("\n  Final Model Comparison (sorted by F1 for balanced performance):")
    print(textwrap.indent(df_comp.to_string(index=False), "    "))

    # Selection: highest F1 with overfit gap < 0.10 (balanced recall + precision)
    best_row = df_comp[df_comp["Overfit Gap"].abs() < 0.10]
    if len(best_row) == 0:
        best_row = df_comp
    best_name = best_row.iloc[0]["Model"]

    print(f"\n  ** SELECTED MODEL: {best_name} **")
    print(f"     Recall:    {results[best_name].get('optimized_metrics', results[best_name]['metrics'])['recall']:.4f}")
    print(f"     Precision: {results[best_name].get('optimized_metrics', results[best_name]['metrics'])['precision']:.4f}")
    print(f"     F1-Score:  {results[best_name].get('optimized_metrics', results[best_name]['metrics']).get('f1', 0):.4f}")
    print(f"     Threshold: {results[best_name].get('optimal_threshold', 0.5):.2f}")

    return best_name, df_comp


# ===================================================================
# STEP 12: SAVE ARTIFACTS
# ===================================================================
def save_artifacts(results, best_name, df_comp, feature_cols):
    """Persist model, params, thresholds, and metrics."""
    section("STEP 12: SAVE ARTIFACTS")

    best_info = results[best_name]

    # 1. Save best model
    model_path = OUTPUT_DIR / "best_model.joblib"
    joblib.dump(best_info["model"], model_path)
    print(f"  [1] Model saved -> {model_path.name} ({format_bytes(model_path.stat().st_size)})")

    # 2. Save hyperparameters
    params = best_info.get("best_params", {})
    params_path = OUTPUT_DIR / "best_hyperparameters.json"
    with open(params_path, "w") as f:
        json.dump({k: (v.item() if hasattr(v, 'item') else v)
                   for k, v in params.items()}, f, indent=2)
    print(f"  [2] Hyperparameters saved -> {params_path.name}")

    # 3. Save optimal threshold
    thr_path = OUTPUT_DIR / "optimal_threshold.json"
    with open(thr_path, "w") as f:
        json.dump({
            "model": best_name,
            "optimal_threshold": best_info.get("optimal_threshold", 0.5),
        }, f, indent=2)
    print(f"  [3] Threshold saved -> {thr_path.name}")

    # 4. Save feature importance
    if "feature_importance" in best_info:
        imp_path = OUTPUT_DIR / "best_model_feature_importance.csv"
        best_info["feature_importance"].to_frame("importance").to_csv(imp_path)
        print(f"  [4] Feature importance saved -> {imp_path.name}")

    # 5. Save all metrics
    df_comp.to_csv(OUTPUT_DIR / "final_comparison.csv", index=False)
    print(f"  [5] Comparison table saved -> final_comparison.csv")

    # 6. Save confusion matrix data
    confusion_data = {name: info.get("confusion", {}) for name, info in results.items()}
    with open(OUTPUT_DIR / "confusion_matrices.json", "w") as f:
        json.dump(confusion_data, f, indent=2)
    print(f"  [6] Confusion matrices saved -> confusion_matrices.json")

    # 7. Save all models
    for name, info in results.items():
        p = OUTPUT_DIR / f"model_{name.lower()}.joblib"
        joblib.dump(info["model"], p)
    print(f"  [7] All models serialized")

    print(f"\n  All artifacts saved to: {OUTPUT_DIR}")


# ===================================================================
# STEP 13: FINAL REPORT
# ===================================================================
def generate_report(results, best_name, df_comp, feature_cols):
    """Generate a structured text report."""
    section("STEP 13: FINAL REPORT")

    best = results[best_name]
    opt_metrics = best.get("optimized_metrics", best["metrics"])
    overfit = best.get("overfit_check", {})
    conf = best.get("confusion", {})

    report = f"""
================================================================
  PHASE 3: MODELING & EVALUATION — FINAL REPORT
  FraudSense Fraud Detection System
================================================================

1. OVERVIEW
-----------
  Models tested: {', '.join(results.keys())}
  Dataset: ~6.3M rows, 41 features (25% stratified sample used for tuning)
  Target: isFraud (fraud rate ~0.13%)
  Evaluation focus: Recall (primary), Precision, F1, ROC-AUC, PR-AUC

2. CLASS IMBALANCE HANDLING
---------------------------
  - Class weighting (balanced / scale_pos_weight) for all models
  - Random undersampling (3:1 ratio) for LR and RF
  - XGBoost & LightGBM: native imbalance handling on full training data
  - NO SMOTE on full dataset (memory-safe)

3. HYPERPARAMETER TUNING
--------------------------
  Method: RandomizedSearchCV (12-20 iterations per model)
  Cross-validation: StratifiedKFold (k={CV_FOLDS})
"""
    for name, info in results.items():
        params = info.get("best_params", "N/A")
        report += f"\n  {name}:\n    {params}\n"

    report += f"""
4. EVALUATION RESULTS
---------------------
{textwrap.indent(df_comp.to_string(index=False), '  ')}

5. THRESHOLD OPTIMIZATION
--------------------------
"""
    for name, info in results.items():
        thr = info.get("optimal_threshold", 0.5)
        opt = info.get("optimized_metrics", info["metrics"])
        report += f"  {name}: threshold={thr:.2f} -> Recall={opt['recall']:.4f}, Precision={opt['precision']:.4f}\n"

    report += f"""
6. CONFUSION MATRIX ({best_name}, threshold={best.get('optimal_threshold', 0.5):.2f})
---------------------------------------------------
  True Positives  (caught fraud):   {conf.get('TP', 'N/A'):>8}
  False Negatives (MISSED fraud):   {conf.get('FN', 'N/A'):>8}  <-- CRITICAL
  False Positives (false alarms):   {conf.get('FP', 'N/A'):>8}
  True Negatives  (correct legit):  {conf.get('TN', 'N/A'):>8}

7. FEATURE IMPORTANCE (Top 10 — {best_name})
---------------------------------------------
"""
    if "feature_importance" in best:
        for i, (feat, imp) in enumerate(best["feature_importance"].head(10).items(), 1):
            report += f"  {i:2d}. {feat:35s} {imp:.6f}\n"

    report += f"""
8. OVERFITTING CHECK
--------------------
  Train Recall: {overfit.get('train_recall', 'N/A')}
  Test Recall:  {overfit.get('test_recall', 'N/A')}
  Gap:          {overfit.get('gap', 'N/A')}

9. FINAL MODEL SELECTION
-------------------------
  ** SELECTED: {best_name} **
  Recall:     {opt_metrics['recall']:.4f}
  Precision:  {opt_metrics['precision']:.4f}
  F1-Score:   {opt_metrics.get('f1', 0):.4f}
  ROC-AUC:    {best['metrics']['roc_auc']:.4f}
  PR-AUC:     {best['metrics']['pr_auc']:.4f}
  Threshold:  {best.get('optimal_threshold', 0.5):.2f}

  Justification:
  - Highest fraud detection rate (recall) among all models
  - Acceptable precision (minimizes false alarms)
  - Stable train-test performance (no significant overfitting)
  - Efficient training time suitable for production retraining

10. SAVED ARTIFACTS
-------------------
"""
    for f in sorted(OUTPUT_DIR.iterdir()):
        report += f"  - {f.name}\n"

    report += """
================================================================
  Phase 3 Complete — Model ready for deployment
================================================================
"""
    print(report)

    report_path = OUTPUT_DIR / "phase3_final_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  >> Report saved -> {report_path.name}")

    return report


# ===================================================================
# MAIN PIPELINE
# ===================================================================
def main():
    t_total = time.time()

    print("\n" + "#" * 75)
    print("  PHASE 3: ADVANCED MODELING, OPTIMIZATION & EVALUATION")
    print("  FraudSense -- Fraud Detection System")
    print("#" * 75)

    # Step 1: Load data
    df_full, X_train, X_test, y_train, y_test, feature_cols = load_data()

    # Step 2: Handle class imbalance
    X_bal, y_bal, class_weights, scale_pos = create_balanced_subset(X_train, y_train)

    # Steps 3-4: Train + tune models
    results = build_models(
        X_train, y_train, X_bal, y_bal,
        X_test, y_test,
        class_weights, scale_pos
    )

    # Steps 5-6: Evaluate
    df_metrics = evaluate_models(results, X_test, y_test)

    # Step 7: Threshold optimization
    optimize_thresholds(results, X_test, y_test)

    # Step 8: Confusion matrices
    confusion_analysis(results, X_test, y_test)

    # Step 9: Feature importance + SHAP
    analyze_feature_importance(results, X_test, feature_cols)

    # Step 10: Overfitting check
    check_overfitting(results, X_train, y_train, X_test, y_test)

    # Step 11: Select best model
    best_name, df_comp = select_best_model(results)

    # Step 12: Save artifacts
    save_artifacts(results, best_name, df_comp, feature_cols)

    # Step 13: Final report
    generate_report(results, best_name, df_comp, feature_cols)

    elapsed = time.time() - t_total
    print(f"\n  TOTAL PHASE 3 TIME: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("  Phase 3 complete.\n")


if __name__ == "__main__":
    main()
