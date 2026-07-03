"""
=============================================================================
Phase 2: Feature Engineering — Fraud Detection System
=============================================================================
Transforms the raw transaction dataset into a clean, model-ready feature
matrix based on Phase 1 EDA insights. Every feature group is implemented
as an independent, modular function for clarity and testability.

Key design decisions:
  - isFlaggedFraud is DROPPED (data leakage)
  - nameOrig / nameDest are aggregated, then dropped (high cardinality)
  - All operations are vectorized for 6.3M-row performance
  - Memory is optimised incrementally throughout the pipeline

Dataset: Kaggle - Fraud Detection Dataset (PaySim-based)
Author : FraudSense team
Date   : 2026-04-04
=============================================================================
"""

import time
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "AIML Dataset.csv"
OUTPUT_DIR   = Path(__file__).parent / "fe_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ORIGINAL_NUMERIC = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]

# Leakage columns to drop BEFORE any feature engineering
LEAKAGE_COLS = ["isFlaggedFraud"]

# High-risk transaction types (fraud only occurs in these)
HIGH_RISK_TYPES = ["TRANSFER", "CASH_OUT"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def section(title: str):
    """Print a formatted section header."""
    w = 75
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")


def fmt(n: int) -> str:
    """Comma-formatted integer."""
    return f"{n:,}"


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


def timer(func):
    """Decorator to time each pipeline step."""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"  [TIME] {func.__name__}: {elapsed:.1f}s")
        return result
    return wrapper


# ===================================================================
# 1. DATA LOADING & INITIAL CLEANUP
# ===================================================================
@timer
def load_and_clean(path: Path) -> pd.DataFrame:
    """Load dataset, drop leakage columns, optimize dtypes."""
    section("1. DATA LOADING & INITIAL CLEANUP")

    df = pd.read_csv(path)
    print(f"  Loaded: {fmt(len(df))} rows x {df.shape[1]} columns")
    mem_raw = df.memory_usage(deep=True).sum()
    print(f"  Raw memory: {format_bytes(mem_raw)}")

    # Drop leakage columns
    df.drop(columns=LEAKAGE_COLS, inplace=True, errors="ignore")
    print(f"  Dropped leakage columns: {LEAKAGE_COLS}")

    # Optimise dtypes
    df["type"] = df["type"].astype("category")
    df["step"] = pd.to_numeric(df["step"], downcast="integer")
    df["isFraud"] = pd.to_numeric(df["isFraud"], downcast="integer")
    for col in ORIGINAL_NUMERIC:
        df[col] = pd.to_numeric(df[col], downcast="float")

    mem_opt = df.memory_usage(deep=True).sum()
    print(f"  Optimised memory: {format_bytes(mem_opt)} "
          f"(saved {format_bytes(mem_raw - mem_opt)})")

    return df


# ===================================================================
# 2. BASIC DERIVED FEATURES
# ===================================================================
@timer
def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Balance differences, amount-to-balance ratio, zero-balance flag."""
    section("2. BASIC DERIVED FEATURES")

    # Balance change on sender side (how much left the sender)
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

    # Balance change on receiver side (how much the receiver gained)
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Ratio of amount to sender's available balance (+1 to avoid div-by-zero)
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # Binary flag: did the sender's balance drop to exactly zero?
    df["is_zero_balance"] = (df["newbalanceOrig"] == 0).astype(np.int8)

    features = ["balance_diff_orig", "balance_diff_dest",
                "amount_to_balance_ratio", "is_zero_balance"]
    print(f"  Created {len(features)} features: {features}")
    for f in features:
        print(f"    {f:30s}  mean={df[f].mean():>12.2f}  std={df[f].std():>12.2f}")

    return df


# ===================================================================
# 3. BALANCE ERROR FEATURES
# ===================================================================
@timer
def create_balance_error_features(df: pd.DataFrame) -> pd.DataFrame:
    """Detect balance-equation violations (inconsistency signals)."""
    section("3. BALANCE ERROR FEATURES")

    # Sender error: oldBal - amount should equal newBal
    df["error_orig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]

    # Receiver error: oldBal + amount should equal newBal
    df["error_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Absolute versions (magnitude of inconsistency)
    df["abs_error_orig"] = df["error_orig"].abs()
    df["abs_error_dest"] = df["error_dest"].abs()

    features = ["error_orig", "error_dest", "abs_error_orig", "abs_error_dest"]
    print(f"  Created {len(features)} features: {features}")

    # Quick fraud vs non-fraud comparison
    for f in ["error_orig", "error_dest"]:
        fraud_mean = df.loc[df["isFraud"] == 1, f].mean()
        nonfraud_mean = df.loc[df["isFraud"] == 0, f].mean()
        print(f"    {f:20s}  fraud_mean={fraud_mean:>14.2f}  non-fraud_mean={nonfraud_mean:>14.2f}")

    return df


# ===================================================================
# 4. TRANSACTION TYPE FEATURES
# ===================================================================
@timer
def create_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode transaction type + high-risk flag."""
    section("4. TRANSACTION TYPE FEATURES")

    # One-hot encoding
    type_dummies = pd.get_dummies(df["type"], prefix="type", dtype=np.int8)
    df = pd.concat([df, type_dummies], axis=1)
    ohe_cols = type_dummies.columns.tolist()
    print(f"  One-hot columns: {ohe_cols}")

    # High-risk type flag
    df["is_high_risk_type"] = df["type"].isin(HIGH_RISK_TYPES).astype(np.int8)
    print(f"  is_high_risk_type: {df['is_high_risk_type'].mean()*100:.2f}% of transactions")

    return df


# ===================================================================
# 5. TEMPORAL FEATURES
# ===================================================================
@timer
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, day, and night-flag from the step column."""
    section("5. TEMPORAL FEATURES")

    df["hour"] = (df["step"] % 24).astype(np.int8)
    df["day"]  = (df["step"] // 24).astype(np.int16)

    # Night flag: hours 0-6 (Phase 1 showed fraud peaks at 3-5 AM)
    df["is_night"] = (df["hour"].between(0, 6)).astype(np.int8)

    features = ["hour", "day", "is_night"]
    print(f"  Created: {features}")
    print(f"  Night transactions: {df['is_night'].mean()*100:.2f}%")
    print(f"  Night fraud rate:   "
          f"{df.loc[df['is_night']==1, 'isFraud'].mean()*100:.4f}%")
    print(f"  Day fraud rate:     "
          f"{df.loc[df['is_night']==0, 'isFraud'].mean()*100:.4f}%")

    return df


# ===================================================================
# 6. SENDER-LEVEL AGGREGATED FEATURES
# ===================================================================
@timer
def create_sender_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioural aggregations for each sender (nameOrig)."""
    section("6. SENDER-LEVEL AGGREGATED FEATURES")

    sender_agg = df.groupby("nameOrig", observed=True)["amount"].agg(
        txn_count_sender="count",
        avg_amount_sender="mean",
        max_amount_sender="max",
        std_amount_sender="std",
    )
    # Fill NaN std (single-transaction entities) with 0
    sender_agg["std_amount_sender"] = sender_agg["std_amount_sender"].fillna(0)

    # Downcast to save memory
    sender_agg["txn_count_sender"] = pd.to_numeric(
        sender_agg["txn_count_sender"], downcast="integer"
    )
    for col in ["avg_amount_sender", "max_amount_sender", "std_amount_sender"]:
        sender_agg[col] = pd.to_numeric(sender_agg[col], downcast="float")

    df = df.merge(sender_agg, on="nameOrig", how="left")

    features = ["txn_count_sender", "avg_amount_sender",
                "max_amount_sender", "std_amount_sender"]
    print(f"  Created {len(features)} features: {features}")
    for f in features:
        print(f"    {f:25s}  mean={df[f].mean():>14.2f}")

    return df


# ===================================================================
# 7. RECEIVER-LEVEL AGGREGATED FEATURES
# ===================================================================
@timer
def create_receiver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioural aggregations for each receiver (nameDest)."""
    section("7. RECEIVER-LEVEL AGGREGATED FEATURES")

    recv_agg = df.groupby("nameDest", observed=True)["amount"].agg(
        txn_count_receiver="count",
        avg_amount_receiver="mean",
    )
    recv_agg["txn_count_receiver"] = pd.to_numeric(
        recv_agg["txn_count_receiver"], downcast="integer"
    )
    recv_agg["avg_amount_receiver"] = pd.to_numeric(
        recv_agg["avg_amount_receiver"], downcast="float"
    )

    df = df.merge(recv_agg, on="nameDest", how="left")

    features = ["txn_count_receiver", "avg_amount_receiver"]
    print(f"  Created {len(features)} features: {features}")
    for f in features:
        print(f"    {f:25s}  mean={df[f].mean():>14.2f}")

    return df


# ===================================================================
# 8. VELOCITY FEATURES (BEHAVIORAL)
# ===================================================================
@timer
def create_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions per sender per time step (burst detection)."""
    section("8. VELOCITY FEATURES (BEHAVIORAL)")

    velocity = df.groupby(["nameOrig", "step"], observed=True).agg(
        txn_per_step_sender=("amount", "count"),
        amt_per_step_sender=("amount", "sum"),
    ).reset_index()

    velocity["txn_per_step_sender"] = pd.to_numeric(
        velocity["txn_per_step_sender"], downcast="integer"
    )
    velocity["amt_per_step_sender"] = pd.to_numeric(
        velocity["amt_per_step_sender"], downcast="float"
    )

    df = df.merge(velocity, on=["nameOrig", "step"], how="left")

    features = ["txn_per_step_sender", "amt_per_step_sender"]
    print(f"  Created {len(features)} features: {features}")
    print(f"    txn_per_step_sender max = {df['txn_per_step_sender'].max()}")
    print(f"    amt_per_step_sender mean = {df['amt_per_step_sender'].mean():,.2f}")

    return df


# ===================================================================
# 9. INTERACTION FEATURES
# ===================================================================
@timer
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combined signals from multiple dimensions."""
    section("9. INTERACTION FEATURES")

    # High-amount flag: top 5% of transaction amounts
    amount_95 = df["amount"].quantile(0.95)
    df["high_amount_flag"] = (df["amount"] >= amount_95).astype(np.int8)
    print(f"  Amount 95th percentile threshold: {amount_95:,.2f}")

    # Interaction: high amount AND balance zeroed
    df["high_amount_and_zero_balance"] = (
        (df["high_amount_flag"] == 1) & (df["is_zero_balance"] == 1)
    ).astype(np.int8)

    # Interaction: high amount AND high-risk type
    df["high_amount_and_high_risk_type"] = (
        (df["high_amount_flag"] == 1) & (df["is_high_risk_type"] == 1)
    ).astype(np.int8)

    features = ["high_amount_flag", "high_amount_and_zero_balance",
                "high_amount_and_high_risk_type"]
    print(f"  Created {len(features)} features: {features}")
    for f in features:
        pct = df[f].mean() * 100
        fraud_rate = df.loc[df[f] == 1, "isFraud"].mean() * 100 if df[f].sum() > 0 else 0
        print(f"    {f:40s}  prevalence={pct:.2f}%  fraud_rate={fraud_rate:.2f}%")

    return df


# ===================================================================
# 10. LOG-TRANSFORM SKEWED FEATURES
# ===================================================================
@timer
def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transform to heavily skewed financial features."""
    section("10. LOG-TRANSFORM SKEWED FEATURES")

    log_targets = ["amount", "oldbalanceOrg", "newbalanceOrig",
                   "oldbalanceDest", "newbalanceDest",
                   "abs_error_orig", "abs_error_dest"]

    created = []
    for col in log_targets:
        new_col = f"log_{col}"
        df[new_col] = np.log1p(df[col].clip(lower=0))
        df[new_col] = pd.to_numeric(df[new_col], downcast="float")
        created.append(new_col)

    print(f"  Created {len(created)} log features:")
    for c in created:
        print(f"    {c:30s}  mean={df[c].mean():>10.4f}  max={df[c].max():>10.4f}")

    return df


# ===================================================================
# 11. DROP RAW HIGH-CARDINALITY & UNUSED COLUMNS
# ===================================================================
@timer
def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove raw entity IDs and the original type column."""
    section("11. DROP UNUSED COLUMNS")

    to_drop = ["nameOrig", "nameDest", "type"]
    existing = [c for c in to_drop if c in df.columns]
    df.drop(columns=existing, inplace=True)
    print(f"  Dropped: {existing}")
    print(f"  Remaining columns: {df.shape[1]}")

    return df


# ===================================================================
# 12. DATA CLEANING & VALIDATION
# ===================================================================
@timer
def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf, check NaN, validate distributions."""
    section("12. DATA CLEANING & VALIDATION")

    # Replace infinities with large finite values
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[num_cols]).sum().sum()
    print(f"  Infinite values found: {fmt(inf_count)}")
    if inf_count > 0:
        for col in num_cols:
            mask = np.isinf(df[col])
            if mask.any():
                col_max = df.loc[~mask, col].max()
                col_min = df.loc[~mask, col].min()
                df.loc[df[col] == np.inf, col] = col_max
                df.loc[df[col] == -np.inf, col] = col_min
                print(f"    Fixed {mask.sum()} infs in {col}")

    # Check NaN
    nan_count = df.isnull().sum().sum()
    print(f"  NaN values found: {fmt(nan_count)}")
    if nan_count > 0:
        # Fill numeric NaN with 0
        df[num_cols] = df[num_cols].fillna(0)
        print(f"    Filled NaN with 0")

    # Final validation
    assert df.isnull().sum().sum() == 0, "Still have NaN values!"
    assert np.isinf(df.select_dtypes(include=[np.number]).values).sum() == 0, "Still have inf!"
    print("  PASSED: No NaN or Inf values remain")

    # Memory report
    mem = df.memory_usage(deep=True).sum()
    print(f"  Final memory usage: {format_bytes(mem)}")

    return df


# ===================================================================
# 13. FEATURE SUMMARY & CORRELATION WITH TARGET
# ===================================================================
@timer
def summarize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Print feature list, summary stats, correlation with target."""
    section("13. FEATURE SUMMARY & CORRELATION WITH TARGET")

    target = "isFraud"
    feature_cols = [c for c in df.columns if c != target]

    print(f"  Total features: {len(feature_cols)}")
    print(f"  Total samples:  {fmt(len(df))}")
    print(f"  Fraud rate:     {df[target].mean()*100:.4f}%")

    # Summary statistics
    print("\n  Feature summary statistics:")
    stats = df[feature_cols].describe().T
    stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(textwrap.indent(stats.to_string(), "    "))

    # Correlation with target
    print(f"\n  Correlation with {target} (sorted by |r|):")
    num_features = df[feature_cols].select_dtypes(include=[np.number]).columns
    corr_with_target = df[num_features].corrwith(df[target]).dropna()
    corr_sorted = corr_with_target.reindex(
        corr_with_target.abs().sort_values(ascending=False).index
    )
    print(f"    {'Feature':40s} {'Correlation':>12s} {'|r|':>8s}")
    print("    " + "-" * 62)
    for feat, r in corr_sorted.items():
        print(f"    {feat:40s} {r:>12.6f} {abs(r):>8.6f}")

    # --- Visualisation: top correlations ---
    top_n = min(20, len(corr_sorted))
    top_feats = corr_sorted.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_feats.values]
    bars = ax.barh(range(top_n), top_feats.values, color=colors,
                   edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats.index, fontsize=10)
    ax.set_xlabel("Pearson Correlation with isFraud", fontsize=12)
    ax.set_title(f"Top {top_n} Features by Correlation with Target",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
    for bar, val in zip(bars, top_feats.values):
        ax.text(val + 0.001 * np.sign(val), bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "13_feature_target_correlation")

    # --- Feature correlation heatmap (top 15 by target corr) ---
    top15 = corr_sorted.head(15).index.tolist()
    if len(top15) > 2:
        fig2, ax2 = plt.subplots(figsize=(14, 11))
        corr_matrix = df[top15 + [target]].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, square=True, linewidths=0.5,
                    ax=ax2, cbar_kws={"shrink": 0.8})
        ax2.set_title("Inter-Feature Correlation (Top 15 + Target)",
                      fontsize=14, fontweight="bold")
        fig2.tight_layout()
        save_fig(fig2, "13_top_features_correlation_heatmap")

    return df


# ===================================================================
# 14. FEATURE IMPORTANCE (BONUS)
# ===================================================================
@timer
def estimate_feature_importance(df: pd.DataFrame):
    """Quick feature importance using a lightweight model."""
    section("14. FEATURE IMPORTANCE ESTIMATION (BONUS)")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("  sklearn not available, skipping feature importance.")
        return

    target = "isFraud"
    feature_cols = [c for c in df.columns if c != target]
    num_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Subsample for speed (use stratified sample to preserve fraud ratio)
    n_sample = min(200_000, len(df))
    print(f"  Subsampling to {fmt(n_sample)} rows for quick importance estimation...")

    df_sample = df.sample(n=n_sample, random_state=42, replace=False)
    X = df_sample[num_features].values
    y = df_sample[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training quick RandomForest (n_estimators=100, max_depth=10)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"  Train accuracy: {train_score:.4f}")
    print(f"  Test accuracy:  {test_score:.4f}")

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=num_features)
    importances = importances.sort_values(ascending=False)

    print(f"\n  Top 15 features by Gini importance:")
    print(f"    {'Rank':>4s}  {'Feature':40s} {'Importance':>12s}")
    print("    " + "-" * 60)
    for rank, (feat, imp) in enumerate(importances.head(15).items(), 1):
        print(f"    {rank:>4d}  {feat:40s} {imp:>12.6f}")

    # --- Visualisation ---
    top_n = min(20, len(importances))
    fig, ax = plt.subplots(figsize=(12, 8))
    top_imp = importances.head(top_n)
    ax.barh(range(top_n), top_imp.values, color="#2ecc71",
            edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_imp.index, fontsize=10)
    ax.set_xlabel("Gini Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Features by RandomForest Importance",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    for i, (feat, val) in enumerate(top_imp.items()):
        ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "14_feature_importance_rf")

    # Save importance table
    imp_path = OUTPUT_DIR / "feature_importance.csv"
    importances.to_frame("importance").to_csv(imp_path)
    print(f"  >> Saved importance table -> {imp_path.name}")


# ===================================================================
# 15. SAVE ENGINEERED DATASET
# ===================================================================
@timer
def save_dataset(df: pd.DataFrame):
    """Save the final feature-engineered dataset."""
    section("15. SAVE ENGINEERED DATASET")

    target = "isFraud"
    feature_cols = [c for c in df.columns if c != target]

    # Save as compressed parquet for efficiency
    parquet_path = OUTPUT_DIR / "fraud_features.parquet"
    df.to_parquet(parquet_path, index=False, compression="snappy")
    print(f"  Saved parquet -> {parquet_path.name} "
          f"({format_bytes(parquet_path.stat().st_size)})")

    # Also save feature list
    feat_path = OUTPUT_DIR / "feature_list.txt"
    with open(feat_path, "w") as f:
        f.write(f"Total features: {len(feature_cols)}\n")
        f.write(f"Target: {target}\n\n")
        for i, col in enumerate(feature_cols, 1):
            dtype = str(df[col].dtype)
            f.write(f"{i:3d}. {col:40s}  ({dtype})\n")
    print(f"  Saved feature list -> {feat_path.name}")

    # Print final shape
    print(f"\n  Final dataset shape: {df.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target:   {target}")


# ===================================================================
# 16. FINAL REPORT
# ===================================================================
def generate_report(df: pd.DataFrame):
    """Generate a concise summary report."""
    section("PHASE 2 SUMMARY REPORT")

    target = "isFraud"
    feature_cols = [c for c in df.columns if c != target]
    num_features = df[feature_cols].select_dtypes(include=[np.number]).columns

    report = f"""
================================================================
  PHASE 2: FEATURE ENGINEERING SUMMARY
================================================================

  DATASET
  -------
  Rows:     {fmt(len(df))}
  Features: {len(feature_cols)}
  Target:   {target} (fraud rate = {df[target].mean()*100:.4f}%)

  FEATURE GROUPS
  --------------
  Basic derived:      4 features  (balance diffs, ratio, zero flag)
  Balance errors:     4 features  (error_orig/dest, abs versions)
  Transaction type:   6 features  (5 OHE + high-risk flag)
  Temporal:           3 features  (hour, day, is_night)
  Sender-level:       4 features  (count, avg, max, std amount)
  Receiver-level:     2 features  (count, avg amount)
  Velocity:           2 features  (txn & amount per sender per step)
  Interaction:        3 features  (high-amount combos)
  Log-transformed:    7 features  (log1p of skewed features)
  Original numeric:   6 features  (step + 5 raw financials)

  DATA QUALITY
  ------------
  NaN values:  0
  Inf values:  0
  Leakage:     isFlaggedFraud REMOVED

  MEMORY
  ------
  Final: {format_bytes(df.memory_usage(deep=True).sum())}

  OUTPUT FILES
  ------------
"""
    for f in sorted(OUTPUT_DIR.iterdir()):
        report += f"  - {f.name}\n"

    report += """
================================================================
  Phase 2 Complete -- Ready for Phase 3: Model Training
================================================================
"""
    print(report)

    report_path = OUTPUT_DIR / "phase2_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  >> Report saved -> {report_path.name}")


# ===================================================================
# MAIN PIPELINE
# ===================================================================
def main():
    t_start = time.time()

    print("\n" + "#" * 75)
    print("  PHASE 2: FEATURE ENGINEERING")
    print("  FraudSense -- Fraud Detection System")
    print("#" * 75)

    # Step 1: Load & clean
    df = load_and_clean(DATASET_PATH)

    # Step 2: Basic features
    df = create_basic_features(df)

    # Step 3: Balance errors
    df = create_balance_error_features(df)

    # Step 4: Transaction type
    df = create_type_features(df)

    # Step 5: Temporal features
    df = create_temporal_features(df)

    # Step 6: Sender aggregations
    df = create_sender_features(df)

    # Step 7: Receiver aggregations
    df = create_receiver_features(df)

    # Step 8: Velocity features
    df = create_velocity_features(df)

    # Step 9: Interaction features
    df = create_interaction_features(df)

    # Step 10: Log transforms
    df = create_log_features(df)

    # Step 11: Drop unused columns
    df = drop_unused_columns(df)

    # Step 12: Clean & validate
    df = clean_and_validate(df)

    # Step 13: Summary & correlation
    df = summarize_features(df)

    # Step 14: Feature importance (bonus)
    estimate_feature_importance(df)

    # Step 15: Save
    save_dataset(df)

    # Step 16: Report
    generate_report(df)

    elapsed = time.time() - t_start
    print(f"\n  TOTAL PIPELINE TIME: {elapsed:.1f}s")
    print("  Phase 2 complete.\n")


if __name__ == "__main__":
    main()
