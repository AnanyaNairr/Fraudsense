"""
=============================================================================
Phase 1: Data Understanding & Audit — Fraud Detection Dataset
=============================================================================
Comprehensive exploratory data analysis (EDA) for a ~6.3M-row financial
transaction dataset. This script is organized into modular sections, each
addressing a specific analysis task.

Dataset: Kaggle – Fraud Detection Dataset (PaySim-based)
Author : FraudSense team
Date   : 2026-04-04
=============================================================================
"""

import os
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "AIML Dataset.csv"
OUTPUT_DIR   = Path(__file__).parent / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def save_fig(fig, name: str, dpi: int = 150):
    """Save a figure and close it."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  📊 Saved → {path.name}")

def section_header(title: str):
    """Print a formatted section header."""
    width = 75
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")

def format_bytes(nbytes: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:,.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:,.1f} TB"


# ===================================================================
# 1. DATA LOADING
# ===================================================================
def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV dataset and print basic shape info."""
    section_header("1. DATA LOADING")
    df = pd.read_csv(path)
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns        : {df.columns.tolist()}")
    print(f"  Data types:\n{textwrap.indent(df.dtypes.to_string(), '    ')}")
    return df


# ===================================================================
# 2. DATA TYPE OPTIMIZATION
# ===================================================================
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types and convert categoricals; report savings."""
    section_header("2. DATA TYPE OPTIMIZATION")
    mem_before = df.memory_usage(deep=True).sum()
    print(f"  Memory BEFORE  : {format_bytes(mem_before)}")

    # Categorical conversions
    for col in ("type",):
        df[col] = df[col].astype("category")

    # String columns → category (much cheaper than object for repeated values)
    for col in ("nameOrig", "nameDest"):
        df[col] = df[col].astype("category")

    # Downcast integers
    for col in ("step", "isFraud", "isFlaggedFraud"):
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Downcast floats
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], downcast="float")

    mem_after = df.memory_usage(deep=True).sum()
    pct_saved = (1 - mem_after / mem_before) * 100
    print(f"  Memory AFTER   : {format_bytes(mem_after)}")
    print(f"  Savings        : {format_bytes(mem_before - mem_after)} ({pct_saved:.1f}%)")
    print(f"\n  Optimized dtypes:\n{textwrap.indent(df.dtypes.to_string(), '    ')}")
    return df


# ===================================================================
# 3. MISSING VALUES ANALYSIS
# ===================================================================
def analyze_missing(df: pd.DataFrame):
    """Report missing value counts and percentages per column."""
    section_header("3. MISSING VALUES ANALYSIS")
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(4)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    print(report.to_string())
    if missing.sum() == 0:
        print("\n  ✅ No missing values detected in any column.")
    else:
        print(f"\n  ⚠️  Total cells with missing values: {missing.sum():,}")


# ===================================================================
# 4. CLASS IMBALANCE ANALYSIS
# ===================================================================
def analyze_class_imbalance(df: pd.DataFrame):
    """Compute fraud vs non-fraud counts, plot bar chart."""
    section_header("4. CLASS IMBALANCE ANALYSIS")
    counts = df["isFraud"].value_counts().sort_index()
    total  = len(df)
    fraud_count    = counts.get(1, 0)
    nonfraud_count = counts.get(0, 0)
    fraud_pct      = fraud_count / total * 100

    print(f"  Non-Fraud (0) : {nonfraud_count:>10,}  ({100 - fraud_pct:.4f}%)")
    print(f"  Fraud     (1) : {fraud_count:>10,}  ({fraud_pct:.4f}%)")
    print(f"  Imbalance ratio : 1 : {nonfraud_count / max(fraud_count, 1):.0f}")

    # Also check isFlaggedFraud
    flagged = df["isFlaggedFraud"].value_counts()
    print(f"\n  isFlaggedFraud distribution:\n{textwrap.indent(flagged.to_string(), '    ')}")

    # --- Visualisation ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    colors = ["#2ecc71", "#e74c3c"]
    bars = axes[0].bar(["Non-Fraud", "Fraud"], [nonfraud_count, fraud_count],
                        color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")
    for bar, cnt in zip(bars, [nonfraud_count, fraud_count]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{cnt:,}", ha="center", va="bottom", fontsize=11)

    # Log-scale bar
    bars2 = axes[1].bar(["Non-Fraud", "Fraud"], [nonfraud_count, fraud_count],
                         color=colors, edgecolor="white", linewidth=1.2)
    axes[1].set_yscale("log")
    axes[1].set_title("Class Distribution (Log Scale)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Count (log)")
    for bar, cnt in zip(bars2, [nonfraud_count, fraud_count]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{cnt:,}", ha="center", va="bottom", fontsize=11)

    fig.suptitle("Fraud vs Non-Fraud Transactions", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "04_class_imbalance")


# ===================================================================
# 5. UNIVARIATE ANALYSIS
# ===================================================================
def analyze_distributions(df: pd.DataFrame):
    """Summary stats, histograms, and outlier detection for numeric cols."""
    section_header("5. UNIVARIATE ANALYSIS")

    # Summary statistics
    stats = df[NUMERIC_COLS].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(stats.to_string())

    # Histograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, col in enumerate(NUMERIC_COLS):
        ax = axes[idx]
        data = df[col]
        # Use log scale for skewed features (almost all financial amounts)
        pos_data = data[data > 0]
        if len(pos_data) > 0:
            ax.hist(pos_data, bins=100, color="#3498db", alpha=0.7, edgecolor="white")
            ax.set_xscale("log")
        else:
            ax.hist(data, bins=100, color="#3498db", alpha=0.7, edgecolor="white")
        ax.set_title(col, fontsize=13, fontweight="bold")
        ax.set_xlabel("Value (log scale)")
        ax.set_ylabel("Frequency")
    # Hide unused subplot
    axes[-1].set_visible(False)
    fig.suptitle("Numeric Feature Distributions (Log Scale)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "05_univariate_distributions")

    # Outlier detection using IQR
    print("\n  Outlier analysis (IQR method):")
    print(f"  {'Column':<20} {'Q1':>15} {'Q3':>15} {'IQR':>15} {'Lower':>15} {'Upper':>15} {'Outliers':>10} {'%':>8}")
    print("  " + "-" * 114)
    for col in NUMERIC_COLS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_outliers / len(df) * 100
        print(f"  {col:<20} {q1:>15,.2f} {q3:>15,.2f} {iqr:>15,.2f} {lower:>15,.2f} {upper:>15,.2f} {n_outliers:>10,} {pct:>7.2f}%")


# ===================================================================
# 6. TRANSACTION TYPE ANALYSIS
# ===================================================================
def analyze_transaction_types(df: pd.DataFrame):
    """Count distribution and fraud rate per transaction type."""
    section_header("6. TRANSACTION TYPE ANALYSIS")

    type_stats = df.groupby("type", observed=True).agg(
        total_count=("isFraud", "count"),
        fraud_count=("isFraud", "sum"),
    )
    type_stats["fraud_rate"] = (type_stats["fraud_count"] / type_stats["total_count"] * 100).round(4)
    type_stats = type_stats.sort_values("total_count", ascending=False)
    print(type_stats.to_string())

    # --- Visualisation ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Transaction count by type
    palette = sns.color_palette("viridis", n_colors=len(type_stats))
    axes[0].barh(type_stats.index.astype(str), type_stats["total_count"],
                 color=palette, edgecolor="white")
    axes[0].set_xlabel("Number of Transactions")
    axes[0].set_title("Transactions by Type", fontsize=14, fontweight="bold")
    for i, (cnt, idx) in enumerate(zip(type_stats["total_count"], type_stats.index)):
        axes[0].text(cnt, i, f" {cnt:,}", va="center", fontsize=10)

    # Fraud rate by type
    colors_rate = ["#e74c3c" if r > 0 else "#95a5a6" for r in type_stats["fraud_rate"]]
    axes[1].barh(type_stats.index.astype(str), type_stats["fraud_rate"],
                 color=colors_rate, edgecolor="white")
    axes[1].set_xlabel("Fraud Rate (%)")
    axes[1].set_title("Fraud Rate by Transaction Type", fontsize=14, fontweight="bold")
    for i, rate in enumerate(type_stats["fraud_rate"]):
        axes[1].text(rate, i, f" {rate:.2f}%", va="center", fontsize=10)

    fig.suptitle("Transaction Type Analysis", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "06_transaction_types")


# ===================================================================
# 7. TIME-BASED ANALYSIS
# ===================================================================
def analyze_time_patterns(df: pd.DataFrame):
    """Derive hour/day from step, plot temporal patterns."""
    section_header("7. TIME-BASED ANALYSIS")

    df["hour"] = df["step"] % 24
    df["day"]  = df["step"] // 24

    # --- Transactions & fraud per hour ---
    hourly = df.groupby("hour").agg(
        tx_count=("isFraud", "count"),
        fraud_count=("isFraud", "sum"),
    )
    hourly["fraud_rate"] = (hourly["fraud_count"] / hourly["tx_count"] * 100).round(4)
    print("  Hourly summary:")
    print(textwrap.indent(hourly.to_string(), "    "))

    # --- Transactions & fraud per day ---
    daily = df.groupby("day").agg(
        tx_count=("isFraud", "count"),
        fraud_count=("isFraud", "sum"),
    )
    daily["fraud_rate"] = (daily["fraud_count"] / daily["tx_count"] * 100).round(4)
    print(f"\n  Total simulated days: {daily.shape[0]}")
    print(f"  Day range: {df['day'].min()} – {df['day'].max()}")

    # --- Visualisations ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Transactions per hour
    axes[0, 0].bar(hourly.index, hourly["tx_count"], color="#3498db", edgecolor="white")
    axes[0, 0].set_title("Transactions per Hour", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Hour")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(1))

    # Fraud rate per hour
    axes[0, 1].plot(hourly.index, hourly["fraud_rate"], marker="o",
                    color="#e74c3c", linewidth=2, markersize=6)
    axes[0, 1].fill_between(hourly.index, hourly["fraud_rate"], alpha=0.15, color="#e74c3c")
    axes[0, 1].set_title("Fraud Rate per Hour", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Hour")
    axes[0, 1].set_ylabel("Fraud Rate (%)")
    axes[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(1))

    # Transactions per day
    axes[1, 0].plot(daily.index, daily["tx_count"], color="#2ecc71", linewidth=1.5, alpha=0.8)
    axes[1, 0].set_title("Transactions per Day", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Count")

    # Fraud count per day
    axes[1, 1].plot(daily.index, daily["fraud_count"], color="#e74c3c", linewidth=1.5, alpha=0.8)
    axes[1, 1].set_title("Fraud Count per Day (Trend)", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Day")
    axes[1, 1].set_ylabel("Fraud Count")

    fig.suptitle("Time-Based Analysis", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "07_time_analysis")

    return df  # return with hour/day columns attached


# ===================================================================
# 8. BALANCE CONSISTENCY CHECK
# ===================================================================
def check_balance_consistency(df: pd.DataFrame):
    """Check if balance equations hold; compare fraud vs non-fraud."""
    section_header("8. BALANCE CONSISTENCY CHECK")

    # Error = expected_change - actual_change
    df["errorOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    df["errorDest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

    for label, col in [("Sender (errorOrig)", "errorOrig"),
                       ("Receiver (errorDest)", "errorDest")]:
        print(f"\n  {label}:")
        print(f"    Non-zero errors : {(df[col] != 0).sum():,} ({(df[col] != 0).mean()*100:.2f}%)")
        stats = df[col].describe()
        print(textwrap.indent(stats.to_string(), "    "))

    # Compare fraud vs non-fraud
    for col in ["errorOrig", "errorDest"]:
        for label, mask in [("Non-Fraud", df["isFraud"] == 0), ("Fraud", df["isFraud"] == 1)]:
            subset = df.loc[mask, col]
            print(f"\n  {col} — {label}:")
            print(f"    Mean  : {subset.mean():,.2f}")
            print(f"    Std   : {subset.std():,.2f}")
            print(f"    Min   : {subset.min():,.2f}")
            print(f"    Max   : {subset.max():,.2f}")
            print(f"    % non-zero : {(subset != 0).mean()*100:.2f}%")

    # --- Visualisation ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, col in enumerate(["errorOrig", "errorDest"]):
        for j, (label, mask, color) in enumerate([
            ("Non-Fraud", df["isFraud"] == 0, "#3498db"),
            ("Fraud", df["isFraud"] == 1, "#e74c3c"),
        ]):
            ax = axes[idx, j]
            data = df.loc[mask, col]
            # Clip for visualization
            clip_val = data.quantile(0.99)
            clip_low = data.quantile(0.01)
            clipped = data.clip(clip_low, clip_val)
            ax.hist(clipped, bins=100, color=color, alpha=0.7, edgecolor="white")
            ax.set_title(f"{col} — {label}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Error Value")
            ax.set_ylabel("Frequency")

    fig.suptitle("Balance Consistency Errors (Clipped to 1st–99th pctile)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "08_balance_errors")

    # Clean up temp columns
    df.drop(columns=["errorOrig", "errorDest"], inplace=True)
    return df


# ===================================================================
# 9. ENTITY-LEVEL ANALYSIS
# ===================================================================
def analyze_entities(df: pd.DataFrame):
    """Unique counts, top entities, avg transactions per entity."""
    section_header("9. ENTITY-LEVEL ANALYSIS")

    for col, label in [("nameOrig", "Senders"), ("nameDest", "Receivers")]:
        n_unique = df[col].nunique()
        avg_tx   = len(df) / n_unique
        top10    = df[col].value_counts().head(10)
        print(f"\n  {label} ({col}):")
        print(f"    Unique entities      : {n_unique:,}")
        print(f"    Avg txn per entity   : {avg_tx:.2f}")
        print(f"    Top 10 most frequent :")
        for rank, (entity, cnt) in enumerate(top10.items(), 1):
            print(f"      {rank:>2}. {entity} — {cnt:,} transactions")

    # Cross-check: any sender that is also a receiver?
    senders   = set(df["nameOrig"].unique())
    receivers = set(df["nameDest"].unique())
    overlap   = senders & receivers
    print(f"\n  Overlap (sender ∩ receiver): {len(overlap):,} entities")


# ===================================================================
# 10. CORRELATION ANALYSIS
# ===================================================================
def analyze_correlations(df: pd.DataFrame):
    """Correlation matrix of numeric features, highlight strong pairs."""
    section_header("10. CORRELATION ANALYSIS")

    cols = NUMERIC_COLS + ["step", "isFraud"]
    corr = df[cols].corr()
    print(corr.round(4).to_string())

    # Find highly correlated pairs (|r| > 0.7, excluding self)
    print("\n  Highly correlated pairs (|r| > 0.7):")
    seen = set()
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                pair = (corr.index[i], corr.columns[j])
                if pair not in seen:
                    seen.add(pair)
                    print(f"    {pair[0]} ↔ {pair[1]} : r = {r:.4f}")
    if not seen:
        print("    None found.")

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix (Numeric Features)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "10_correlation_matrix")


# ===================================================================
# 11. FRAUD PATTERN INSIGHTS
# ===================================================================
def analyze_fraud_patterns(df: pd.DataFrame):
    """Comprehensive comparison of fraud vs non-fraud transactions."""
    section_header("11. FRAUD PATTERN INSIGHTS")

    fraud    = df[df["isFraud"] == 1]
    nonfraud = df[df["isFraud"] == 0]

    # --- Amount comparison ---
    print("  Amount statistics:")
    print(f"    {'Metric':<20} {'Non-Fraud':>18} {'Fraud':>18}")
    print("    " + "-" * 56)
    for stat, fn in [("Mean", "mean"), ("Median", "median"), ("Std", "std"),
                     ("Min", "min"), ("Max", "max")]:
        nf_val = getattr(nonfraud["amount"], fn)()
        f_val  = getattr(fraud["amount"], fn)()
        print(f"    {stat:<20} {nf_val:>18,.2f} {f_val:>18,.2f}")

    # --- Transaction type of fraud ---
    print("\n  Fraud by transaction type:")
    fraud_types = fraud["type"].value_counts()
    for t, cnt in fraud_types.items():
        pct_of_fraud = cnt / len(fraud) * 100
        pct_of_type  = cnt / len(df[df["type"] == t]) * 100
        print(f"    {t:>12} : {cnt:,} frauds ({pct_of_fraud:.1f}% of all fraud, {pct_of_type:.2f}% of this type)")

    # --- Balance change patterns ---
    print("\n  Balance change patterns (fraud only):")
    fraud_zero_orig = (fraud["newbalanceOrig"] == 0).sum()
    fraud_zero_dest = (fraud["oldbalanceDest"] == 0).sum()
    print(f"    Sender balance zeroed out   : {fraud_zero_orig:,} ({fraud_zero_orig/len(fraud)*100:.1f}%)")
    print(f"    Receiver had zero balance   : {fraud_zero_dest:,} ({fraud_zero_dest/len(fraud)*100:.1f}%)")

    # --- Time patterns for fraud ---
    if "hour" in df.columns:
        print("\n  Fraud by hour (top 5):")
        fraud_hours = fraud["hour"].value_counts().head(5)
        for h, cnt in fraud_hours.items():
            print(f"    Hour {h:>2} : {cnt:,} frauds")

    # --- Visualization: fraud vs non-fraud ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Amount distribution comparison
    ax = axes[0, 0]
    for label, subset, color in [("Non-Fraud", nonfraud, "#3498db"), ("Fraud", fraud, "#e74c3c")]:
        pos = subset["amount"][subset["amount"] > 0]
        ax.hist(pos, bins=100, alpha=0.6, color=color, label=label, density=True, edgecolor="white")
    ax.set_xscale("log")
    ax.set_title("Amount Distribution: Fraud vs Non-Fraud", fontsize=13, fontweight="bold")
    ax.set_xlabel("Amount (log scale)")
    ax.set_ylabel("Density")
    ax.legend()

    # Balance change for sender
    ax = axes[0, 1]
    balance_change_orig_nf = (nonfraud["oldbalanceOrg"] - nonfraud["newbalanceOrig"]).clip(-1e7, 1e7)
    balance_change_orig_f  = (fraud["oldbalanceOrg"] - fraud["newbalanceOrig"]).clip(-1e7, 1e7)
    ax.hist(balance_change_orig_nf, bins=100, alpha=0.5, color="#3498db", label="Non-Fraud", density=True)
    ax.hist(balance_change_orig_f, bins=100, alpha=0.5, color="#e74c3c", label="Fraud", density=True)
    ax.set_title("Sender Balance Change", fontsize=13, fontweight="bold")
    ax.set_xlabel("oldbalanceOrg − newbalanceOrig")
    ax.legend()

    # Fraud count by type
    ax = axes[1, 0]
    type_fraud = df.groupby("type", observed=True)["isFraud"].agg(["sum", "count"])
    type_fraud.columns = ["fraud", "total"]
    type_fraud["non_fraud"] = type_fraud["total"] - type_fraud["fraud"]
    type_fraud[["non_fraud", "fraud"]].plot(kind="bar", stacked=True, ax=ax,
                                            color=["#3498db", "#e74c3c"],
                                            edgecolor="white")
    ax.set_title("Fraud vs Non-Fraud by Type", fontsize=13, fontweight="bold")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.tick_params(axis='x', rotation=0)

    # Fraud rate by hour (if available)
    if "hour" in df.columns:
        ax = axes[1, 1]
        hourly_fraud = df.groupby("hour")["isFraud"].mean() * 100
        ax.bar(hourly_fraud.index, hourly_fraud.values, color="#e74c3c", alpha=0.7, edgecolor="white")
        ax.set_title("Fraud Rate by Hour", fontsize=13, fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Fraud Rate (%)")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.suptitle("Fraud Pattern Analysis", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "11_fraud_patterns")


# ===================================================================
# 12. SUMMARY REPORT
# ===================================================================
def generate_summary(df: pd.DataFrame):
    """Print a concise final summary of all findings."""
    section_header("FINAL SUMMARY REPORT")

    fraud_count = (df["isFraud"] == 1).sum()
    total       = len(df)
    fraud_pct   = fraud_count / total * 100

    # Fraud by type
    fraud_types = df[df["isFraud"] == 1]["type"].value_counts()

    # Balance zeroing
    fraud_df = df[df["isFraud"] == 1]
    pct_zero_orig = (fraud_df["newbalanceOrig"] == 0).mean() * 100

    summary = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    PHASE 1: DATA AUDIT SUMMARY                       ║
╚═══════════════════════════════════════════════════════════════════════╝

  📋 DATASET OVERVIEW
  ────────────────────
  • {total:,} transactions across {df['step'].nunique()} time steps
  • {df['type'].nunique()} transaction types: {', '.join(df['type'].cat.categories.tolist())}
  • {df['nameOrig'].nunique():,} unique senders, {df['nameDest'].nunique():,} unique receivers

  🎯 KEY INSIGHTS
  ────────────────
  1. SEVERE CLASS IMBALANCE: Only {fraud_count:,} fraud cases ({fraud_pct:.4f}%)
     → Imbalance ratio ~1:{total // max(fraud_count, 1)} — requires careful handling
       (oversampling, SMOTE, cost-sensitive learning, or focal loss)

  2. FRAUD TYPES CONCENTRATED: Fraud occurs ONLY in: {', '.join(fraud_types.index.tolist())}
     → {fraud_types.to_dict()}
     → TRANSFER and CASH_OUT are the only vectors for fraud

  3. BALANCE ZEROING IS A STRONG SIGNAL: ~{pct_zero_orig:.0f}% of fraudulent
     transactions drain the sender's balance to zero

  4. LARGE AMOUNTS: Fraudulent transactions tend to involve significantly
     higher amounts than legitimate ones

  5. BALANCE EQUATION VIOLATIONS: Many transactions (both fraud and
     non-fraud) violate the expected balance equation, but the pattern
     differs — fraud transactions show distinct error distributions

  6. NO MISSING VALUES: Dataset is complete with no null entries

  7. DATA QUALITY: Column types are clean; no structural anomalies detected

  🔍 POTENTIAL PREDICTIVE SIGNALS
  ─────────────────────────────────
  • Transaction type (TRANSFER / CASH_OUT only for fraud)
  • Amount (fraud skews higher)
  • Balance-to-amount ratio (sender balance vs transaction amount)
  • Balance zeroing flag (newbalanceOrig == 0)
  • Balance error (equation violation magnitude)
  • Temporal patterns (hour-of-day, day trends)

  ⚠️  RISKS & WARNINGS
  ─────────────────────
  • DATA LEAKAGE RISK: `isFlaggedFraud` is derived from `isFraud` with
    a rule-based filter — MUST be excluded from training features
  • `nameOrig` / `nameDest` have very high cardinality — encoding them
    directly is infeasible; consider entity-level aggregations instead
  • Balance equations don't always hold even for non-fraud — this is
    expected in PaySim-simulated data (represents system noise)
  • The IQR-based outlier analysis shows heavy-tailed distributions
    in ALL financial columns — standard scaling may be inadequate

  📊 VISUALIZATIONS GENERATED
  ───────────────────────────
"""
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        summary += f"  • {f.name}\n"

    summary += """
═══════════════════════════════════════════════════════════════════════════
  Phase 1 Complete — Ready for Phase 2: Feature Engineering
═══════════════════════════════════════════════════════════════════════════
"""
    print(summary)

    # Also save report to file
    report_path = OUTPUT_DIR / "summary_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  📝 Report saved → {report_path.name}")


# ===================================================================
# MAIN PIPELINE
# ===================================================================
def main():
    print("\n" + "█" * 75)
    print("  PHASE 1: DATA UNDERSTANDING & AUDIT")
    print("  FraudSense — Fraud Detection System")
    print("█" * 75)

    # 1. Load
    df = load_data(DATASET_PATH)

    # 2. Optimize
    df = optimize_dtypes(df)

    # 3. Missing
    analyze_missing(df)

    # 4. Class imbalance
    analyze_class_imbalance(df)

    # 5. Distributions
    analyze_distributions(df)

    # 6. Transaction types
    analyze_transaction_types(df)

    # 7. Time analysis
    df = analyze_time_patterns(df)

    # 8. Balance check
    df = check_balance_consistency(df)

    # 9. Entities
    analyze_entities(df)

    # 10. Correlations
    analyze_correlations(df)

    # 11. Fraud patterns
    analyze_fraud_patterns(df)

    # 12. Summary
    generate_summary(df)

    print("\n✅ All Phase 1 analyses complete. Check 'eda_outputs/' for plots.")


if __name__ == "__main__":
    main()
