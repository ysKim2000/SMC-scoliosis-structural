"""
Error Analysis — clinical characteristics of FP/FN cases
Model: lumbar_only + film + resnet101 (5-fold CV)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR   = os.path.join(BASE_DIR, "test_results", "test_20260405_004430",
                          "Rank3_lumbar_only__film__resnet101")
CSV_PATH   = os.path.join(BASE_DIR, "data", "AIS_Surgery_clean.csv")
OUT_DIR    = os.path.join(BASE_DIR, "test_results", "error_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

CLINICAL_COLS = [
    "Sex", "Age", "FVC", "FEV1", "FEV1/FVC",
    "Lenke", "Lumbar_modifier",
    "Cobb", "Thoracic_Cobb",
    "T1_12_Kyphosis", "T4_12_Kyphosis",
    "Lumbar_Cobb", "Lumbar_Cobb_Bending", "L1_S1_Lordosis",
]

# ─────────────────────────────────────────────
# 1. Load predictions + merge clinical data
# ─────────────────────────────────────────────
dfs = []
for f in sorted(glob.glob(os.path.join(PRED_DIR, "fold*_predictions.csv"))):
    d = pd.read_csv(f)
    d["fold"] = int(os.path.basename(f)[4])
    dfs.append(d)
pred_df = pd.concat(dfs, ignore_index=True)

# Derive outcome at threshold 0.5
pred_df["pred"] = (pred_df["prob"] >= 0.5).astype(int)
pred_df["outcome"] = pred_df.apply(
    lambda r: "TP" if r.label == 1 and r.pred == 1
         else "TN" if r.label == 0 and r.pred == 0
         else "FP" if r.label == 0 and r.pred == 1
         else "FN", axis=1
)

# Add bending-based gray zone variable
clin = pd.read_csv(CSV_PATH)
clin["patient_id"] = clin["ID"].astype(str)
clin["Lumbar_Reduction_pct"] = (
    (clin["Lumbar_Cobb"] - clin["Lumbar_Cobb_Bending"]) / clin["Lumbar_Cobb"] * 100
).round(1)
pred_df["patient_id"] = pred_df["patient_id"].astype(str)
df = pred_df.merge(clin[["patient_id"] + CLINICAL_COLS + ["Lumbar_Reduction_pct"]],
                   on="patient_id", how="left")

print(f"Total: {len(df)}")
print(df["outcome"].value_counts())

# ─────────────────────────────────────────────
# 2. Clinical characteristics comparison table by group
# ─────────────────────────────────────────────
numeric_cols = [
    "Age", "Cobb", "Lumbar_Cobb", "L1_S1_Lordosis",
    "Lumbar_Cobb_Bending", "Lumbar_Reduction_pct",
    "Thoracic_Cobb", "T1_12_Kyphosis", "T4_12_Kyphosis",
    "FVC", "FEV1", "prob",
]

summary_rows = []
groups = ["TP", "TN", "FP", "FN"]
for col in numeric_cols:
    row = {"feature": col}
    for g in groups:
        vals = df[df["outcome"] == g][col].dropna()
        row[f"{g}_mean"] = round(vals.mean(), 2)
        row[f"{g}_std"]  = round(vals.std(), 2)
        row[f"{g}_n"]    = len(vals)
    # FN vs TP test (sensitivity miss vs hit)
    fn_vals = df[df["outcome"] == "FN"][col].dropna()
    tp_vals = df[df["outcome"] == "TP"][col].dropna()
    if len(fn_vals) >= 3 and len(tp_vals) >= 3:
        _, p = stats.mannwhitneyu(fn_vals, tp_vals, alternative="two-sided")
        row["FN_vs_TP_p"] = round(p, 4)
    else:
        row["FN_vs_TP_p"] = np.nan
    # FP vs TN test (specificity miss vs hit)
    fp_vals = df[df["outcome"] == "FP"][col].dropna()
    tn_vals = df[df["outcome"] == "TN"][col].dropna()
    if len(fp_vals) >= 3 and len(tn_vals) >= 3:
        _, p = stats.mannwhitneyu(fp_vals, tn_vals, alternative="two-sided")
        row["FP_vs_TN_p"] = round(p, 4)
    else:
        row["FP_vs_TN_p"] = np.nan
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
out_csv = os.path.join(OUT_DIR, "error_clinical_summary.csv")
summary.to_csv(out_csv, index=False)
print(f"\nClinical comparison table saved → {out_csv}")
print(summary[["feature","TP_mean","FN_mean","FN_vs_TP_p","TN_mean","FP_mean","FP_vs_TN_p"]].to_string(index=False))

# ─────────────────────────────────────────────
# 3. Sex / Lenke / Lumbar_modifier distribution comparison
# ─────────────────────────────────────────────
cat_rows = []
for cat_col in ["Sex", "Lenke", "Lumbar_modifier"]:
    for g in groups:
        vc = df[df["outcome"] == g][cat_col].value_counts(normalize=True).round(3)
        for val, pct in vc.items():
            cat_rows.append({"feature": cat_col, "value": val, "group": g, "pct": pct,
                             "n": int(df[(df["outcome"]==g)&(df[cat_col]==val)].shape[0])})
cat_summary = pd.DataFrame(cat_rows)
cat_summary.to_csv(os.path.join(OUT_DIR, "error_categorical_summary.csv"), index=False)

# ─────────────────────────────────────────────
# 4. Visualization (1) — Boxplot: key variables by TP/FN/FP/TN
# ─────────────────────────────────────────────
plot_cols = ["Lumbar_Cobb", "Lumbar_Cobb_Bending", "Lumbar_Reduction_pct",
             "Cobb", "L1_S1_Lordosis", "prob"]
palette = {"TP": "#4CAF50", "TN": "#2196F3", "FP": "#FF9800", "FN": "#F44336"}
order = ["TP", "TN", "FP", "FN"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, col in zip(axes.flatten(), plot_cols):
    sns.boxplot(data=df, x="outcome", y=col, order=order,
                palette=palette, ax=ax, width=0.5, flierprops=dict(marker="o", markersize=4))
    ax.set_title(col, fontsize=12)
    ax.set_xlabel("")
plt.suptitle("Clinical Variables by Prediction Outcome (TP/TN/FP/FN)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "error_boxplot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Boxplot saved")

# ─────────────────────────────────────────────
# 5. Visualization (2) — Bending reduction % distribution (gray zone check)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# FN analysis: truly Structural cases that were missed
ax = axes[0]
fn_df = df[df["outcome"].isin(["TP", "FN"])].copy()
bins = [0, 15, 25, 35, 100]
labels_bin = ["<15%", "15–25%", "25–35%", "≥35%"]
fn_df["reduction_bin"] = pd.cut(fn_df["Lumbar_Reduction_pct"], bins=bins, labels=labels_bin, right=False)
ct = fn_df.groupby(["reduction_bin", "outcome"]).size().unstack(fill_value=0)
ct.plot(kind="bar", ax=ax, color=["#F44336", "#4CAF50"], edgecolor="black")
ax.set_title("Structural (label=1): TP vs FN\nby Lumbar Reduction %", fontsize=12)
ax.set_xlabel("Lumbar Reduction %")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Outcome")

# FP analysis: truly Non-structural cases predicted as structural
ax = axes[1]
fp_df = df[df["outcome"].isin(["TN", "FP"])].copy()
fp_df["reduction_bin"] = pd.cut(fp_df["Lumbar_Reduction_pct"], bins=bins, labels=labels_bin, right=False)
ct2 = fp_df.groupby(["reduction_bin", "outcome"]).size().unstack(fill_value=0)
ct2.plot(kind="bar", ax=ax, color=["#FF9800", "#2196F3"], edgecolor="black")
ax.set_title("Non-structural (label=0): TN vs FP\nby Lumbar Reduction %", fontsize=12)
ax.set_xlabel("Lumbar Reduction %")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Outcome")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "error_bending_reduction.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Bending reduction distribution saved")

# ─────────────────────────────────────────────
# 6. Visualization (3) — Prediction probability scatter
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for outcome, color in palette.items():
    sub = df[df["outcome"] == outcome]
    ax.scatter(sub["Lumbar_Cobb"], sub["prob"],
               c=color, label=f"{outcome} (n={len(sub)})",
               alpha=0.7, s=60, edgecolors="white", linewidths=0.5)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="threshold=0.5")
ax.set_xlabel("Lumbar Cobb Angle (°)", fontsize=12)
ax.set_ylabel("Predicted Probability", fontsize=12)
ax.set_title("Predicted Probability vs Lumbar Cobb — by Outcome", fontsize=13)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "error_prob_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Scatter plot saved")

# ─────────────────────────────────────────────
# 7. Summary output
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  ERROR ANALYSIS SUMMARY")
print("="*60)
print(f"\nTotal N={len(df)} | TP={sum(df.outcome=='TP')} TN={sum(df.outcome=='TN')} "
      f"FP={sum(df.outcome=='FP')} FN={sum(df.outcome=='FN')}")
print("\n[FN vs TP — significant variables (p<0.05)]")
sig = summary[(summary["FN_vs_TP_p"] < 0.05)][["feature","TP_mean","FN_mean","FN_vs_TP_p"]]
print(sig.to_string(index=False) if len(sig) else "  None")
print("\n[FP vs TN — significant variables (p<0.05)]")
sig2 = summary[(summary["FP_vs_TN_p"] < 0.05)][["feature","TN_mean","FP_mean","FP_vs_TN_p"]]
print(sig2.to_string(index=False) if len(sig2) else "  None")
print(f"\nResults saved to: {OUT_DIR}")
