"""Subgroup performance analysis for AIS Structural Change Prediction model.

Computes fold-level and aggregate metrics for:
  1. Overall test set
  2. Cobb angle >= 30°
  3. Cobb angle >= 40°

Subgroup criterion uses the STANDING (neutral) Lumbar Cobb angle — NOT bending.
The script supports two prediction-file layouts:
  A) one Excel file with one sheet per fold
  B) one CSV per fold (filename pattern with `{fold}`)
  C) a single CSV with a `fold` column

Run: python subgroup_analysis.py
"""
import os
from typing import Optional

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# Config — edit these for your data layout
# ============================================================

# ---- Prediction file (results to evaluate) ----------------
# Three supported modes — set the unused ones to None.
PRED_PATH_EXCEL = str(ROOT / "results" / "fold_patient_outcomes.xlsx")  # mode A: multi-sheet Excel
PRED_PATH_PATTERN = None  # e.g. "/path/fold{fold}_predictions.csv"    (mode B)
PRED_PATH_SINGLE_CSV = None  # e.g. "/path/all_folds_predictions.csv"  (mode C)
FOLD_RANGE = range(1, 6)  # folds 1..5

# ---- Clinical / radiographic file (Cobb angle source) -----
# If None, the prediction file must already contain COBB_COL.
CLINICAL_PATH = str(ROOT / "data" / "AIS_Surgery_clean.csv")
MERGE_ON = "ID"  # column to join predictions ↔ clinical (both sides must have it)

# ---- Column names -----------------------------------------
ID_COL    = "ID"            # patient identifier in prediction file
FOLD_COL  = "fold"          # only used when reading a single CSV (mode C)
LABEL_COL = "label"         # 0/1 ground truth
PROB_COL  = "prob"          # predicted probability for class 1
PRED_COL  = "pred"          # binary prediction (optional — generated from prob if missing)
# STANDING Cobb angle columns (NOT bending) — both are analyzed.
# `Lumbar_Cobb`  : standing Cobb of the lumbar curve only
# `Cobb`         : major Cobb angle (largest of all curves, typically thoracic)
COBB_COLS = ["Lumbar_Cobb", "Cobb"]
THRESHOLD = 0.5
# Match the main performance table: population SD across the fixed CV folds.
SD_DDOF = 0

# ---- Subgroups --------------------------------------------
# (name, filter_fn) — filter_fn returns a boolean Series on the dataframe
# Built dynamically: overall + each Cobb column × {>=30, >=40}
def _mk_ge(col, thr):
    return lambda df: df[col] >= thr

SUBGROUPS = [("overall", lambda df: pd.Series(True, index=df.index))]
for _col in COBB_COLS:
    for _thr in (30, 40):
        SUBGROUPS.append((f"{_col}_ge_{_thr}", _mk_ge(_col, _thr)))

# ---- Output -----------------------------------------------
OUT_XLSX = str(ROOT / "results" / "subgroup_analysis.xlsx")


# ============================================================
# Loading
# ============================================================
def load_predictions() -> pd.DataFrame:
    """Load predictions from one of three layouts and return a single DataFrame
    with a `fold` column added (if not already present)."""
    if PRED_PATH_EXCEL is not None:
        # Mode A — multi-sheet Excel, one sheet per fold (Fold1, Fold2, ...)
        xl = pd.ExcelFile(PRED_PATH_EXCEL)
        frames = []
        for fold in FOLD_RANGE:
            # Match the sheet name flexibly (Fold1, fold1, 1, etc.)
            sheet = next((s for s in xl.sheet_names if str(fold) in s), None)
            if sheet is None:
                continue
            df = pd.read_excel(xl, sheet_name=sheet)
            df[FOLD_COL] = fold
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    if PRED_PATH_PATTERN is not None:
        # Mode B — one CSV per fold
        frames = []
        for fold in FOLD_RANGE:
            path = PRED_PATH_PATTERN.format(fold=fold)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df[FOLD_COL] = fold
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    if PRED_PATH_SINGLE_CSV is not None:
        # Mode C — single CSV containing the fold column
        return pd.read_csv(PRED_PATH_SINGLE_CSV)

    raise ValueError("Set one of PRED_PATH_EXCEL / PRED_PATH_PATTERN / PRED_PATH_SINGLE_CSV")


def attach_cobb(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Merge clinical Cobb-angle columns into prediction df if not present."""
    need = [c for c in COBB_COLS if c not in pred_df.columns]
    if not need:
        return pred_df

    if CLINICAL_PATH is None:
        raise ValueError(f"Columns {need} missing and CLINICAL_PATH not set")

    clinical = pd.read_csv(CLINICAL_PATH)
    for side, df in (("pred", pred_df), ("clinical", clinical)):
        if MERGE_ON not in df.columns:
            raise ValueError(f"{side} file has no '{MERGE_ON}' column for merge")

    # Cast both sides to the same dtype to avoid silent merge misses
    pred_df  = pred_df.copy()
    clinical = clinical.copy()
    pred_df[MERGE_ON]  = pred_df[MERGE_ON].astype(str).str.lstrip("0")
    clinical[MERGE_ON] = clinical[MERGE_ON].astype(str).str.lstrip("0")

    merged = pred_df.merge(
        clinical[[MERGE_ON] + need],
        on=MERGE_ON, how="left"
    )
    for c in need:
        missing = merged[c].isna().sum()
        if missing:
            print(f"[WARN] {missing} rows have no '{c}' after merge")
    return merged


# ============================================================
# Metric helpers
# ============================================================
def _safe_div(num, den):
    """Division returning NaN when denominator is 0 (no false-zero metrics)."""
    return float(num) / den if den > 0 else float("nan")


def compute_metrics(df: pd.DataFrame, threshold: float = THRESHOLD) -> dict:
    """Compute Acc / Sens / Spec / AUROC on a single subgroup slice."""
    y_true = df[LABEL_COL].astype(int).to_numpy()

    # Use stored prediction if present, otherwise threshold the probability
    if PRED_COL in df.columns:
        y_pred = df[PRED_COL].astype(int).to_numpy()
    else:
        y_pred = (df[PROB_COL].to_numpy() >= threshold).astype(int)

    y_prob = df[PROB_COL].to_numpy() if PROB_COL in df.columns else None

    n        = int(len(df))
    pos_n    = int((y_true == 1).sum())
    neg_n    = int((y_true == 0).sum())

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy    = _safe_div(tp + tn, tp + tn + fp + fn)
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)

    # AUROC only defined when both classes present
    if y_prob is not None and pos_n > 0 and neg_n > 0:
        auroc = float(roc_auc_score(y_true, y_prob))
    else:
        auroc = float("nan")

    return dict(
        N=n, positive_N=pos_n, negative_N=neg_n,
        accuracy=accuracy, sensitivity=sensitivity,
        specificity=specificity, auroc=auroc,
    )


# ============================================================
# Aggregation
# ============================================================
def fold_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for every (fold, subgroup) combination."""
    rows = []
    for fold, fold_df in df.groupby(FOLD_COL):
        for name, mask_fn in SUBGROUPS:
            mask = mask_fn(fold_df).fillna(False)
            sub  = fold_df[mask]
            m    = compute_metrics(sub)
            rows.append({
                "fold": int(fold),
                "subgroup": name,
                "threshold": THRESHOLD,
                **m,
            })
    return pd.DataFrame(rows)


def _fmt_mean_std(mean, std, digits=4):
    """Format 'mean ± std' string; NaN-safe."""
    if pd.isna(mean):
        return "NaN"
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summary_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level metrics → mean ± std per subgroup."""
    metric_cols = ["N", "positive_N", "negative_N",
                   "accuracy", "sensitivity", "specificity", "auroc"]

    grouped = fold_df.groupby("subgroup")[metric_cols]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std(ddof=SD_DDOF).add_suffix("_std")
    agg = means.join(stds).reset_index()
    agg.insert(1, "threshold", THRESHOLD)

    # Combine performance metrics into single "mean ± std" strings
    for m in ("accuracy", "sensitivity", "specificity", "auroc"):
        agg[m] = [_fmt_mean_std(mu, sd) for mu, sd in zip(agg[f"{m}_mean"], agg[f"{m}_std"])]

    out_cols = [
        "subgroup", "threshold",
        "N_mean", "positive_N_mean", "negative_N_mean",
        "accuracy", "sensitivity", "specificity", "auroc",
    ]
    agg = agg[out_cols]

    # Preserve original subgroup order
    order = {name: i for i, (name, _) in enumerate(SUBGROUPS)}
    agg = agg.sort_values("subgroup", key=lambda s: s.map(order)).reset_index(drop=True)
    return agg


# ============================================================
# Main
# ============================================================
def main():
    print(f"[INFO] Loading predictions...")
    pred = load_predictions()
    print(f"  rows={len(pred)}  cols={list(pred.columns)}")

    print(f"[INFO] Attaching Cobb angle(s) from '{CLINICAL_PATH}'")
    pred = attach_cobb(pred)
    for c in COBB_COLS:
        print(f"  {c} range: {pred[c].min():.1f}–{pred[c].max():.1f}")

    print(f"[INFO] Computing fold-level metrics...")
    fold_df = fold_level_table(pred)

    print(f"[INFO] Aggregating to summary...")
    summ = summary_table(fold_df)

    # ---- Console output -----------------------------------
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\n========== FOLD-LEVEL METRICS ==========")
    print(fold_df.to_string(index=False))

    print("\n========== SUMMARY (mean ± std over folds) ==========")
    print(summ.to_string(index=False))

    # ---- Save to Excel ------------------------------------
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        fold_df.to_excel(writer, sheet_name="fold_level_metrics", index=False)
        summ.to_excel(writer, sheet_name="summary_metrics", index=False)
    print(f"\n[INFO] Saved → {OUT_XLSX}")


if __name__ == "__main__":
    main()
