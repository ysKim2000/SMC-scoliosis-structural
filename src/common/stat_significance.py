"""
Statistical significance testing for AUROC comparisons (reviewer Concern #2).

All comparisons use out-of-fold (OOF) predictions pooled across the 5 CV folds,
matched per patient by ID. This works even when two models used different
CV fold partitions (verified: main DL model and ablation/ML baselines use
different random folds but cover the same 155 patients exactly once each).

Two independent tests are reported per comparison:
  1. DeLong test (Sun & Xu, 2014 fast implementation) — analytic test for the
     difference between two correlated AUROCs on the same subjects.
  2. Paired bootstrap (10,000 resamples over patients) — resampling-based
     cross-check, robust to DeLong's asymptotic-normality assumption at N=155.

Run: python stat_significance.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd

RNG_SEED = 42
N_BOOT = 10000


# ============================================================
# Fast DeLong (Sun & Xu, 2014)
# ============================================================
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive_examples = preds_sorted_transposed[:, :m]
    negative_examples = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b):
    """Two-sided p-value for AUROC(prob_a) == AUROC(prob_b) on the same subjects."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind="stable")
    y_sorted = y_true[order]
    m = int(y_sorted.sum())

    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    aucs, delongcov = _fast_delong(preds, m)

    diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], diff, float("nan"), float("nan")
    z = diff / np.sqrt(var)
    from scipy import stats
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], diff, z, p


# ============================================================
# Paired bootstrap (resample patients, not folds)
# ============================================================
def paired_bootstrap_auc_test(y_true, prob_a, prob_b, n_boot=N_BOOT, seed=RNG_SEED):
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    n = len(y_true)
    rng = np.random.RandomState(seed)

    obs_diff = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    diffs = np.empty(n_boot)
    i = 0
    while i < n_boot:
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue  # need both classes present
        diffs[i] = roc_auc_score(yt, prob_a[idx]) - roc_auc_score(yt, prob_b[idx])
        i += 1

    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value: proportion of resamples crossing 0, relative to observed sign
    p_boot = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_boot = min(p_boot, 1.0)
    return obs_diff, ci_lo, ci_hi, p_boot


# ============================================================
# Data loading
# ============================================================
def load_main_oof():
    sheets = pd.read_excel(str(ROOT / "results" / "fold_patient_outcomes.xlsx"), sheet_name=None)
    d = pd.concat(sheets.values(), ignore_index=True)
    d["ID"] = d["ID"].astype(int)
    return d[["ID", "label", "prob"]].rename(columns={"prob": "prob_main"})


def load_clinical_only_oof():
    base = (str(ROOT / "experiments_lumbar_ablation" / "scoliosis_structural") + "/"
             "run_t1/cv5_20260406_173119_512/clinical_only")
    dfs = [pd.read_csv(f"{base}/fold{i}/logs/best_auroc_test_predictions.csv") for i in range(1, 6)]
    d = pd.concat(dfs, ignore_index=True)
    d["patient_id"] = d["patient_id"].astype(int)
    return d[["patient_id", "label", "prob"]].rename(
        columns={"patient_id": "ID", "prob": "prob_clinonly"})


def load_lr_oof():
    d = pd.read_csv(str(ROOT / "results" / "ml_logistic_regression_oof_predictions.csv"))
    d["ID"] = d["ID"].astype(int)
    return d[["ID", "label", "prob"]].rename(columns={"prob": "prob_lr"})


def run_comparison(name, y_true, prob_a, prob_b, label_a, label_b):
    print(f"\n{'='*70}\n  {name}: {label_a} vs {label_b}\n{'='*70}")
    auc_a, auc_b, diff, z, p_delong = delong_roc_test(y_true, prob_a, prob_b)
    print(f"  AUROC {label_a} = {auc_a:.4f}")
    print(f"  AUROC {label_b} = {auc_b:.4f}")
    print(f"  Diff            = {diff:+.4f}")
    print(f"  DeLong z        = {z:.4f}   p = {p_delong:.4f}")

    obs_diff, ci_lo, ci_hi, p_boot = paired_bootstrap_auc_test(y_true, prob_a, prob_b)
    print(f"  Bootstrap diff  = {obs_diff:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]   p = {p_boot:.4f}")
    return dict(comparison=name, auc_a=auc_a, auc_b=auc_b, diff=diff,
                delong_z=z, delong_p=p_delong,
                boot_diff=obs_diff, boot_ci_lo=ci_lo, boot_ci_hi=ci_hi, boot_p=p_boot)


def main():
    main_oof = load_main_oof()
    clin_oof = load_clinical_only_oof()

    m1 = main_oof.merge(clin_oof, on=["ID", "label"], how="inner")
    assert len(m1) == 155, f"expected 155 matched patients, got {len(m1)}"
    r1 = run_comparison("Proposed vs Clinical-only (DL)", m1["label"],
                         m1["prob_main"], m1["prob_clinonly"],
                         "PA+LAT+Clinical", "Clinical-only")

    results = [r1]
    try:
        lr_oof = load_lr_oof()
        m2 = main_oof.merge(lr_oof, on=["ID", "label"], how="inner")
        assert len(m2) == 155, f"expected 155 matched patients, got {len(m2)}"
        r2 = run_comparison("Proposed vs Logistic Regression (ML)", m2["label"],
                             m2["prob_main"], m2["prob_lr"],
                             "PA+LAT+Clinical", "Logistic Regression")
        results.append(r2)
    except FileNotFoundError:
        print("\n[WARN] ML/ml_lr_oof_predictions.csv not found — run ML/ml_structural_L.py first "
              "(now modified to dump OOF predictions).")

    out = pd.DataFrame(results)
    out.to_csv(str(ROOT / "results" / "stat_significance_results.csv"), index=False)
    print(f"\n[INFO] Saved -> stat_significance_results.csv")


if __name__ == "__main__":
    main()
