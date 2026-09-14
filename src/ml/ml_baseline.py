"""
Structural_L Binary Classification — Clinical Variables Only
- Patients: 155 matched (only patients with both PA and LAT DICOMs)
- Features: all clinical variables except label-related bending variables
- Target: Structural_L (0/1)
- CV: 5-fold stratified
- Models: LR, SVM, RF, GBM, XGBoost, LightGBM
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, f1_score
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Data loading & matching
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "AIS_Surgery_clean.csv")
CACHE_PATH = os.path.join(BASE_DIR, "cache", "scoliosis_lumbar_matched_data_512.pkl")
CANONICAL_PRED_DIR = os.path.join(
    BASE_DIR,
    "multimodal",
    "test_results",
    "test_20260406_160438",
    "Rank3_lumbar_only__film__resnet101",
)

df = pd.read_csv(CSV_PATH)

# Same 155 patients as the DL model (those present in the DICOM cache)
cache = pickle.load(open(CACHE_PATH, "rb"))
matched_ids = set(int(k) for k in cache.keys())
df = df[df["ID"].isin(matched_ids)].reset_index(drop=True)
print(f"Matched patients: {len(df)}")
print(f"Structural_L: 1={int(df['Structural_L'].sum())}, 0={int((df['Structural_L']==0).sum())}")

# ─────────────────────────────────────────────
# 2. Feature definition
# ─────────────────────────────────────────────
# Excluded variables:
#   ID               — identifier
#   Structural_T     — the other target (leakage)
#   Structural_L     — the target itself
#   Lumbar_Cobb_Bending   — bending-radiograph variable used directly to define the label
#   Thoracic_Cobb_Bending — same reason

EXCLUDE = ["ID", "Structural_T", "Structural_L",
           "Lumbar_Cobb_Bending", "Thoracic_Cobb_Bending"]

TARGET = "Structural_L"

feature_cols = [c for c in df.columns if c not in EXCLUDE]
print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols].copy()
y = df[TARGET].astype(int).values

# ─────────────────────────────────────────────
# 3. Preprocessing pipeline
# ─────────────────────────────────────────────
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# ─────────────────────────────────────────────
# 4. Model definitions
# ─────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, random_state=42
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", probability=True, C=1.0, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        random_state=42, verbose=-1
    ),
}

# ─────────────────────────────────────────────
# 5. 5-Fold CV
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auroc    = roc_auc_score(y_true, y_prob)
    acc      = accuracy_score(y_true, y_pred)
    sens     = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec     = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv      = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv      = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1       = f1_score(y_true, y_pred, zero_division=0)
    return dict(auroc=auroc, accuracy=acc, sensitivity=sens,
                specificity=spec, ppv=ppv, npv=npv, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)

# Reuse the exact held-out patient sets stored for the final proposed model.
# Patient IDs are used instead of positional indices because the clinical CSV
# and multimodal cache do not necessarily share the same row ordering.
canonical_test_ids = []
for fold_idx in range(1, 6):
    pred_path = os.path.join(CANONICAL_PRED_DIR, f"fold{fold_idx}_predictions.csv")
    pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
    canonical_test_ids.append(set(pred_df["patient_id"].astype(int)))

all_canonical_ids = set().union(*canonical_test_ids)
current_ids = set(df["ID"].astype(int))
if all_canonical_ids != current_ids:
    raise RuntimeError(
        "Canonical final-model folds and the ML cohort contain different patient IDs: "
        f"missing_from_ml={sorted(all_canonical_ids - current_ids)}, "
        f"extra_in_ml={sorted(current_ids - all_canonical_ids)}"
    )
if sum(len(ids) for ids in canonical_test_ids) != len(all_canonical_ids):
    raise RuntimeError("Canonical final-model test folds overlap.")

canonical_splits = []
id_values = df["ID"].astype(int).to_numpy()
for test_ids in canonical_test_ids:
    test_mask = np.isin(id_values, list(test_ids))
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)
    canonical_splits.append((train_idx, test_idx))

print("Canonical final-model folds:", [len(test_idx) for _, test_idx in canonical_splits])

all_results = {}
oof_predictions = {name: [] for name in MODELS}  # for DeLong/bootstrap significance testing

for model_name, clf in MODELS.items():
    fold_metrics = []
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    for fold_idx, (train_idx, test_idx) in enumerate(canonical_splits, 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        ids_test = df["ID"].iloc[test_idx].to_numpy()

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        oof_predictions[model_name].append(pd.DataFrame({
            "ID": ids_test, "fold": fold_idx, "label": y_test, "prob": y_prob,
        }))

        m = compute_metrics(y_test, y_prob)
        fold_metrics.append(m)
        print(f"  Fold {fold_idx}: AUROC={m['auroc']:.4f}  Acc={m['accuracy']:.4f}  "
              f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}")

    # Mean / standard deviation
    metric_keys = ["auroc", "accuracy", "sensitivity", "specificity", "ppv", "npv", "f1"]
    means = {k: np.nanmean([f[k] for f in fold_metrics]) for k in metric_keys}
    stds  = {k: np.nanstd( [f[k] for f in fold_metrics]) for k in metric_keys}

    print(f"\n  [CV Summary]")
    for k in metric_keys:
        print(f"    {k:<15}: {means[k]:.4f} ± {stds[k]:.4f}")

    all_results[model_name] = {"means": means, "stds": stds, "folds": fold_metrics}

# ─────────────────────────────────────────────
# 6. Final comparison table
# ─────────────────────────────────────────────
print(f"\n\n{'='*80}")
print("  FINAL COMPARISON TABLE (5-Fold CV Mean ± Std)")
print(f"{'='*80}")

header = f"{'Model':<22} {'AUROC':>16} {'Accuracy':>16} {'Sensitivity':>16} {'Specificity':>16} {'F1':>16}"
print(header)
print("-" * len(header))

for model_name, res in all_results.items():
    m, s = res["means"], res["stds"]
    row = (f"{model_name:<22} "
           f"{m['auroc']:.4f}±{s['auroc']:.4f}  "
           f"{m['accuracy']:.4f}±{s['accuracy']:.4f}  "
           f"{m['sensitivity']:.4f}±{s['sensitivity']:.4f}  "
           f"{m['specificity']:.4f}±{s['specificity']:.4f}  "
           f"{m['f1']:.4f}±{s['f1']:.4f}")
    print(row)

# Save CSV
rows = []
for model_name, res in all_results.items():
    m, s = res["means"], res["stds"]
    rows.append({
        "model": model_name,
        **{f"{k}_mean": round(m[k], 4) for k in ["auroc","accuracy","sensitivity","specificity","ppv","npv","f1"]},
        **{f"{k}_std":  round(s[k], 4) for k in ["auroc","accuracy","sensitivity","specificity","ppv","npv","f1"]},
    })

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_structural_L_results.csv")
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\nResults saved → {out_path}")

# ─────────────────────────────────────────────
# 7. OOF predictions (for DeLong / paired bootstrap significance testing)
# ─────────────────────────────────────────────
for model_name, fold_dfs in oof_predictions.items():
    oof_df = pd.concat(fold_dfs, ignore_index=True).sort_values("ID").reset_index(drop=True)
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    oof_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ml_{safe_name}_oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved → {oof_path}  (n={len(oof_df)})")
