"""
evaluate.py
Loads top-k model checkpoints and runs 5-fold CV testing.
- Reports Accuracy, Sensitivity, Specificity, and AUROC per fold
- Computes 5-fold mean ± standard deviation
- Saves per-patient predictions as CSV
- Plots confusion matrix and ROC curve

Usage:
    python evaluate.py
    python evaluate.py --output_dir test_results
"""

import pathlib
import os
import json
import argparse
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import ScoliosisLumbarDataset, custom_collate_fn
from model import ScoliosisLumbarModel


# =========================================================
# Top 3 Model Configurations
# =========================================================
PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])

TOP_MODELS = {
    "Rank1_lumbar_only__film__resnet101": {
        "description": "lumbar_only + film + resnet101 (AUROC 0.9022)",
        "clinical_set": "lumbar_only",
        "fusion_type": "film",
        "backbone": "resnet101.tv_in1k",
        "checkpoint_pattern": f"{PROJECT_ROOT}/experiments_lumbar/scoliosis_structural/cv5_20260330_213940_512/lumbar_only__film__resnet101_tv_in1k/fold{{fold}}/checkpoints/best_auroc.pth",
    },
    "Rank2_global_lumbar__concat__convnext_tiny": {
        "description": "global_lumbar + concat + convnext_tiny (AUROC 0.9222)",
        "clinical_set": "global_lumbar",
        "fusion_type": "concat",
        "backbone": "convnext_tiny.fb_in1k",
        "checkpoint_pattern": f"{PROJECT_ROOT}/experiments_lumbar/scoliosis_structural/cv5_20260330_213950_512/global_lumbar__concat__convnext_tiny_fb_in1k/fold{{fold}}/checkpoints/best_auroc.pth",
    },
    "Rank3_global_lumbar__gated__resnet101": {
        "description": "global_lumbar + gated + resnet101 (AUROC 0.9222)",
        "clinical_set": "global_lumbar",
        "fusion_type": "gated",
        "backbone": "resnet101.tv_in1k",
        "checkpoint_pattern": f"{PROJECT_ROOT}/experiments_lumbar/scoliosis_structural/cv5_20260330_213950_512/global_lumbar__gated__resnet101_tv_in1k/fold{{fold}}/checkpoints/best_auroc.pth",
    },
}

DEFAULT_MODEL_KEYS = ["Rank1_lumbar_only__film__resnet101"]
CLASS_NAMES = ["Non-Structural", "Structural"]
OUTCOME_LABELS = {
    "TP": "True Positive",
    "TN": "True Negative",
    "FP": "False Positive",
    "FN": "False Negative",
}


# =========================================================
# Model Loading
# =========================================================
def load_model_from_checkpoint(checkpoint_path, device):
    """Read config from the checkpoint, build the model, and load its weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = ScoliosisLumbarModel(
        pa_backbone=config["pa_backbone"],
        lat_backbone=config["lat_backbone"],
        clinical_dim=config["clinical_dim"],
        hidden_dim=config["hidden_dim"],
        dropout_p=config.get("dropout_p", 0.4),
        fusion_type=config["fusion_type"],
        num_heads=config.get("num_heads", 8),
        num_transformer_layers=config.get("num_transformer_layers", 2),
        view_dropout_p=config.get("view_dropout_p", 0.0),
        use_aux_heads=config.get("use_aux_heads", False),
        aux_hidden_dim=config.get("aux_hidden_dim", 256),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config, ckpt


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate the model — returns logits, probs, labels, patient_ids."""
    model.eval()
    all_labels, all_probs, all_logits, all_pids = [], [], [], []

    for batch_data, labels in dataloader:
        pa = batch_data["pa"].to(device, non_blocking=True)
        lat = batch_data["lat"].to(device, non_blocking=True)
        clin = batch_data["clinical"].to(device, non_blocking=True)

        logits = model(pa, lat, clin)
        probs = torch.sigmoid(logits)

        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_pids.extend(batch_data["patient_id"])

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_logits = np.concatenate(all_logits)

    return all_labels, all_probs, all_logits, all_pids


def find_last_conv_layer(module):
    """Find the last Conv2d layer for Grad-CAM."""
    last_conv = None
    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d):
            last_conv = child
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last_conv


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")

    return {
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "AUROC": auroc,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def compute_threshold_sweep(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    rows = []
    for threshold in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold)
        rows.append({"threshold": threshold, **metrics})
    return pd.DataFrame(rows)


def compute_brier_score(y_true, y_prob):
    try:
        return brier_score_loss(y_true.astype(int), y_prob.astype(float))
    except Exception:
        return float("nan")


def compute_calibration_errors(y_true, y_prob, n_bins=5, strategy="quantile"):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0:
        return {"ECE": float("nan"), "MCE": float("nan")}

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(y_prob, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return {"ECE": 0.0, "MCE": 0.0}
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    total = len(y_true)

    for idx in range(len(bin_edges) - 1):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == len(bin_edges) - 2:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        if not np.any(mask):
            continue

        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        gap = abs(bin_acc - bin_conf)
        ece += gap * (mask.sum() / total)
        mce = max(mce, gap)

    return {"ECE": float(ece), "MCE": float(mce)}


def denormalize_image(tensor, mean, std):
    mean_t = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    image = tensor * std_t + mean_t
    image = torch.clamp(image, 0, 1)
    return image.detach().cpu().permute(1, 2, 0).numpy()


def overlay_cam_on_image(image_rgb, cam_map, alpha=0.35):
    image_uint8 = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(np.clip(cam_map * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_uint8, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


def compute_single_view_gradcam(model, image_tensor, other_tensor, clinical_tensor, view="pa"):
    model.eval()
    if view == "pa":
        encoder = model.pa_model
    elif view == "lat":
        encoder = model.lat_model
    else:
        raise ValueError(f"Unknown view: {view}")

    target_layer = find_last_conv_layer(encoder.feature_extractor)
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        if view == "pa":
            logits = model(image_tensor, other_tensor, clinical_tensor)
        else:
            logits = model(other_tensor, image_tensor, clinical_tensor)

        score = logits.squeeze()
        score.backward(retain_graph=False)

        if not activations or not gradients:
            raise RuntimeError("Failed to capture Grad-CAM activations or gradients.")

        act = activations[-1].detach()
        grad = gradients[-1].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), float(torch.sigmoid(score).item())
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


# =========================================================
# Visualization
# =========================================================
def plot_roc_curves(fold_results, model_name, save_dir):
    """Plot all 5-fold ROC curves on a single figure."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for fold_idx, result in enumerate(fold_results, 1):
        fpr, tpr, _ = roc_curve(result["labels"], result["probs"])
        auroc = result["metrics"]["AUROC"]
        ax.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold_idx} (AUROC={auroc:.4f})')
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auroc = np.mean([r["metrics"]["AUROC"] for r in fold_results])
    std_auroc = np.std([r["metrics"]["AUROC"] for r in fold_results])

    ax.plot(mean_fpr, mean_tpr, color='b', linewidth=2,
            label=f'Mean (AUROC={mean_auroc:.4f}±{std_auroc:.4f})')

    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1),
                     np.clip(mean_tpr + std_tpr, 0, 1), color='b', alpha=0.1)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {model_name}', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"roc_curve_{model_name}.png"), dpi=150)
    plt.close()


def plot_precision_recall_curves(fold_results, model_name, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    mean_recall = np.linspace(0, 1, 200)
    precisions = []
    ap_scores = []

    for fold_idx, result in enumerate(fold_results, 1):
        precision, recall, _ = precision_recall_curve(result["labels"], result["probs"])
        ap = average_precision_score(result["labels"], result["probs"])
        ax.plot(recall, precision, alpha=0.35, label=f"Fold {fold_idx} (AP={ap:.4f})")
        order = np.argsort(recall)
        precisions.append(np.interp(mean_recall, recall[order], precision[order]))
        ap_scores.append(ap)

    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    ax.plot(
        mean_recall,
        mean_precision,
        color="darkorange",
        linewidth=2,
        label=f"Mean (AP={np.mean(ap_scores):.4f}±{np.std(ap_scores):.4f})",
    )
    ax.fill_between(
        mean_recall,
        np.clip(mean_precision - std_precision, 0, 1),
        np.clip(mean_precision + std_precision, 0, 1),
        color="darkorange",
        alpha=0.15,
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pr_curve_{model_name}.png"), dpi=150)
    plt.close()


def plot_probability_distribution(y_true, y_prob, model_name, fold_label, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    neg_probs = y_prob[y_true == 0]
    pos_probs = y_prob[y_true == 1]

    bins = np.linspace(0, 1, 21)
    axes[0].hist(neg_probs, bins=bins, alpha=0.7, color="#4c78a8", label="Non-Structural")
    axes[0].hist(pos_probs, bins=bins, alpha=0.7, color="#e45756", label="Structural")
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Probability Histogram")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    plot_df = pd.DataFrame({
        "Probability": y_prob,
        "Class": np.where(y_true.astype(int) == 1, "Structural", "Non-Structural"),
    })
    sns.violinplot(data=plot_df, x="Class", y="Probability", palette=["#4c78a8", "#e45756"], ax=axes[1])
    sns.stripplot(data=plot_df, x="Class", y="Probability", color="black", alpha=0.45, size=3, ax=axes[1])
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Probability by True Class")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Probability Distribution — {model_name} ({fold_label})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"prob_distribution_{model_name}_{fold_label}.png"), dpi=150)
    plt.close()


def plot_threshold_sweep(y_true, y_prob, threshold_df, model_name, fold_label, save_dir, default_threshold=0.5):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    metrics_to_plot = ["Accuracy", "Sensitivity", "Specificity", "F1", "PPV", "NPV"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]

    for metric, color in zip(metrics_to_plot, colors):
        ax.plot(threshold_df["threshold"], threshold_df[metric], label=metric, color=color, linewidth=2)

    ax.axvline(default_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Default={default_threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Sweep — {model_name} ({fold_label})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"threshold_sweep_{model_name}_{fold_label}.png"), dpi=150)
    plt.close()


def plot_calibration_curves(fold_results, model_name, save_dir, n_bins=5, strategy="quantile"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    brier_scores = []

    for fold_idx, result in enumerate(fold_results, 1):
        frac_pos, mean_pred = calibration_curve(
            result["labels"].astype(int),
            result["probs"],
            n_bins=n_bins,
            strategy=strategy,
        )
        brier = compute_brier_score(result["labels"], result["probs"])
        cal_errors = compute_calibration_errors(
            result["labels"], result["probs"], n_bins=n_bins, strategy=strategy
        )
        brier_scores.append(brier)
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            linewidth=1.3,
            alpha=0.45,
            label=f"Fold {fold_idx} (Brier={brier:.4f}, ECE={cal_errors['ECE']:.4f})",
        )

    pooled_labels = np.concatenate([result["labels"].astype(int) for result in fold_results], axis=0)
    pooled_probs = np.concatenate([result["probs"] for result in fold_results], axis=0)
    pooled_frac_pos, pooled_mean_pred = calibration_curve(
        pooled_labels,
        pooled_probs,
        n_bins=n_bins,
        strategy=strategy,
    )
    pooled_brier = compute_brier_score(pooled_labels, pooled_probs)
    pooled_errors = compute_calibration_errors(
        pooled_labels, pooled_probs, n_bins=n_bins, strategy=strategy
    )
    ax.plot(
        pooled_mean_pred,
        pooled_frac_pos,
        marker="o",
        markersize=8,
        linewidth=2.8,
        color="black",
        label=(
            f"Pooled (Brier={pooled_brier:.4f}, "
            f"ECE={pooled_errors['ECE']:.4f}, MCE={pooled_errors['MCE']:.4f})"
        ),
    )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.6, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve — {model_name}\n({strategy}, {n_bins} bins)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"calibration_curve_{model_name}.png"), dpi=150)
    plt.close()

    calibration_summary = []
    for result in fold_results:
        errors = compute_calibration_errors(
            result["labels"], result["probs"], n_bins=n_bins, strategy=strategy
        )
        calibration_summary.append({
            "fold": result["fold"],
            "Brier": compute_brier_score(result["labels"], result["probs"]),
            "ECE": errors["ECE"],
            "MCE": errors["MCE"],
        })

    calibration_summary.append({
        "fold": "pooled",
        "Brier": pooled_brier,
        "ECE": pooled_errors["ECE"],
        "MCE": pooled_errors["MCE"],
    })
    pd.DataFrame(calibration_summary).to_csv(
        os.path.join(save_dir, f"calibration_summary_{model_name}.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def collect_gradcam_cases(pred_df, top_k_per_type=1):
    outcome_masks = {
        "TP": (pred_df["label"] == 1) & (pred_df["pred"] == 1),
        "TN": (pred_df["label"] == 0) & (pred_df["pred"] == 0),
        "FP": (pred_df["label"] == 0) & (pred_df["pred"] == 1),
        "FN": (pred_df["label"] == 1) & (pred_df["pred"] == 0),
    }

    case_rows = []
    for outcome, mask in outcome_masks.items():
        subset = pred_df.loc[mask].copy()
        if subset.empty:
            continue
        subset["confidence"] = np.where(
            outcome in {"TP", "FP"},
            subset["prob"],
            1.0 - subset["prob"],
        )
        subset = subset.sort_values("confidence", ascending=False)
        if top_k_per_type > 0:
            subset = subset.head(top_k_per_type)
        subset["outcome"] = outcome
        case_rows.append(subset)

    if not case_rows:
        return pd.DataFrame(columns=["patient_id", "label", "prob", "pred", "logit", "confidence", "outcome"])
    return pd.concat(case_rows, ignore_index=True)


def _generate_single_gradcam(model, dataset, dataset_index_by_pid, row, mean, std, device, save_path):
    """Save the GradCAM image for a single patient."""
    sample_idx = dataset_index_by_pid[row["patient_id"]]
    batch_data, _ = dataset[sample_idx]

    pa = batch_data["pa"].unsqueeze(0).to(device)
    lat = batch_data["lat"].unsqueeze(0).to(device)
    clin = batch_data["clinical"].unsqueeze(0).to(device)

    pa_cam, score = compute_single_view_gradcam(model, pa, lat, clin, view="pa")
    lat_cam, _ = compute_single_view_gradcam(model, pa, lat, clin, view="lat")

    pa_img = denormalize_image(batch_data["pa"], mean, std)
    lat_img = denormalize_image(batch_data["lat"], mean, std)

    pa_overlay = overlay_cam_on_image(pa_img, pa_cam)
    lat_overlay = overlay_cam_on_image(lat_img, lat_cam)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(pa_img)
    axes[0].set_title(f"PA Original\nPID {row['patient_id']}")
    axes[1].imshow(pa_overlay)
    axes[1].set_title("PA Grad-CAM")
    axes[2].imshow(denormalize_image(batch_data["lat"], mean, std))
    axes[2].set_title("LAT Original")
    axes[3].imshow(lat_overlay)
    axes[3].set_title(
        f"LAT Grad-CAM\n{row['outcome']} | y={int(row['label'])} pred={int(row['pred'])} p={score:.3f}"
    )
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"PID {row['patient_id']} — {row['outcome']} (prob={row['prob']:.3f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def generate_gradcam_gallery(model, dataset, patient_ids, pred_df, save_path, mean, std, device):
    case_df = pred_df[pred_df["patient_id"].isin(patient_ids)].copy()
    if case_df.empty:
        return

    dataset_index_by_pid = {dataset.label_csv.iloc[i]["patient_id"]: i for i in range(len(dataset))}
    case_df["outcome_order"] = case_df["outcome"].map({"TP": 0, "TN": 1, "FP": 2, "FN": 3})
    case_df = case_df.sort_values(["outcome_order", "confidence"], ascending=[True, False]).reset_index(drop=True)

    # Save as individual images when there are many patients
    if len(case_df) > 8:
        indiv_dir = save_path.replace(".png", "_individual")
        os.makedirs(indiv_dir, exist_ok=True)
        for row_idx, row in case_df.iterrows():
            fname = f"{row['outcome']}_{row['patient_id']}_p{row['prob']:.3f}.png"
            _generate_single_gradcam(
                model, dataset, dataset_index_by_pid, row, mean, std, device,
                os.path.join(indiv_dir, fname),
            )
            print(f"    GradCAM [{row_idx+1}/{len(case_df)}] {row['outcome']} PID {row['patient_id']}")
        print(f"  [GradCAM] {len(case_df)} individual images saved to {indiv_dir}")
        return

    # Use the gallery layout for 8 patients or fewer
    fig, axes = plt.subplots(len(case_df), 3, figsize=(12, 4 * len(case_df)))
    if len(case_df) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in case_df.iterrows():
        sample_idx = dataset_index_by_pid[row["patient_id"]]
        batch_data, _ = dataset[sample_idx]

        pa = batch_data["pa"].unsqueeze(0).to(device)
        lat = batch_data["lat"].unsqueeze(0).to(device)
        clin = batch_data["clinical"].unsqueeze(0).to(device)

        pa_cam, score = compute_single_view_gradcam(model, pa, lat, clin, view="pa")
        lat_cam, _ = compute_single_view_gradcam(model, pa, lat, clin, view="lat")

        pa_img = denormalize_image(batch_data["pa"], mean, std)
        lat_img = denormalize_image(batch_data["lat"], mean, std)

        pa_overlay = overlay_cam_on_image(pa_img, pa_cam)
        lat_overlay = overlay_cam_on_image(lat_img, lat_cam)

        axes[row_idx, 0].imshow(pa_img)
        axes[row_idx, 0].set_title(f"PA\nPID {row['patient_id']}")
        axes[row_idx, 1].imshow(pa_overlay)
        axes[row_idx, 1].set_title("PA Grad-CAM")
        axes[row_idx, 2].imshow(lat_overlay)
        axes[row_idx, 2].set_title(
            f"LAT Grad-CAM\n{row['outcome']} | y={int(row['label'])} pred={int(row['pred'])} p={score:.3f}"
        )

        for col_idx in range(3):
            axes[row_idx, col_idx].axis("off")

    fig.suptitle("Grad-CAM Case Gallery", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name, fold_label, save_dir):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Structural', 'Structural'],
                yticklabels=['Non-Structural', 'Structural'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name} ({fold_label})', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cm_{model_name}_{fold_label}.png"), dpi=150)
    plt.close()


def plot_summary_comparison(all_model_results, save_dir):
    """Bar chart comparing the top 3 models."""
    metrics_to_plot = ["AUROC", "Accuracy", "Sensitivity", "Specificity", "F1"]
    model_names = list(all_model_results.keys())

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    for i, name in enumerate(model_names):
        means = [all_model_results[name]["mean"][m] for m in metrics_to_plot]
        stds = [all_model_results[name]["std"][m] for m in metrics_to_plot]
        short_name = name.split("_", 1)[1] if "_" in name else name
        ax.bar(x + i * width, means, width, yerr=stds, label=short_name, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Top 3 Models — 5-Fold CV Comparison', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=150)
    plt.close()


# =========================================================
# Main Test Runner
# =========================================================
def run_test(
    output_dir="test_results",
    data_dir="data/matched_data",
    label_csv="data/AIS_Surgery_clean.csv",
    batch_size=8,
    model_keys=None,
    gradcam_cases_per_type=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    if model_keys is None:
        model_keys = list(TOP_MODELS.keys())

    selected_models = {key: TOP_MODELS[key] for key in model_keys}

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Testing {len(selected_models)} models × 5 folds\n")

    all_model_results = {}

    for model_key, model_info in selected_models.items():
        print("=" * 100)
        print(f"[MODEL] {model_key}")
        print(f"        {model_info['description']}")
        print("=" * 100)

        model_dir = os.path.join(output_dir, model_key)
        os.makedirs(model_dir, exist_ok=True)

        fold_results = []

        for fold in range(1, 6):
            ckpt_path = model_info["checkpoint_pattern"].format(fold=fold)

            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] Fold {fold}: checkpoint not found at {ckpt_path}")
                continue

            print(f"\n  --- Fold {fold} ---")
            print(f"  Checkpoint: {ckpt_path}")

            # Load model
            model, config, ckpt = load_model_from_checkpoint(ckpt_path, device)

            # Build dataset
            dataset = ScoliosisLumbarDataset(
                data_dir=data_dir,
                label_csv=label_csv,
                img_size=config.get("img_size", 512),
                mode="test",
                clinical_set=config["clinical_set"],
                use_cache=True,
                cache_dir=f"{PROJECT_ROOT}/cache",
            )

            # Load test indices
            test_indices = np.array(config["test_indices"])
            test_dataset = Subset(dataset, test_indices)
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, collate_fn=custom_collate_fn, pin_memory=True
            )

            # Evaluate
            labels, probs, logits, pids = evaluate_model(model, test_loader, device)
            threshold = config.get("threshold", 0.5)
            metrics = compute_metrics(labels, probs, threshold)

            print(f"  AUROC: {metrics['AUROC']:.4f} | "
                  f"Acc: {metrics['Accuracy']:.4f} | "
                  f"Sens: {metrics['Sensitivity']:.4f} | "
                  f"Spec: {metrics['Specificity']:.4f} | "
                  f"F1: {metrics['F1']:.4f} | "
                  f"PPV: {metrics['PPV']:.4f} | "
                  f"NPV: {metrics['NPV']:.4f}")
            print(f"  TP={metrics['TP']} TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']}")

            # Save per-patient predictions
            pred_df = pd.DataFrame({
                "patient_id": pids,
                "label": labels.astype(int),
                "prob": probs,
                "pred": (probs >= threshold).astype(int),
                "logit": logits,
            })
            pred_df.to_csv(os.path.join(model_dir, f"fold{fold}_predictions.csv"),
                           index=False, encoding="utf-8-sig")

            # Confusion matrix
            plot_confusion_matrix(
                labels.astype(int), (probs >= threshold).astype(int),
                model_key, f"fold{fold}", model_dir
            )

            plot_probability_distribution(
                labels.astype(int), probs,
                model_key, f"fold{fold}", model_dir
            )

            threshold_df = compute_threshold_sweep(labels, probs)
            threshold_df.to_csv(
                os.path.join(model_dir, f"fold{fold}_threshold_sweep.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            plot_threshold_sweep(
                labels.astype(int), probs, threshold_df,
                model_key, f"fold{fold}", model_dir, threshold
            )

            gradcam_case_df = collect_gradcam_cases(pred_df, top_k_per_type=gradcam_cases_per_type)
            gradcam_csv_path = os.path.join(model_dir, f"fold{fold}_gradcam_cases.csv")
            gradcam_case_df.to_csv(gradcam_csv_path, index=False, encoding="utf-8-sig")
            if not gradcam_case_df.empty:
                generate_gradcam_gallery(
                    model=model,
                    dataset=dataset,
                    patient_ids=gradcam_case_df["patient_id"].tolist(),
                    pred_df=gradcam_case_df,
                    save_path=os.path.join(model_dir, f"fold{fold}_gradcam_gallery.png"),
                    mean=dataset.mean,
                    std=dataset.std,
                    device=device,
                )

            fold_results.append({
                "fold": fold,
                "metrics": metrics,
                "labels": labels,
                "probs": probs,
                "pids": pids,
                "threshold": threshold,
                "brier_score": compute_brier_score(labels, probs),
            })

            del model
            torch.cuda.empty_cache()

        if not fold_results:
            print(f"\n  [WARNING] No folds completed for {model_key}")
            continue

        # 5-fold summary
        metric_keys = ["AUROC", "Accuracy", "Sensitivity", "Specificity", "F1", "PPV", "NPV"]
        fold_metrics = {k: [r["metrics"][k] for r in fold_results] for k in metric_keys}

        mean_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
        std_metrics = {k: np.std(v) for k, v in fold_metrics.items()}

        print(f"\n  {'='*60}")
        print(f"  5-Fold Summary: {model_key}")
        print(f"  {'='*60}")
        for k in metric_keys:
            print(f"  {k:15s}: {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}  "
                  f"({', '.join(f'{v:.4f}' for v in fold_metrics[k])})")

        # ROC curve
        plot_roc_curves(fold_results, model_key, model_dir)
        plot_precision_recall_curves(fold_results, model_key, model_dir)
        plot_calibration_curves(fold_results, model_key, model_dir)

        # Per-fold metrics CSV
        fold_summary_df = pd.DataFrame([
            {"fold": r["fold"], **r["metrics"]} for r in fold_results
        ])
        fold_summary_df.loc[len(fold_summary_df)] = {
            "fold": "mean", **mean_metrics
        }
        fold_summary_df.loc[len(fold_summary_df)] = {
            "fold": "std", **std_metrics
        }
        fold_summary_df.to_csv(os.path.join(model_dir, "fold_summary.csv"),
                                index=False, encoding="utf-8-sig")

        all_model_results[model_key] = {
            "mean": mean_metrics,
            "std": std_metrics,
            "fold_metrics": fold_metrics,
            "fold_results": fold_results,
        }

        # Per-fold JSON
        summary_json = {
            "model": model_key,
            "description": model_info["description"],
            "clinical_set": model_info["clinical_set"],
            "fusion_type": model_info["fusion_type"],
            "backbone": model_info["backbone"],
            "n_folds": len(fold_results),
            "mean": mean_metrics,
            "std": std_metrics,
            "mean_brier_score": float(np.mean([r["brier_score"] for r in fold_results])),
            "std_brier_score": float(np.std([r["brier_score"] for r in fold_results])),
            "per_fold": [{
                "fold": r["fold"],
                "metrics": r["metrics"],
                "threshold": r["threshold"],
                "brier_score": r["brier_score"],
            } for r in fold_results],
        }
        with open(os.path.join(model_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=4, ensure_ascii=False)

    # =========================================================
    # Overall model comparison
    # =========================================================
    if len(all_model_results) > 1:
        plot_summary_comparison(all_model_results, output_dir)

    # Overall comparison table
    print("\n" + "=" * 100)
    print("FINAL COMPARISON — Top 3 Models (5-Fold CV)")
    print("=" * 100)
    print(f"{'Model':<50s} {'AUROC':>14s} {'Acc':>14s} {'Sens':>14s} {'Spec':>14s} {'F1':>14s}")
    print("-" * 120)

    comparison_rows = []
    for name, result in all_model_results.items():
        m, s = result["mean"], result["std"]
        short = name.split("_", 1)[1] if "_" in name else name
        print(f"{short:<50s} "
              f"{m['AUROC']:.4f}±{s['AUROC']:.4f} "
              f"{m['Accuracy']:.4f}±{s['Accuracy']:.4f} "
              f"{m['Sensitivity']:.4f}±{s['Sensitivity']:.4f} "
              f"{m['Specificity']:.4f}±{s['Specificity']:.4f} "
              f"{m['F1']:.4f}±{s['F1']:.4f}")

        comparison_rows.append({
            "model": name,
            **{f"{k}_mean": m[k] for k in m},
            **{f"{k}_std": s[k] for k in s},
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"),
                          index=False, encoding="utf-8-sig")

    print(f"\n[INFO] Results saved to: {output_dir}")
    return output_dir


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test top-k scoliosis lumbar models")
    parser.add_argument("--output_dir", type=str, default="test_results")
    parser.add_argument("--data_dir", type=str, default=f"{PROJECT_ROOT}/data/matched_data")
    parser.add_argument("--label_csv", type=str, default=f"{PROJECT_ROOT}/data/AIS_Surgery_clean.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=None, help="Specific GPU index to use")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_KEYS,
        choices=list(TOP_MODELS.keys()),
        help="Model keys to evaluate",
    )
    parser.add_argument("--gradcam_cases_per_type", type=int, default=1)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    run_test(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        label_csv=args.label_csv,
        batch_size=args.batch_size,
        model_keys=args.models,
        gradcam_cases_per_type=args.gradcam_cases_per_type,
    )
