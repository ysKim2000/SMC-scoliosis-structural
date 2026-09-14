"""
gradcam.py
Generates and compares multiple CAM methods on train/test data.

Supported CAM methods:
  gradcam     — Grad-CAM (Selvaraju et al., 2017)
  gradcampp   — Grad-CAM++ (Chattopadhay et al., 2018)
  hirescam    — HiResCAM (Draelos & Carin, 2020)
  scorecam    — Score-CAM (Wang et al., 2020)
  eigencam    — EigenCAM (Muhammad & Yeasin, 2020)
  layercam    — LayerCAM (Jiang et al., 2021)

Usage:
    # Rank3, full train split, all CAM methods
    python gradcam.py --model Rank3 --split train --gpu 3

    # Selected CAM methods only
    python gradcam.py --model Rank3 --cam_methods gradcam gradcampp --gpu 3

    # FN cases only (analysis of missed patients)
    python gradcam.py --model Rank3 --outcomes FN FP --gpu 3
"""

import pathlib
import os
import argparse
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import ScoliosisLumbarDataset, custom_collate_fn
from model import ScoliosisLumbarModel

# =========================================================
# Config
# =========================================================
PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])

TOP_MODELS = {
    "Rank1": {
        "description": "global_lumbar + gated + resnet101 (AUROC 0.9222)",
        "checkpoint_pattern": f"{PROJECT_ROOT}/experiments_lumbar/scoliosis_structural/cv5_20260330_213950_512/global_lumbar__gated__resnet101_tv_in1k/fold{{fold}}/checkpoints/best_auroc.pth",
    },
    "Rank2": {
        "description": "global_lumbar + concat + convnext_tiny (AUROC 0.9222)",
        "checkpoint_pattern": f"{PROJECT_ROOT}/experiments_lumbar/scoliosis_structural/cv5_20260330_213950_512/global_lumbar__concat__convnext_tiny_fb_in1k/fold{{fold}}/checkpoints/best_auroc.pth",
    },
    "Rank3": {
        "description": "lumbar_only + film + resnet101 (AUROC 0.9122)",
        "checkpoint_pattern": f"{PROJECT_ROOT}/experiments_lumbar/scoliosis_structural/cv5_20260330_213940_512/lumbar_only__film__resnet101_tv_in1k/fold{{fold}}/checkpoints/best_auroc.pth",
    },
}

CAM_METHODS = ["gradcam", "gradcampp", "hirescam", "scorecam", "eigencam", "layercam"]


# =========================================================
# Utilities
# =========================================================
def find_last_conv_layer(module):
    """Return the last Conv2d with kernel 3×3 or larger (excluding 1×1 pointwise)."""
    last_conv = None
    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d) and child.kernel_size not in ((1, 1), (1,)):
            last_conv = child
    if last_conv is None:
        # fallback: last one overall, including 1×1
        for child in module.modules():
            if isinstance(child, torch.nn.Conv2d):
                last_conv = child
    if last_conv is None:
        raise ValueError("No Conv2d layer found.")
    return last_conv


def denormalize_image(tensor, mean, std):
    mean_t = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    image = tensor * std_t + mean_t
    return torch.clamp(image, 0, 1).detach().cpu().permute(1, 2, 0).numpy()


def overlay_cam_on_image(image_rgb, cam_map, alpha=0.35):
    image_uint8 = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(
        np.clip(cam_map * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_uint8, 1.0 - alpha, heatmap, alpha, 0)


def normalize_cam(cam):
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


def resize_cam(cam, target_size):
    """cam: (H, W) numpy → resized numpy"""
    cam_t = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_t = F.interpolate(cam_t, size=target_size, mode="bilinear", align_corners=False)
    return cam_t[0, 0].numpy()


# =========================================================
# CAM Implementations
# =========================================================

def _get_activations_and_gradients(model, img_t, other_t, clin_t, view, retain=False):
    """Shared: collect activations and gradients via forward/backward hooks."""
    encoder = model.pa_model if view == "pa" else model.lat_model
    target_layer = find_last_conv_layer(encoder.feature_extractor)
    activations, gradients = [], []

    fh = target_layer.register_forward_hook(lambda _, __, o: activations.append(o))
    bh = target_layer.register_full_backward_hook(lambda _, gi, go: gradients.append(go[0]))

    model.zero_grad(set_to_none=True)
    logits = model(img_t, other_t, clin_t) if view == "pa" \
        else model(other_t, img_t, clin_t)
    score = logits.squeeze()
    score.backward(retain_graph=retain)

    act = activations[-1].detach()
    grad = gradients[-1].detach()
    prob = float(torch.sigmoid(score).item())

    fh.remove()
    bh.remove()
    model.zero_grad(set_to_none=True)
    return act, grad, prob, target_layer, encoder


def cam_gradcam(act, grad, img_size):
    """Grad-CAM: global average pool of gradients"""
    weights = grad.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))  # (1, 1, H, W)
    cam = F.interpolate(cam, size=img_size, mode="bilinear", align_corners=False)
    return normalize_cam(cam[0, 0].cpu().numpy())


def cam_gradcampp(act, grad, img_size):
    """Grad-CAM++: weights based on second-order derivatives."""
    # alpha = grad^2 / (2*grad^2 + act * grad^3 + eps)
    grad2 = grad ** 2
    grad3 = grad ** 3
    denom = 2 * grad2 + (act * grad3).sum(dim=(2, 3), keepdim=True) + 1e-8
    alpha = grad2 / denom
    weights = (alpha * torch.relu(grad)).sum(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=img_size, mode="bilinear", align_corners=False)
    return normalize_cam(cam[0, 0].cpu().numpy())


def cam_hirescam(act, grad, img_size):
    """HiResCAM: element-wise grad × act (no global pooling)"""
    cam = torch.relu((grad * act).sum(dim=1, keepdim=True))  # (1, 1, H, W)
    cam = F.interpolate(cam, size=img_size, mode="bilinear", align_corners=False)
    return normalize_cam(cam[0, 0].cpu().numpy())


def cam_eigencam(act, img_size):
    """EigenCAM: PCA 1st principal component of activation"""
    # act: (1, C, H, W)
    a = act[0]  # (C, H, W)
    C, H, W = a.shape
    # reshape to (HW, C) — samples × features for SVD
    a_flat = a.reshape(C, H * W).cpu().numpy()  # (C, HW)
    a_centered = a_flat - a_flat.mean(axis=0, keepdims=True)  # center over channels
    # SVD on (C, HW): first right singular vector of V = 1st PC in HW space
    _, _, Vt = np.linalg.svd(a_centered, full_matrices=False)
    # Vt[0]: (HW,) → 1st principal component
    cam = Vt[0].reshape(H, W)
    cam = np.maximum(cam, 0)
    cam = resize_cam(cam, img_size)
    return normalize_cam(cam)


def cam_layercam(act, grad, img_size):
    """LayerCAM: spatial gradient × activation (positive only)"""
    # element-wise, keep spatial detail
    spatial_weights = torch.relu(grad)                      # (1, C, H, W)
    cam = torch.relu((spatial_weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=img_size, mode="bilinear", align_corners=False)
    return normalize_cam(cam[0, 0].cpu().numpy())


def cam_scorecam(model, img_t, other_t, clin_t, view, act, img_size, batch_size=16):
    """Score-CAM: measure contribution via activation channel masking (gradient-free)."""
    C, H, W = act[0].shape
    target_h, target_w = img_size

    # baseline score (zero input)
    with torch.no_grad():
        baseline = torch.zeros_like(img_t)
        logit_base = model(baseline, other_t, clin_t) if view == "pa" \
            else model(other_t, baseline, clin_t)
        score_base = torch.sigmoid(logit_base).squeeze().item()

    scores = []
    with torch.no_grad():
        for start in range(0, C, batch_size):
            end = min(start + batch_size, C)
            batch_masks = []
            for c in range(start, end):
                mask = act[0, c].cpu().numpy()
                mask = resize_cam(mask, (target_h, target_w))
                mask = normalize_cam(mask)
                batch_masks.append(mask)

            # (batch, 1, H, W) → broadcast over 3 channels
            masks_t = torch.tensor(
                np.stack(batch_masks), dtype=torch.float32, device=img_t.device
            ).unsqueeze(1)  # (B, 1, H, W)

            masked = img_t * masks_t  # (B, 3, H, W)
            other_rep = other_t.expand(end - start, -1, -1, -1)
            clin_rep = clin_t.expand(end - start, -1)

            if view == "pa":
                logits = model(masked, other_rep, clin_rep)
            else:
                logits = model(other_rep, masked, clin_rep)
            s = torch.sigmoid(logits).squeeze().cpu().numpy()
            scores.extend(s.tolist() if s.ndim > 0 else [float(s)])

    scores = np.array(scores) - score_base
    scores = np.maximum(scores, 0)

    # weighted sum of activation maps
    cam = np.zeros((H, W), dtype=np.float32)
    for c, w in enumerate(scores):
        mask = act[0, c].cpu().numpy()
        cam += w * mask

    cam = resize_cam(cam, img_size)
    return normalize_cam(cam)


# =========================================================
# Unified CAM dispatcher
# =========================================================
def compute_cam(model, img_t, other_t, clin_t, view, method):
    """Compute the CAM matching ``method`` and return ``(cam_np, prob)``.

    For ``view="pa"``, ``img_t`` is PA and ``other_t`` is LAT.
    For ``view="lat"``, ``img_t`` must be LAT and ``other_t`` must be PA.
    """
    img_size = (img_t.shape[-2], img_t.shape[-1])
    model.eval()

    if method == "scorecam":
        # scorecam needs no gradients — collect activations only
        encoder = model.pa_model if view == "pa" else model.lat_model
        target_layer = find_last_conv_layer(encoder.feature_extractor)
        activations = []
        fh = target_layer.register_forward_hook(lambda _, __, o: activations.append(o))
        with torch.no_grad():
            logits = model(img_t, other_t, clin_t) if view == "pa" \
                else model(other_t, img_t, clin_t)
        prob = float(torch.sigmoid(logits).squeeze().item())
        fh.remove()
        act = activations[-1].detach()
        cam = cam_scorecam(model, img_t, other_t, clin_t, view, act, img_size)
        return cam, prob

    elif method == "eigencam":
        # eigencam also needs no gradients
        encoder = model.pa_model if view == "pa" else model.lat_model
        target_layer = find_last_conv_layer(encoder.feature_extractor)
        activations = []
        fh = target_layer.register_forward_hook(lambda _, __, o: activations.append(o))
        with torch.no_grad():
            logits = model(img_t, other_t, clin_t) if view == "pa" \
                else model(other_t, img_t, clin_t)
        prob = float(torch.sigmoid(logits).squeeze().item())
        fh.remove()
        act = activations[-1].detach()
        cam = cam_eigencam(act, img_size)
        return cam, prob

    else:
        act, grad, prob, _, _ = _get_activations_and_gradients(
            model, img_t, other_t, clin_t, view
        )
        if method == "gradcam":
            cam = cam_gradcam(act, grad, img_size)
        elif method == "gradcampp":
            cam = cam_gradcampp(act, grad, img_size)
        elif method == "hirescam":
            cam = cam_hirescam(act, grad, img_size)
        elif method == "layercam":
            cam = cam_layercam(act, grad, img_size)
        else:
            raise ValueError(f"Unknown CAM method: {method}")
        return cam, prob


# =========================================================
# Model loading / Inference
# =========================================================
def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ScoliosisLumbarModel(
        pa_backbone=cfg["pa_backbone"],
        lat_backbone=cfg["lat_backbone"],
        clinical_dim=cfg["clinical_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout_p=cfg.get("dropout_p", 0.4),
        fusion_type=cfg["fusion_type"],
        num_heads=cfg.get("num_heads", 8),
        num_transformer_layers=cfg.get("num_transformer_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def run_inference(model, dataloader, device, threshold=0.5):
    all_labels, all_probs, all_pids = [], [], []
    for batch_data, labels in dataloader:
        pa = batch_data["pa"].to(device, non_blocking=True)
        lat = batch_data["lat"].to(device, non_blocking=True)
        clin = batch_data["clinical"].to(device, non_blocking=True)
        probs = torch.sigmoid(model(pa, lat, clin))
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
        all_pids.extend(batch_data["patient_id"])

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    pred_df = pd.DataFrame({
        "patient_id": all_pids,
        "label": all_labels.astype(int),
        "prob": all_probs,
        "pred": (all_probs >= threshold).astype(int),
    })
    pred_df["outcome"] = pred_df.apply(
        lambda r: "TP" if r.label == 1 and r.pred == 1
        else "TN" if r.label == 0 and r.pred == 0
        else "FP" if r.label == 0 and r.pred == 1
        else "FN", axis=1,
    )
    return pred_df


# =========================================================
# Save: 1 patient × N methods → 1 figure
# =========================================================
def save_cam_comparison(model, dataset, pid_to_idx, row, mean, std, device, methods, save_path):
    idx = pid_to_idx[row["patient_id"]]
    batch_data, _ = dataset[idx]

    pa = batch_data["pa"].unsqueeze(0).to(device)
    lat = batch_data["lat"].unsqueeze(0).to(device)
    clin = batch_data["clinical"].unsqueeze(0).to(device)

    pa_img = denormalize_image(batch_data["pa"], mean, std)
    lat_img = denormalize_image(batch_data["lat"], mean, std)

    # rows: PA / LAT, cols: Original + each method
    n_methods = len(methods)
    n_cols = 1 + n_methods   # original + CAMs
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    # originals
    axes[0, 0].imshow(pa_img);  axes[0, 0].set_title("PA Original", fontsize=9)
    axes[1, 0].imshow(lat_img); axes[1, 0].set_title("LAT Original", fontsize=9)

    prob_pa = None
    for col, method in enumerate(methods, start=1):
        pa_cam, prob_pa = compute_cam(model, pa, lat, clin, view="pa", method=method)
        lat_cam, _      = compute_cam(model, lat, pa, clin, view="lat", method=method)

        axes[0, col].imshow(overlay_cam_on_image(pa_img, pa_cam))
        axes[0, col].set_title(method, fontsize=9)
        axes[1, col].imshow(overlay_cam_on_image(lat_img, lat_cam))
        axes[1, col].set_title(method, fontsize=9)

    for ax in axes.flat:
        ax.axis("off")

    prob_str = f"{prob_pa:.3f}" if prob_pa is not None else "N/A"
    fig.suptitle(
        f"PID {row['patient_id']} | {row['outcome']} | "
        f"label={row['label']} pred={row['pred']} prob={prob_str}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


# =========================================================
# Main
# =========================================================
def run_gradcam(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = TOP_MODELS[args.model]
    methods = args.cam_methods
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(
        PROJECT_ROOT, args.output_dir,
        f"gradcam_{timestamp}_{args.model}_{args.split}_{'_'.join(methods)}"
    )
    os.makedirs(out_root, exist_ok=True)

    print(f"[INFO] Output : {out_root}")
    print(f"[INFO] Model  : {args.model} — {model_cfg['description']}")
    print(f"[INFO] Split  : {args.split}")
    print(f"[INFO] Methods: {methods}")
    print(f"[INFO] Folds  : {args.folds}")

    all_rows = []

    for fold in args.folds:
        ckpt_path = model_cfg["checkpoint_pattern"].format(fold=fold)
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*60}\n  Fold {fold}\n{'='*60}")
        model, cfg = load_model(ckpt_path, device)

        dataset = ScoliosisLumbarDataset(
            data_dir=os.path.join(PROJECT_ROOT, "data/matched_data"),
            label_csv=os.path.join(PROJECT_ROOT, "data/AIS_Surgery_clean.csv"),
            img_size=cfg.get("img_size", 512),
            mode="test",
            clinical_set=cfg["clinical_set"],
            use_cache=True,
            cache_dir=os.path.join(PROJECT_ROOT, "cache"),
        )

        split_indices = np.array(
            cfg["train_indices"] if args.split == "train" else cfg["test_indices"]
        )
        loader = DataLoader(
            Subset(dataset, split_indices),
            batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=custom_collate_fn,
        )

        pred_df = run_inference(model, loader, device, threshold=args.threshold)

        if args.outcomes:
            pred_df = pred_df[pred_df["outcome"].isin(args.outcomes)].reset_index(drop=True)

        n = len(pred_df)
        print(f"  Patients : {n}")
        print(f"  Outcomes : {pred_df['outcome'].value_counts().to_dict()}")
        if len(pred_df["label"].unique()) > 1:
            print(f"  AUROC    : {roc_auc_score(pred_df['label'], pred_df['prob']):.4f}")

        fold_dir = os.path.join(out_root, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        pred_df.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False, encoding="utf-8-sig")

        pid_to_idx = {
            dataset.label_csv.iloc[i]["patient_id"]: i
            for i in range(len(dataset))
        }

        for i, (_, row) in enumerate(pred_df.iterrows()):
            fname = f"{row['outcome']}_{row['patient_id']}_p{row['prob']:.3f}.png"
            save_cam_comparison(
                model, dataset, pid_to_idx, row,
                dataset.mean, dataset.std, device,
                methods, os.path.join(fold_dir, fname),
            )
            print(f"  [{i+1:03d}/{n}] {row['outcome']} PID {row['patient_id']} (prob={row['prob']:.3f})")

        row_summary = pred_df.copy()
        row_summary["fold"] = fold
        all_rows.append(row_summary)
        print(f"  [Done] {n} images → {fold_dir}")

    if all_rows:
        summary_df = pd.concat(all_rows, ignore_index=True)
        summary_df.to_csv(os.path.join(out_root, "all_predictions.csv"), index=False, encoding="utf-8-sig")
        print(f"\n[INFO] Total: {len(summary_df)} images")
        print(f"[INFO] Breakdown:\n{summary_df['outcome'].value_counts().to_string()}")


def parse_args():
    p = argparse.ArgumentParser(description="Multi-method CAM for lumbar scoliosis")
    p.add_argument("--model", type=str, default="Rank3", choices=list(TOP_MODELS.keys()))
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument(
        "--cam_methods", nargs="+", type=str, default=CAM_METHODS,
        choices=CAM_METHODS,
        help="CAM methods to compare. Default: all 6",
    )
    p.add_argument(
        "--outcomes", nargs="+", type=str, default=None,
        choices=["TP", "TN", "FP", "FN"],
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="multimodal/test_results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gradcam(args)
