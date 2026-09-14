import os
import json
import argparse
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from train import run_train


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1200)


# =========================================================
# Clinical Set Mapping
# =========================================================
CLINICAL_SET_ALIAS = {
    "setA": "lumbar_only",
    "setB": "global_lumbar",
    "setC": "full_spine",
    "lumbar_only": "lumbar_only",
    "global_lumbar": "global_lumbar",
    "full_spine": "full_spine",
}


# =========================================================
# Load Best Summary
# =========================================================
def load_best_summary(exp_dir):
    best_auroc_path = os.path.join(exp_dir, "logs", "best_auroc_summary.json")
    best_loss_path = os.path.join(exp_dir, "logs", "best_loss_summary.json")
    final_summary_path = os.path.join(exp_dir, "logs", "final_summary.json")
    train_log_path = os.path.join(exp_dir, "logs", "train_log.csv")

    result = {
        "exp_dir": exp_dir,

        "best_auroc_epoch": None,
        "best_auroc_acc": None,
        "best_auroc_sens": None,
        "best_auroc_spec": None,
        "best_auroc": None,
        "best_auroc_loss": None,

        "best_loss_epoch": None,
        "best_loss_acc": None,
        "best_loss_sens": None,
        "best_loss_spec": None,
        "best_loss_auroc": None,
        "best_loss": None,
    }

    # 1) best_auroc_summary.json
    if os.path.exists(best_auroc_path):
        with open(best_auroc_path, "r", encoding="utf-8") as f:
            best_auroc_summary = json.load(f)

        best_auroc_info = best_auroc_summary.get("Best_AUROC", {})
        result.update({
            "best_auroc_epoch": best_auroc_info.get("epoch"),
            "best_auroc_acc": best_auroc_info.get("test_acc"),
            "best_auroc_sens": best_auroc_info.get("test_sens"),
            "best_auroc_spec": best_auroc_info.get("test_spec"),
            "best_auroc": best_auroc_info.get("test_auroc"),
            "best_auroc_loss": best_auroc_info.get("test_loss"),
        })

    # 2) best_loss_summary.json
    if os.path.exists(best_loss_path):
        with open(best_loss_path, "r", encoding="utf-8") as f:
            best_loss_summary = json.load(f)

        best_loss_info = best_loss_summary.get("Best_Loss", {})
        result.update({
            "best_loss_epoch": best_loss_info.get("epoch"),
            "best_loss_acc": best_loss_info.get("test_acc"),
            "best_loss_sens": best_loss_info.get("test_sens"),
            "best_loss_spec": best_loss_info.get("test_spec"),
            "best_loss_auroc": best_loss_info.get("test_auroc"),
            "best_loss": best_loss_info.get("test_loss"),
        })

    # 3) Supplement with final_summary.json
    if os.path.exists(final_summary_path):
        with open(final_summary_path, "r", encoding="utf-8") as f:
            final_summary = json.load(f)

        if result["best_auroc"] is None:
            result["best_auroc"] = final_summary.get("best_auroc")
        if result["best_auroc_epoch"] is None:
            result["best_auroc_epoch"] = final_summary.get("best_auroc_epoch")

        if result["best_loss"] is None:
            result["best_loss"] = final_summary.get("best_loss")
        if result["best_loss_epoch"] is None:
            result["best_loss_epoch"] = final_summary.get("best_loss_epoch")

    # 4) If the summary json is missing, fall back to train_log.csv
    if os.path.exists(train_log_path):
        log_df = pd.read_csv(train_log_path)
        if len(log_df) > 0:
            if result["best_auroc"] is None:
                best_auroc_row = log_df.loc[log_df["test_auroc"].idxmax()]
                result.update({
                    "best_auroc_epoch": int(best_auroc_row["epoch"]),
                    "best_auroc_acc": float(best_auroc_row["test_acc"]),
                    "best_auroc_sens": float(best_auroc_row["test_sens"]),
                    "best_auroc_spec": float(best_auroc_row["test_spec"]),
                    "best_auroc": float(best_auroc_row["test_auroc"]),
                    "best_auroc_loss": float(best_auroc_row["test_loss"]),
                })

            if result["best_loss"] is None:
                best_loss_row = log_df.loc[log_df["test_loss"].idxmin()]
                result.update({
                    "best_loss_epoch": int(best_loss_row["epoch"]),
                    "best_loss_acc": float(best_loss_row["test_acc"]),
                    "best_loss_sens": float(best_loss_row["test_sens"]),
                    "best_loss_spec": float(best_loss_row["test_spec"]),
                    "best_loss_auroc": float(best_loss_row["test_auroc"]),
                    "best_loss": float(best_loss_row["test_loss"]),
                })

    return result


# =========================================================
# Parse Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Lumbar single-task sweep runner")

    parser.add_argument("--mode", type=str, default="cv_train", choices=["cv_train"])
    parser.add_argument("--data_dir", type=str, default="data/matched_data")
    parser.add_argument("--label_csv", type=str, default="data/AIS_Surgery_clean.csv")
    parser.add_argument("--exp_root", type=str, default="experiments_lumbar/scoliosis_structural")
    parser.add_argument("--img_size", type=int, default=512)

    parser.add_argument(
        "--clinical_sets",
        nargs="+",
        default=["setA", "setB", "setC"],
        help="Choose from: setA, setB, setC, lumbar_only, global_lumbar, full_spine"
    )

    parser.add_argument(
        "--model_names",
        nargs="+",
        default=[
            "convnext_small.fb_in1k",
            "convnext_base.fb_in1k",
            "convnext_large.fb_in1k",
        ]
    )

    parser.add_argument(
        "--fusion_types",
        nargs="+",
        default=[
            "concat",
            "projection",
            "film",
            "gated",
            "cross_attn",
            "self_attn",
        ]
    )

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=None,
                        help="Backbone LR (differential). Default: same as --lr")
    parser.add_argument("--lr_head", type=float, default=None,
                        help="Head/fusion LR (differential). Default: same as --lr")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout_p", type=float, default=0.4)
    parser.add_argument("--view_dropout_p", type=float, default=0.0,
                        help="PA view dropout probability (0.0 = disabled)")
    parser.add_argument("--use_aux_heads", action="store_true", default=False,
                        help="Enable per-view auxiliary classifier heads")
    parser.add_argument("--aux_alpha", type=float, default=0.3,
                        help="Weight for auxiliary loss: total = main + alpha * (pa_aux + lat_aux)")
    parser.add_argument("--aux_hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_transformer_layers", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--use_dataparallel", action="store_true", default=True)
    parser.add_argument("--use_cache", action="store_true", default=False)

    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--save_images", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="experiments_lumbar")

    parser.add_argument("--cuda_visible_devices", type=str, default=None)

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.mode == "cv_train":
        sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Normalize clinical set names
        clinical_sets = []
        for item in args.clinical_sets:
            if item not in CLINICAL_SET_ALIAS:
                raise ValueError(
                    f"Unknown clinical set: {item}. "
                    f"Available: {list(CLINICAL_SET_ALIAS.keys())}"
                )
            clinical_sets.append(CLINICAL_SET_ALIAS[item])

        # Load data
        df = pd.read_csv(args.label_csv)

        if "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype(str).str.extract(r"(\d+)")[0].str.zfill(8)
        elif "ID" in df.columns:
            df["patient_id"] = df["ID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(8)
        else:
            raise ValueError("label_csv must contain a patient_id or ID column.")

        # Match only patients whose image files actually exist
        pa_dir = os.path.join(args.data_dir, "PA")
        lat_dir = os.path.join(args.data_dir, "LAT")

        pa_ids = set([f[:8] for f in os.listdir(pa_dir) if f.lower().endswith(".dcm")])
        lat_ids = set([f[:8] for f in os.listdir(lat_dir) if f.lower().endswith(".dcm")])
        available_ids = pa_ids & lat_ids

        df = df[df["patient_id"].isin(available_ids)].reset_index(drop=True)

        # single-task lumbar label
        y = df["Structural_L"].values.astype(int)
        indices = np.arange(len(df))

        skf = StratifiedKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.seed
        )

        summary_rows = []
        SWEEP_ROOT = os.path.join(
            args.exp_root,
            f"cv{args.n_splits}_{sweep_timestamp}_{args.img_size}"
        )
        os.makedirs(SWEEP_ROOT, exist_ok=True)

        print(f"\n[INFO] Sweep root: {SWEEP_ROOT}")
        print(f"[INFO] Clinical sets: {clinical_sets}")
        print(f"[INFO] Fusion types: {args.fusion_types}")
        print(f"[INFO] Model names: {args.model_names}")

        # =====================================================
        # clinical_set -> fusion_type -> model_name -> fold
        # =====================================================
        for clinical_set in clinical_sets:
            for fusion_type in args.fusion_types:
                for model_name in args.model_names:
                    safe_model_name = model_name.replace("/", "_").replace(".", "_").replace("-", "_")
                    model_exp_root = os.path.join(
                        SWEEP_ROOT,
                        f"{clinical_set}__{fusion_type}__{safe_model_name}"
                    )

                    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(indices, y), start=1):
                        print("\n" + "=" * 140)
                        print(
                            f"[CV{args.n_splits}] "
                            f"ClinicalSet: {clinical_set} | "
                            f"Fusion: {fusion_type} | "
                            f"Backbone: {model_name} | "
                            f"Fold: {fold_idx}"
                        )
                        print("=" * 140)

                        config = {
                            "mode": "train",
                            "data_dir": args.data_dir,
                            "label_csv": args.label_csv,

                            "experiment_root": model_exp_root,
                            "experiment_name": f"fold{fold_idx}",

                            "clinical_set": clinical_set,
                            "img_size": args.img_size,

                            "pa_backbone": model_name,
                            "lat_backbone": model_name,
                            "fusion_type": fusion_type,

                            "hidden_dim": args.hidden_dim,
                            "dropout_p": args.dropout_p,
                            "view_dropout_p": args.view_dropout_p,
                            "use_aux_heads": args.use_aux_heads,
                            "aux_alpha": args.aux_alpha,
                            "aux_hidden_dim": args.aux_hidden_dim,
                            "num_heads": args.num_heads,
                            "num_transformer_layers": args.num_transformer_layers,

                            "epochs": args.epochs,
                            "batch_size": args.batch_size,
                            "num_workers": args.num_workers,

                            "lr": args.lr,
                            "lr_backbone": args.lr_backbone if args.lr_backbone is not None else args.lr,
                            "lr_head":     args.lr_head     if args.lr_head     is not None else args.lr,
                            "weight_decay": args.weight_decay,
                            "step_size": args.step_size,
                            "gamma": args.gamma,

                            "seed": args.seed,
                            "threshold": args.threshold,

                            "use_amp": args.use_amp,
                            "use_dataparallel": args.use_dataparallel,
                            "use_cache": args.use_cache,
                            "cache_dir": args.cache_dir,
                            "save_images": args.save_images,
                            "save_dir": args.save_dir,

                            "train_indices": train_idx.tolist(),
                            "test_indices": test_idx.tolist(),
                        }

                        row = {
                            "clinical_set": clinical_set,
                            "fusion_type": fusion_type,
                            "model_name": model_name,
                            "fold": fold_idx,
                            "status": "failed",
                            "error": None,
                            "exp_dir": None,
                            "best_epoch": None,
                            "best_auroc": None,
                            "best_acc": None,
                            "best_sens": None,
                            "best_spec": None,
                        }

                        try:
                            exp_dir = run_train(config)
                            row["status"] = "success"
                            row["exp_dir"] = exp_dir

                            best_result = load_best_summary(exp_dir)
                            row.update(best_result)
                            # load_best_summary uses best_auroc_* keys; map to display keys
                            row["best_acc"]   = best_result.get("best_auroc_acc")
                            row["best_sens"]  = best_result.get("best_auroc_sens")
                            row["best_spec"]  = best_result.get("best_auroc_spec")
                            row["best_epoch"] = best_result.get("best_auroc_epoch")

                        except Exception as e:
                            row["status"] = "failed"
                            row["error"] = str(e)
                            print(
                                f"[ERROR] Failed | "
                                f"ClinicalSet: {clinical_set} | "
                                f"Fusion: {fusion_type} | "
                                f"Model: {model_name} | "
                                f"Fold: {fold_idx}"
                            )
                            print(traceback.format_exc())

                        summary_rows.append(row)

                        # Save intermediate summary
                        summary_df = pd.DataFrame(summary_rows)
                        summary_csv = os.path.join(
                            SWEEP_ROOT,
                            f"cv{args.n_splits}_summary_{sweep_timestamp}.csv"
                        )
                        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

                        print("\n[CURRENT SUMMARY]")
                        show_cols = [
                            "clinical_set",
                            "fusion_type",
                            "model_name",
                            "fold",
                            "status",
                            "best_auroc",
                            "best_acc",
                            "best_sens",
                            "best_spec",
                        ]
                        print(summary_df[show_cols].sort_values(
                            by=["clinical_set", "fusion_type", "model_name", "fold"]
                        ))

        # =====================================================
        # Final Mean Summary
        # =====================================================
        print("\n" + "#" * 140)
        print("[FINAL SUMMARY]")

        final_df = pd.DataFrame(summary_rows)
        success_df = final_df[final_df["status"] == "success"].copy()

        if len(success_df) > 0:
            mean_df = (
                success_df.groupby(["clinical_set", "fusion_type", "model_name"])[
                    ["best_auroc", "best_acc", "best_sens", "best_spec"]
                ]
                .mean()
                .reset_index()
                .sort_values(by="best_auroc", ascending=False)
            )

            print("\n[MEAN OVER FOLDS]")
            print(mean_df)

            mean_csv = os.path.join(
                SWEEP_ROOT,
                f"cv{args.n_splits}_mean_summary_{sweep_timestamp}.csv"
            )
            mean_df.to_csv(mean_csv, index=False, encoding="utf-8-sig")
            print(f"\n[INFO] Mean summary saved to: {mean_csv}")
        else:
            print("[WARNING] No successful runs found.")

        final_csv = os.path.join(
            SWEEP_ROOT,
            f"cv{args.n_splits}_full_summary_{sweep_timestamp}.csv"
        )
        final_df.to_csv(final_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] Full summary saved to: {final_csv}")
