import os
import json
import time
import shutil
import random
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import ScoliosisLumbarDataset, custom_collate_fn
from ablation_model import ScoliosisLumbarAblationModel


def build_logger(exp_dir):
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "console.log")

    logger_name = f"ablation_train_{os.path.basename(exp_dir)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_experiment_dir(config):
    exp_root = config["experiment_root"]
    os.makedirs(exp_root, exist_ok=True)

    if config.get("experiment_name") is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"exp_{timestamp}"
    else:
        exp_name = config["experiment_name"]

    exp_dir = os.path.join(exp_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "codes"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    return exp_dir


def backup_code_files(exp_dir, file_list=None):
    if file_list is None:
        file_list = [
            "dataset.py",
            "ablation_model.py",
            "ablation_train.py",
            "run_cv_sweep.py",
        ]

    code_dir = os.path.join(exp_dir, "codes")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for file_name in file_list:
        src = os.path.join(base_dir, file_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(code_dir, file_name))


def save_config(config, exp_dir):
    save_path = os.path.join(exp_dir, "config.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def build_dataset(config):
    return ScoliosisLumbarDataset(
        data_dir=config["data_dir"],
        label_csv=config["label_csv"],
        img_size=config["img_size"],
        mode="train",
        clinical_set=config["clinical_set"],
        use_cache=config["use_cache"],
        cache_dir=config["cache_dir"],
        save_images=config.get("save_images", False),
        save_dir=config.get("save_dir", "experiments_lumbar"),
    )


def split_indices(dataset, test_size=0.2, random_state=42):
    indices = np.arange(len(dataset))
    y = dataset.label_csv["Structural_L"].values.astype(int)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(indices, y))
    return indices[train_idx], indices[test_idx]


def summarize_split(dataset, indices, split_name="train"):
    label_df = dataset.label_csv.iloc[indices].copy()
    counts = label_df["Structural_L"].value_counts().sort_index().to_dict()
    return {
        "split": split_name,
        "num_samples": len(label_df),
        "Structural_L_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "positive_ratio": float(label_df["Structural_L"].mean()) if len(label_df) > 0 else 0.0,
        "clinical_set": dataset.clinical_set,
        "clinical_columns": dataset.get_clinical_columns(),
    }


def build_dataloaders(dataset, train_indices, test_indices, config):
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )
    return train_loader, test_loader


def build_model(config, device):
    model = ScoliosisLumbarAblationModel(
        pa_backbone=config["pa_backbone"],
        lat_backbone=config["lat_backbone"],
        clinical_dim=config["clinical_dim"],
        hidden_dim=config["hidden_dim"],
        dropout_p=config["dropout_p"],
        use_pa=config["use_pa"],
        use_lat=config["use_lat"],
        use_clinical=config["use_clinical"],
        fusion_type=config.get("fusion_type", "film"),
    ).to(device)

    if torch.cuda.device_count() > 1 and config["use_dataparallel"]:
        model = torch.nn.DataParallel(model)
    return model


def compute_pos_weight(dataset, train_indices, device):
    labels = dataset.label_csv.iloc[train_indices]["Structural_L"].values.astype(np.float32)
    pos_count = labels.sum()
    neg_count = len(labels) - pos_count
    pos_weight = neg_count / (pos_count + 1e-8)
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")

    return {
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "AUROC": auroc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, config, epoch=None, epochs=None):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_patient_ids = [], [], []

    desc = f"Test  [{epoch}/{epochs}]" if epoch is not None else "Test"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    for batch_data, labels in pbar:
        pa = batch_data["pa"].to(device, non_blocking=True)
        lat = batch_data["lat"].to(device, non_blocking=True)
        clinical = batch_data["clinical"].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(pa, lat, clinical)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_patient_ids.extend(batch_data["patient_id"])
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    avg_loss = total_loss / len(dataloader.dataset)

    metrics = compute_binary_metrics(
        y_true=all_labels,
        y_prob=all_probs,
        threshold=config.get("threshold", 0.5),
    )
    pred_df = pd.DataFrame(
        {
            "patient_id": all_patient_ids,
            "label": all_labels,
            "prob": all_probs,
            "pred": (all_probs >= config.get("threshold", 0.5)).astype(int),
        }
    )
    return {"loss": avg_loss, **metrics}, pred_df


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, epoch=None, epochs=None):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    desc = f"Train [{epoch}/{epochs}]" if epoch is not None else "Train"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    for batch_data, labels in pbar:
        pa = batch_data["pa"].to(device, non_blocking=True)
        lat = batch_data["lat"].to(device, non_blocking=True)
        clinical = batch_data["clinical"].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(pa, lat, clinical)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pa, lat, clinical)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits)
        total_loss += loss.item() * labels.size(0)
        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    avg_loss = total_loss / len(dataloader.dataset)
    train_metrics = compute_binary_metrics(
        y_true=all_labels,
        y_prob=all_probs,
        threshold=0.5,
    )
    return avg_loss, train_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, config, save_path):
    model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
        },
        save_path,
    )


def save_train_logs(train_logs, exp_dir):
    log_dir = os.path.join(exp_dir, "logs")
    pd.DataFrame(train_logs).to_csv(
        os.path.join(log_dir, "train_log.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    with open(os.path.join(log_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_logs, f, indent=4, ensure_ascii=False)


def save_test_results(metrics, pred_df, exp_dir, prefix="epoch_test"):
    log_dir = os.path.join(exp_dir, "logs")
    with open(os.path.join(log_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    pred_df.to_csv(
        os.path.join(log_dir, f"{prefix}_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def run_train(config):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = create_experiment_dir(config)
    backup_code_files(exp_dir)
    save_config(config, exp_dir)
    logger = build_logger(exp_dir)

    logger.info(f"[INFO] Experiment directory: {exp_dir}")
    logger.info(f"[INFO] Ablation: {config['ablation_name']}")
    logger.info(
        "[INFO] Modalities - "
        f"PA: {config['use_pa']} | LAT: {config['use_lat']} | Clinical: {config['use_clinical']}"
    )

    dataset = build_dataset(config)
    config["clinical_dim"] = dataset.get_clinical_dim()
    config["clinical_columns"] = dataset.get_clinical_columns()

    if "train_indices" in config and "test_indices" in config:
        train_indices = np.array(config["train_indices"])
        test_indices = np.array(config["test_indices"])
    else:
        train_indices, test_indices = split_indices(
            dataset,
            test_size=config["test_size"],
            random_state=config["seed"],
        )

    train_summary = summarize_split(dataset, train_indices, split_name="train")
    test_summary = summarize_split(dataset, test_indices, split_name="test")
    with open(os.path.join(exp_dir, "logs", "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_train": len(train_indices),
                "num_test": len(test_indices),
                "train_summary": train_summary,
                "test_summary": test_summary,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    train_loader, test_loader = build_dataloaders(dataset, train_indices, test_indices, config)
    model = build_model(config, device)

    pos_weight = compute_pos_weight(dataset, train_indices, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["step_size"],
        gamma=config["gamma"],
    )
    scaler = torch.amp.GradScaler("cuda") if (config["use_amp"] and device.type == "cuda") else None

    train_logs = []
    best_auroc = -float("inf")
    best_loss = float("inf")
    best_auroc_epoch = -1
    best_loss_epoch = -1
    best_auroc_model_state = None
    patience = config.get("patience", 7)
    early_stop_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        start_time = time.time()

        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            epochs=config["epochs"],
        )

        test_metrics, test_pred_df = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            config=config,
            epoch=epoch,
            epochs=config["epochs"],
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["Accuracy"],
            "train_sens": train_metrics["Sensitivity"],
            "train_spec": train_metrics["Specificity"],
            "train_auroc": train_metrics["AUROC"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["Accuracy"],
            "test_sens": test_metrics["Sensitivity"],
            "test_spec": test_metrics["Specificity"],
            "test_auroc": test_metrics["AUROC"],
            "lr": lr_now,
            "elapsed_sec": time.time() - start_time,
        }
        train_logs.append(epoch_log)

        logger.info(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_metrics['Accuracy']:.4f} | "
            f"Train Sens: {train_metrics['Sensitivity']:.4f} | "
            f"Train Spec: {train_metrics['Specificity']:.4f} | "
            f"Train AUROC: {train_metrics['AUROC']:.4f} | "
            f"Val Loss: {test_metrics['loss']:.4f} | "
            f"Val Acc: {test_metrics['Accuracy']:.4f} | "
            f"Val Sens: {test_metrics['Sensitivity']:.4f} | "
            f"Val Spec: {test_metrics['Specificity']:.4f} | "
            f"Val AUROC: {test_metrics['AUROC']:.4f} | "
            f"LR: {lr_now:.6f}"
        )

        if config.get("save_last_checkpoint", True):
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                os.path.join(exp_dir, "checkpoints", "last.pth"),
            )

        current_auroc = test_metrics["AUROC"]
        current_loss = test_metrics["loss"]

        if current_auroc > best_auroc:
            best_auroc = current_auroc
            best_auroc_epoch = epoch
            if config.get("defer_best_auroc_checkpoint", False):
                source_state = (
                    model.module.state_dict()
                    if isinstance(model, torch.nn.DataParallel)
                    else model.state_dict()
                )
                best_auroc_model_state = {
                    key: value.detach().cpu().clone() for key, value in source_state.items()
                }
            else:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    config,
                    os.path.join(exp_dir, "checkpoints", "best_auroc.pth"),
                )
            best_auroc_summary = {
                "Best_AUROC": epoch_log,
                "clinical_set": config["clinical_set"],
                "clinical_dim": config["clinical_dim"],
                "clinical_columns": config["clinical_columns"],
                "ablation_name": config["ablation_name"],
            }
            with open(os.path.join(exp_dir, "logs", "best_auroc_summary.json"), "w", encoding="utf-8") as f:
                json.dump(best_auroc_summary, f, indent=4, ensure_ascii=False)
            save_test_results(
                metrics=test_metrics,
                pred_df=test_pred_df,
                exp_dir=exp_dir,
                prefix="best_auroc_test",
            )

        if current_loss < best_loss:
            best_loss = current_loss
            best_loss_epoch = epoch
            early_stop_counter = 0
            if config.get("save_best_loss_checkpoint", True):
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    config,
                    os.path.join(exp_dir, "checkpoints", "best_loss.pth"),
                )
            best_loss_summary = {
                "Best_Loss": epoch_log,
                "clinical_set": config["clinical_set"],
                "clinical_dim": config["clinical_dim"],
                "clinical_columns": config["clinical_columns"],
                "ablation_name": config["ablation_name"],
            }
            with open(os.path.join(exp_dir, "logs", "best_loss_summary.json"), "w", encoding="utf-8") as f:
                json.dump(best_loss_summary, f, indent=4, ensure_ascii=False)
            save_test_results(
                metrics=test_metrics,
                pred_df=test_pred_df,
                exp_dir=exp_dir,
                prefix="best_loss_test",
            )
        else:
            early_stop_counter += 1

        save_train_logs(train_logs, exp_dir)
        if early_stop_counter >= patience:
            break

    if config.get("defer_best_auroc_checkpoint", False) and best_auroc_model_state is not None:
        torch.save(
            {
                "epoch": best_auroc_epoch,
                "model_state_dict": best_auroc_model_state,
                "optimizer_state_dict": None,
                "scheduler_state_dict": None,
                "config": config,
            },
            os.path.join(exp_dir, "checkpoints", "best_auroc.pth"),
        )

    final_summary = {
        "best_auroc": float(best_auroc),
        "best_auroc_epoch": int(best_auroc_epoch),
        "best_loss": float(best_loss),
        "best_loss_epoch": int(best_loss_epoch),
        "clinical_set": config["clinical_set"],
        "clinical_dim": config["clinical_dim"],
        "clinical_columns": config["clinical_columns"],
        "ablation_name": config["ablation_name"],
        "use_pa": bool(config["use_pa"]),
        "use_lat": bool(config["use_lat"]),
        "use_clinical": bool(config["use_clinical"]),
    }
    with open(os.path.join(exp_dir, "logs", "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)

    print(
        f"[INFO] Training finished. "
        f"Best AUROC: {best_auroc:.4f} at epoch {best_auroc_epoch:03d} | "
        f"Best Loss: {best_loss:.4f} at epoch {best_loss_epoch:03d}"
    )
    logger.info(
        f"[INFO] Training finished. "
        f"Best AUROC: {best_auroc:.4f} at epoch {best_auroc_epoch:03d} | "
        f"Best Loss: {best_loss:.4f} at epoch {best_loss_epoch:03d}"
    )
    return exp_dir
