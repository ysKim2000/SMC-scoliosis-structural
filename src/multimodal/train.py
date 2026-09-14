import os
import json
import time
import shutil
import random
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
from model import ScoliosisLumbarModel


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
        file_list = ["dataset.py", "model.py", "train.py"]

    code_dir = os.path.join(exp_dir, "codes")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for file_name in file_list:
        src = os.path.join(base_dir, file_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(code_dir, os.path.basename(file_name)))


def save_config(config, exp_dir):
    save_path = os.path.join(exp_dir, "config.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def build_dataset(config):
    dataset = ScoliosisLumbarDataset(
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
    return dataset


def split_indices(dataset, test_size=0.2, random_state=42):
    indices = np.arange(len(dataset))
    y = dataset.label_csv["Structural_L"].values.astype(int)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(indices, y))
    return indices[train_idx], indices[test_idx]


def summarize_split(dataset, indices, split_name="train"):
    label_df = dataset.label_csv.iloc[indices].copy()
    counts = label_df["Structural_L"].value_counts().sort_index().to_dict()

    summary = {
        "split": split_name,
        "num_samples": len(label_df),
        "Structural_L_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "positive_ratio": float(label_df["Structural_L"].mean()) if len(label_df) > 0 else 0.0,
        "clinical_set": dataset.clinical_set,
        "clinical_columns": dataset.get_clinical_columns(),
    }
    return summary


def build_dataloaders(dataset, train_indices, test_indices, config):
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    return train_loader, test_loader


def build_model(config, device):
    model = ScoliosisLumbarModel(
        pa_backbone=config["pa_backbone"],
        lat_backbone=config["lat_backbone"],
        clinical_dim=config["clinical_dim"],
        hidden_dim=config["hidden_dim"],
        dropout_p=config["dropout_p"],
        fusion_type=config["fusion_type"],
        num_heads=config.get("num_heads", 8),
        num_transformer_layers=config.get("num_transformer_layers", 2),
        view_dropout_p=config.get("view_dropout_p", 0.0),
        use_aux_heads=config.get("use_aux_heads", False),
        aux_hidden_dim=config.get("aux_hidden_dim", 256),
    ).to(device)

    if torch.cuda.device_count() > 1 and config["use_dataparallel"]:
        model = torch.nn.DataParallel(model)

    return model


def build_optimizer(model, config):
    """AdamW with optional differential LR (backbone vs head).

    If lr_backbone == lr_head (or only lr is set), uses a single param group.
    If lr_backbone != lr_head, backbone params get lr_backbone and all other
    params get lr_head.
    """
    lr          = config["lr"]
    lr_backbone = config.get("lr_backbone", lr)
    lr_head     = config.get("lr_head",     lr)
    wd          = config["weight_decay"]

    if abs(lr_backbone - lr_head) < 1e-12:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    bb_ids = set(
        id(p)
        for p in list(base.pa_model.parameters()) + list(base.lat_model.parameters())
    )
    bb_params = [p for p in model.parameters() if id(p) in bb_ids]
    hd_params  = [p for p in model.parameters() if id(p) not in bb_ids]

    print(f"[Optimizer] Differential LR — backbone={lr_backbone:.1e}  head={lr_head:.1e}")
    return torch.optim.AdamW([
        {"params": bb_params, "lr": lr_backbone},
        {"params": hd_params, "lr": lr_head},
    ], weight_decay=wd)


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
        threshold=config.get("threshold", 0.5)
    )

    pred_df = pd.DataFrame({
        "patient_id": all_patient_ids,
        "label": all_labels,
        "prob": all_probs,
        "pred": (all_probs >= config.get("threshold", 0.5)).astype(int)
    })

    return {"loss": avg_loss, **metrics}, pred_df


def train_one_epoch(
    model, dataloader, criterion, optimizer, device,
    scaler=None, epoch=None, epochs=None,
    use_aux_heads=False, aux_alpha=0.0,
):
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0

    desc = f"Train [{epoch}/{epochs}]" if epoch is not None else "Train"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    def _forward_loss(pa, lat, clin, y):
        if use_aux_heads:
            main_logits, pa_logits, lat_logits = model(pa, lat, clin, return_aux=True)
            loss_main = criterion(main_logits, y)
            loss_pa   = criterion(pa_logits, y)
            loss_lat  = criterion(lat_logits, y)
            loss_aux  = loss_pa + loss_lat
            loss = loss_main + aux_alpha * loss_aux
            return loss, loss_main.detach(), loss_aux.detach()
        else:
            logits = model(pa, lat, clin)
            loss = criterion(logits, y)
            return loss, loss.detach(), torch.tensor(0.0, device=loss.device)

    for batch_data, labels in pbar:
        pa = batch_data["pa"].to(device, non_blocking=True)
        lat = batch_data["lat"].to(device, non_blocking=True)
        clinical = batch_data["clinical"].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                loss, loss_main, loss_aux = _forward_loss(pa, lat, clinical, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_main, loss_aux = _forward_loss(pa, lat, clinical, labels)
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        total_loss      += loss.item()      * bs
        total_main_loss += loss_main.item() * bs
        total_aux_loss  += loss_aux.item()  * bs

        if use_aux_heads:
            pbar.set_postfix({
                "loss":  f"{loss.item():.4f}",
                "main":  f"{loss_main.item():.4f}",
                "aux":   f"{loss_aux.item():.4f}",
            })
        else:
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    n = len(dataloader.dataset)
    return {
        "loss":      total_loss      / n,
        "main_loss": total_main_loss / n,
        "aux_loss":  total_aux_loss  / n,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, config, save_path):
    model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }, save_path)


def save_train_logs(train_logs, exp_dir):
    log_dir = os.path.join(exp_dir, "logs")

    pd.DataFrame(train_logs).to_csv(
        os.path.join(log_dir, "train_log.csv"),
        index=False,
        encoding="utf-8-sig"
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
        encoding="utf-8-sig"
    )


def run_train(config):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = create_experiment_dir(config)
    backup_code_files(exp_dir)
    save_config(config, exp_dir)

    print(f"[INFO] Experiment directory: {exp_dir}")
    print(f"[INFO] Clinical set: {config['clinical_set']}")
    print(f"[INFO] Fusion type: {config['fusion_type']}")

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
            random_state=config["seed"]
        )

    train_summary = summarize_split(dataset, train_indices, split_name="train")
    test_summary = summarize_split(dataset, test_indices, split_name="test")

    with open(os.path.join(exp_dir, "logs", "split_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "num_train": len(train_indices),
            "num_test": len(test_indices),
            "train_summary": train_summary,
            "test_summary": test_summary,
        }, f, indent=4, ensure_ascii=False)

    train_loader, test_loader = build_dataloaders(dataset, train_indices, test_indices, config)
    model = build_model(config, device)

    pos_weight = compute_pos_weight(dataset, train_indices, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = build_optimizer(model, config)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["step_size"],
        gamma=config["gamma"]
    )

    scaler = torch.amp.GradScaler("cuda") if (config["use_amp"] and device.type == "cuda") else None

    train_logs = []
    best_auroc = -float("inf")
    best_loss = float("inf")
    best_auroc_epoch = -1
    best_loss_epoch = -1

    patience = config.get("patience", 7)
    early_stop_counter = 0

    use_aux_heads = config.get("use_aux_heads", False)
    aux_alpha     = float(config.get("aux_alpha", 0.0))

    if use_aux_heads:
        print(f"[INFO] Aux heads enabled (alpha={aux_alpha})")

    for epoch in range(1, config["epochs"] + 1):
        start_time = time.time()

        train_loss_dict = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            epochs=config["epochs"],
            use_aux_heads=use_aux_heads,
            aux_alpha=aux_alpha,
        )
        train_loss      = train_loss_dict["loss"]
        train_main_loss = train_loss_dict["main_loss"]
        train_aux_loss  = train_loss_dict["aux_loss"]

        test_metrics, test_pred_df = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            config=config,
            epoch=epoch,
            epochs=config["epochs"]
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_main_loss": train_main_loss,
            "train_aux_loss": train_aux_loss,
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["Accuracy"],
            "test_sens": test_metrics["Sensitivity"],
            "test_spec": test_metrics["Specificity"],
            "test_auroc": test_metrics["AUROC"],
            "lr": lr_now,
            "elapsed_sec": time.time() - start_time,
        }
        train_logs.append(epoch_log)

        aux_str = (
            f" | Train Aux: {train_aux_loss:.4f}" if use_aux_heads else ""
        )
        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f}"
            f"{aux_str} | "
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Acc: {test_metrics['Accuracy']:.4f} | "
            f"Sens: {test_metrics['Sensitivity']:.4f} | "
            f"Spec: {test_metrics['Specificity']:.4f} | "
            f"AUROC: {test_metrics['AUROC']:.4f} | "
            f"LR: {lr_now:.6f}"
        )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            config,
            os.path.join(exp_dir, "checkpoints", "last.pth")
        )

        current_auroc = test_metrics["AUROC"]
        current_loss = test_metrics["loss"]

        if current_auroc > best_auroc:
            best_auroc = current_auroc
            best_auroc_epoch = epoch

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                os.path.join(exp_dir, "checkpoints", "best_auroc.pth")
            )

            best_auroc_summary = {
                "Best_AUROC": epoch_log,
                "clinical_set": config["clinical_set"],
                "clinical_dim": config["clinical_dim"],
                "clinical_columns": config["clinical_columns"],
            }

            with open(os.path.join(exp_dir, "logs", "best_auroc_summary.json"), "w", encoding="utf-8") as f:
                json.dump(best_auroc_summary, f, indent=4, ensure_ascii=False)

            save_test_results(
                metrics=test_metrics,
                pred_df=test_pred_df,
                exp_dir=exp_dir,
                prefix="best_auroc_test"
            )

            print(
                f"[BEST AUROC UPDATED] "
                f"Epoch {epoch:03d} | "
                f"AUROC: {current_auroc:.4f} | "
                f"Loss: {current_loss:.4f} | "
                f"Acc: {test_metrics['Accuracy']:.4f} | "
                f"Sens: {test_metrics['Sensitivity']:.4f} | "
                f"Spec: {test_metrics['Specificity']:.4f}"
            )

        if current_loss < best_loss:
            best_loss = current_loss
            best_loss_epoch = epoch
            early_stop_counter = 0

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                os.path.join(exp_dir, "checkpoints", "best_loss.pth")
            )

            best_loss_summary = {
                "Best_Loss": epoch_log,
                "clinical_set": config["clinical_set"],
                "clinical_dim": config["clinical_dim"],
                "clinical_columns": config["clinical_columns"],
            }

            with open(os.path.join(exp_dir, "logs", "best_loss_summary.json"), "w", encoding="utf-8") as f:
                json.dump(best_loss_summary, f, indent=4, ensure_ascii=False)

            save_test_results(
                metrics=test_metrics,
                pred_df=test_pred_df,
                exp_dir=exp_dir,
                prefix="best_loss_test"
            )

            print(
                f"[BEST LOSS UPDATED] "
                f"Epoch {epoch:03d} | "
                f"Loss: {current_loss:.4f} | "
                f"AUROC: {current_auroc:.4f} | "
                f"Acc: {test_metrics['Accuracy']:.4f} | "
                f"Sens: {test_metrics['Sensitivity']:.4f} | "
                f"Spec: {test_metrics['Specificity']:.4f}"
            )
        else:
            early_stop_counter += 1
            print(f"[EARLY STOP] No loss improvement for {early_stop_counter}/{patience} epochs")

        save_train_logs(train_logs, exp_dir)

        if early_stop_counter >= patience:
            print(
                f"[EARLY STOP TRIGGERED] "
                f"Stopped at epoch {epoch:03d} | "
                f"Best Loss: {best_loss:.4f} (epoch {best_loss_epoch:03d}) | "
                f"Best AUROC: {best_auroc:.4f} (epoch {best_auroc_epoch:03d})"
            )
            break

    final_summary = {
        "best_auroc": float(best_auroc),
        "best_auroc_epoch": int(best_auroc_epoch),
        "best_loss": float(best_loss),
        "best_loss_epoch": int(best_loss_epoch),
        "clinical_set": config["clinical_set"],
        "clinical_dim": config["clinical_dim"],
        "clinical_columns": config["clinical_columns"],
    }
    with open(os.path.join(exp_dir, "logs", "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=4, ensure_ascii=False)

    print(
        f"[INFO] Training finished. "
        f"Best AUROC: {best_auroc:.4f} at epoch {best_auroc_epoch:03d} | "
        f"Best Loss: {best_loss:.4f} at epoch {best_loss_epoch:03d}"
    )

    return exp_dir


if __name__ == "__main__":
    config = {
        "mode": "train",
        "data_dir": "data/matched_data",
        "label_csv": "data/AIS_Surgery_clean.csv",

        "experiment_root": "experiments_lumbar/scoliosis_structural",
        "experiment_name": None,

        "clinical_set": "global_lumbar",
        "img_size": 512,

        "pa_backbone": "convnext_small.fb_in1k",
        "lat_backbone": "convnext_small.fb_in1k",
        "fusion_type": "self_attn",

        "hidden_dim": 256,
        "dropout_p": 0.4,
        "num_heads": 8,
        "num_transformer_layers": 2,

        "epochs": 50,
        "batch_size": 32,
        "num_workers": 8,

        "lr": 1e-4,
        "weight_decay": 5e-4,
        "step_size": 10,
        "gamma": 0.5,

        "seed": 42,
        "test_size": 0.2,
        "threshold": 0.5,
        "patience": 7,

        "use_amp": True,
        "use_dataparallel": True,
        "use_cache": True,
        "cache_dir": "cache",
        "save_images": False,
        "save_dir": "experiments_lumbar",
    }

    run_train(config)
