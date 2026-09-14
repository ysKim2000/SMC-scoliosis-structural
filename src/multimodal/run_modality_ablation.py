"""Run the six reduced-modality FiLM ablations on the final model's folds.

The split arrays are copied verbatim from the saved final PA+LAT+clinical
ResNet-101/FiLM configs. Eleven independent workers are packed across four
A6000 GPUs (3/3/2/3); GPU 2 is deliberately assigned fewer workers because it
is already shared by other processes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = (
    ROOT
    / "experiments_lumbar"
    / "scoliosis_structural"
    / "cv5_20260330_213940_512"
    / "lumbar_only__film__resnet101_tv_in1k"
)
RUN_DIR = (
    ROOT
    / "experiments_lumbar_ablation"
    / "scoliosis_structural"
    / "canonical_cv5_final_indices_512"
)
ORCH_DIR = RUN_DIR / "_orchestration"

ARMS = {
    "clinical_only": dict(use_pa=False, use_lat=False, use_clinical=True),
    "pa_only": dict(use_pa=True, use_lat=False, use_clinical=False),
    "lat_only": dict(use_pa=False, use_lat=True, use_clinical=False),
    "pa_lat": dict(use_pa=True, use_lat=True, use_clinical=False),
    "pa_clinical": dict(use_pa=True, use_lat=False, use_clinical=True),
    "lat_clinical": dict(use_pa=False, use_lat=True, use_clinical=True),
}

# Three concurrent jobs on otherwise empty A6000s; two on shared GPU 2.
GPU_WORKERS = {0: 3, 1: 3, 2: 2, 3: 3}

BASE_CONFIG = {
    "mode": "train",
    "data_dir": str(ROOT / "data" / "matched_data"),
    "label_csv": str(ROOT / "data" / "AIS_Surgery_clean.csv"),
    "clinical_set": "lumbar_only",
    "img_size": 512,
    "pa_backbone": "resnet101.tv_in1k",
    "lat_backbone": "resnet101.tv_in1k",
    "fusion_type": "film",
    "hidden_dim": 256,
    "dropout_p": 0.4,
    "epochs": 50,
    "batch_size": 16,
    "num_workers": 2,
    "lr": 0.0001,
    "weight_decay": 0.0005,
    "step_size": 10,
    "gamma": 0.5,
    "seed": 42,
    "threshold": 0.5,
    "patience": 7,
    "use_amp": True,
    "use_dataparallel": False,
    "use_cache": True,
    "cache_dir": "cache",
    "save_images": False,
    "save_dir": str(RUN_DIR),
    "save_last_checkpoint": False,
    "save_best_loss_checkpoint": False,
    "defer_best_auroc_checkpoint": True,
}


def canonical_folds() -> list[tuple[list[int], list[int]]]:
    folds = []
    seen: set[int] = set()
    for fold in range(1, 6):
        cfg = json.loads((FINAL_DIR / f"fold{fold}" / "config.json").read_text())
        train = [int(value) for value in cfg["train_indices"]]
        test = [int(value) for value in cfg["test_indices"]]
        if len(train) != 124 or len(test) != 31 or set(train) & set(test):
            raise RuntimeError(f"Invalid canonical split in fold {fold}")
        if seen & set(test):
            raise RuntimeError(f"Canonical test folds overlap at fold {fold}")
        seen.update(test)
        folds.append((train, test))
    if seen != set(range(155)):
        raise RuntimeError("Canonical test folds do not cover indices 0..154 exactly")
    return folds


def build_jobs() -> list[dict]:
    jobs = []
    for arm, flags in ARMS.items():
        for fold, (train, test) in enumerate(canonical_folds(), 1):
            config = dict(BASE_CONFIG)
            config.update(flags)
            config.update(
                {
                    "experiment_root": str(RUN_DIR / arm),
                    "experiment_name": f"fold{fold}",
                    "ablation_name": arm,
                    "train_indices": train,
                    "test_indices": test,
                }
            )
            jobs.append(
                {
                    "tag": f"canonical_{arm}_fold{fold}",
                    "ablation_name": arm,
                    "fold": fold,
                    "config": config,
                    "keep_checkpoints": True,
                }
            )
    return jobs


def assign_jobs(jobs: list[dict]) -> list[tuple[int, int, list[dict]]]:
    by_gpu: dict[int, list[dict]] = defaultdict(list)
    # Spread every modality's five folds across all four GPUs. The fifth fold
    # rotates so the extra work is balanced.
    arm_number = {arm: index for index, arm in enumerate(ARMS)}
    for job in jobs:
        gpu = (job["fold"] - 1 + arm_number[job["ablation_name"]]) % 4
        by_gpu[gpu].append(job)

    lanes = []
    for gpu, worker_count in GPU_WORKERS.items():
        shards = [by_gpu[gpu][index::worker_count] for index in range(worker_count)]
        lanes.extend((gpu, lane, shard) for lane, shard in enumerate(shards) if shard)
    return lanes


def main() -> None:
    started = time.time()
    ORCH_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    lanes = assign_jobs(jobs)
    print(f"[INFO] run_dir={RUN_DIR}", flush=True)
    print(f"[INFO] jobs={len(jobs)}, workers={len(lanes)}, fusion=film", flush=True)

    processes = []
    result_paths = []
    for gpu, lane, shard in lanes:
        stem = f"gpu{gpu}_worker{lane}"
        job_path = ORCH_DIR / f"{stem}_jobs.json"
        result_path = ORCH_DIR / f"{stem}_results.csv"
        log_path = ORCH_DIR / f"{stem}.log"
        job_path.write_text(json.dumps(shard, indent=2), encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "multimodal" / "cv_job_worker.py"),
            "--job_file",
            str(job_path),
            "--gpu",
            str(gpu),
            "--out_csv",
            str(result_path),
        ]
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_handle, stem))
        result_paths.append(result_path)
        print(f"[LAUNCH] {stem}: {len(shard)} jobs, pid={process.pid}", flush=True)

    failures = []
    for process, log_handle, stem in processes:
        code = process.wait()
        log_handle.close()
        if code:
            failures.append((stem, code))

    frames = [pd.read_csv(path) for path in result_paths if path.exists()]
    results = pd.concat(frames, ignore_index=True).sort_values(["ablation_name", "fold"])
    results.to_csv(RUN_DIR / "modality_ablation_results.csv", index=False)
    summary = (
        results[results["status"] == "ok"]
        .groupby("ablation_name")
        .agg(
            folds=("fold", "count"),
            auroc_mean=("test_auroc", "mean"),
            auroc_std=("test_auroc", lambda values: values.std(ddof=0)),
            accuracy_mean=("test_acc", "mean"),
            accuracy_std=("test_acc", lambda values: values.std(ddof=0)),
            sensitivity_mean=("test_sens", "mean"),
            sensitivity_std=("test_sens", lambda values: values.std(ddof=0)),
            specificity_mean=("test_spec", "mean"),
            specificity_std=("test_spec", lambda values: values.std(ddof=0)),
        )
        .reset_index()
    )
    summary.to_csv(RUN_DIR / "modality_ablation_summary.csv", index=False)
    print("\n[SUMMARY]", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"[DONE] elapsed={(time.time() - started) / 60:.1f} min", flush=True)
    if failures or len(results) != len(jobs) or (results["status"] != "ok").any():
        raise SystemExit(f"Incomplete run: worker_failures={failures}")


if __name__ == "__main__":
    main()
