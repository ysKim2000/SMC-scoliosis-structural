"""Run reduced-modality fusion ablations on the final model's exact folds."""

from __future__ import annotations

import json
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
    / "canonical_fusion_cv5_final_indices_512"
)
ORCH_DIR = RUN_DIR / "_orchestration"

ARMS = {
    "pa_clinical": dict(use_pa=True, use_lat=False, use_clinical=True),
    "lat_clinical": dict(use_pa=False, use_lat=True, use_clinical=True),
}
FUSIONS = ["self_attn", "concat", "gated", "projection"]

# One-image-branch models use about 6-7 GB each. Five fit comfortably on the
# empty A6000s; shared GPU 2 receives three workers.
GPU_WORKERS = {0: 5, 1: 5, 2: 3, 3: 5}

BASE_CONFIG = {
    "mode": "train",
    "data_dir": str(ROOT / "data" / "matched_data"),
    "label_csv": str(ROOT / "data" / "AIS_Surgery_clean.csv"),
    "clinical_set": "lumbar_only",
    "img_size": 512,
    "pa_backbone": "resnet101.tv_in1k",
    "lat_backbone": "resnet101.tv_in1k",
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
    folds = canonical_folds()
    for arm, flags in ARMS.items():
        for fusion in FUSIONS:
            name = f"{arm}_{fusion}"
            for fold, (train, test) in enumerate(folds, 1):
                config = dict(BASE_CONFIG)
                config.update(flags)
                config.update(
                    {
                        "fusion_type": fusion,
                        "experiment_root": str(RUN_DIR / name),
                        "experiment_name": f"fold{fold}",
                        "ablation_name": name,
                        "train_indices": train,
                        "test_indices": test,
                    }
                )
                jobs.append(
                    {
                        "tag": f"canonical_{name}_fold{fold}",
                        "ablation_name": name,
                        "fold": fold,
                        "config": config,
                        "keep_checkpoints": True,
                    }
                )
    return jobs


def assign_jobs(jobs: list[dict]) -> list[tuple[int, int, list[dict]]]:
    by_gpu: dict[int, list[dict]] = defaultdict(list)
    for index, job in enumerate(jobs):
        by_gpu[index % 4].append(job)
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
    print(f"[INFO] jobs={len(jobs)}, workers={len(lanes)}", flush=True)

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
    results.to_csv(RUN_DIR / "fusion_ablation_results.csv", index=False)
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
    summary.to_csv(RUN_DIR / "fusion_ablation_summary.csv", index=False)
    print("\n[SUMMARY]", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"[DONE] elapsed={(time.time() - started) / 60:.1f} min", flush=True)
    if failures or len(results) != len(jobs) or (results["status"] != "ok").any():
        raise SystemExit(f"Incomplete run: worker_failures={failures}")


if __name__ == "__main__":
    main()
