"""
Repeated-CV stability check for the Clinical-only and PA+LAT (imaging-only)
ablation arms, to put them on equal methodological footing with the proposed
model's repeated-CV result (0.889 +/- 0.013) for the unified comparison table.

4 new seeds (1,2,3,4) x 5 folds x 2 ablation arms = 40 jobs, run via
cv_job_worker.py across 4 GPUs.

Run in background: python repeated_cv_ablations.py > repeated_cv_ablations.log 2>&1 &
"""
import os
import json
import time
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
os.chdir(PROJECT_ROOT)

N_GPUS = 4
SAVE_DIR = "revision_cv_experiments"
RESULTS_DIR = os.path.join(SAVE_DIR, "_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ARMS = {
    "clinical_only": dict(use_pa=False, use_lat=False, use_clinical=True),
    "pa_lat":        dict(use_pa=True,  use_lat=True,  use_clinical=False),
}

BASE_CONFIG = {
    "mode": "train",
    "data_dir": os.path.join(PROJECT_ROOT, "data/matched_data"),
    "label_csv": os.path.join(PROJECT_ROOT, "data/AIS_Surgery_clean.csv"),
    "clinical_set": "lumbar_only",
    "img_size": 512,
    "pa_backbone": "resnet101.tv_in1k",
    "lat_backbone": "resnet101.tv_in1k",
    "fusion_type": "film",
    "hidden_dim": 256, "dropout_p": 0.4,
    "epochs": 50, "batch_size": 16, "num_workers": 2,
    "lr": 0.0001, "weight_decay": 0.0005, "step_size": 10, "gamma": 0.5,
    "seed": 42, "threshold": 0.5, "patience": 7,
    "use_amp": True, "use_dataparallel": False, "use_cache": True,
    "cache_dir": "cache", "save_images": False, "save_dir": SAVE_DIR,
}


def build_canonical_df():
    df = pd.read_csv("data/AIS_Surgery_clean.csv")
    df["patient_id"] = df["ID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(8)
    pa_dir, lat_dir = "data/matched_data/PA", "data/matched_data/LAT"
    available = (set(f[:8] for f in os.listdir(pa_dir) if f.lower().endswith(".dcm")) &
                 set(f[:8] for f in os.listdir(lat_dir) if f.lower().endswith(".dcm")))
    df = df[df["patient_id"].isin(available)].reset_index(drop=True)
    return df


def run_wave(jobs, wave_name, worker_script):
    if not jobs:
        return pd.DataFrame()
    shards = [jobs[i::N_GPUS] for i in range(N_GPUS)]
    procs, out_csvs = [], []
    for gpu, shard in enumerate(shards):
        if not shard:
            continue
        job_file = os.path.join(RESULTS_DIR, f"{wave_name}_jobs_gpu{gpu}.json")
        out_csv = os.path.join(RESULTS_DIR, f"{wave_name}_results_gpu{gpu}.csv")
        json.dump(shard, open(job_file, "w"))
        out_csvs.append(out_csv)
        log_file = os.path.join(RESULTS_DIR, f"{wave_name}_gpu{gpu}.log")
        cmd = ["python", f"multimodal/{worker_script}",
               "--job_file", job_file, "--gpu", str(gpu), "--out_csv", out_csv]
        print(f"[{wave_name}] launching gpu{gpu}: {len(shard)} jobs", flush=True)
        with open(log_file, "w") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
        procs.append(p)
    for p in procs:
        p.wait()
    print(f"[{wave_name}] all workers finished", flush=True)
    dfs = [pd.read_csv(c) for c in out_csvs if os.path.exists(c)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    t0 = time.time()
    df = build_canonical_df()
    y = df["Structural_L"].values.astype(int)
    n = len(df)
    print(f"[INFO] N = {n}", flush=True)

    jobs = []
    for arm_name, flags in ARMS.items():
        for seed in [1, 2, 3, 4]:
            splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                          .split(np.arange(n), y))
            for f_idx, (tr, te) in enumerate(splits):
                tag = f"repcv_{arm_name}_seed{seed}_fold{f_idx}"
                cfg = dict(BASE_CONFIG)
                cfg.update(flags)
                cfg["ablation_name"] = arm_name
                cfg["experiment_root"] = os.path.join(SAVE_DIR, tag)
                cfg["experiment_name"] = "run"
                cfg["train_indices"] = [int(x) for x in tr]
                cfg["test_indices"] = [int(x) for x in te]
                jobs.append({"tag": tag, "ablation_name": arm_name, "repeat_seed": seed,
                             "fold": f_idx, "config": cfg})

    print(f"[INFO] Total jobs: {len(jobs)}", flush=True)
    results = run_wave(jobs, "repcv_ablations", "cv_job_worker.py")
    results.to_csv(os.path.join(RESULTS_DIR, "repcv_ablations_all.csv"), index=False)

    ok = results[results["status"] == "ok"]
    for arm in ARMS:
        sub = ok[ok["ablation_name"] == arm]
        per_seed = sub.groupby("repeat_seed")["test_auroc"].mean()
        print(f"[RESULT] {arm} per-seed means:\n{per_seed}", flush=True)
        print(f"[RESULT] {arm} repeated-CV: {per_seed.mean():.4f} +/- {per_seed.std():.4f}", flush=True)

    print(f"\n[DONE] elapsed {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
