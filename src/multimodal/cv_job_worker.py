"""
Worker process for the canonical ablation runs (uses ablation_train.run_train).
(the use_pa/use_lat/use_clinical-flag based training entrypoint used for the
Clinical-only and PA+LAT modality ablations), needed for their repeated-CV
stability checks in the unified comparison table.

Run: python cv_job_worker.py --job_file jobs_gpu0.json --gpu 0 --out_csv results_gpu0.csv
"""
import os
import sys
import json
import shutil
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--job_file", required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--out_csv", required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402
from ablation_train import run_train  # noqa: E402

jobs = json.load(open(args.job_file))
print(f"[worker gpu={args.gpu}] {len(jobs)} jobs to run", flush=True)

rows = []
for i, job in enumerate(jobs):
    tag = job["tag"]
    config = job["config"]
    print(f"\n[worker gpu={args.gpu}] ({i+1}/{len(jobs)}) {tag}", flush=True)
    row = {"tag": tag, "gpu": args.gpu, **{k: job.get(k) for k in
           ("ablation_name", "repeat_seed", "fold")}}
    try:
        exp_dir = run_train(config)
        summary = json.load(open(os.path.join(exp_dir, "logs", "best_auroc_summary.json")))
        best = summary["Best_AUROC"]
        row.update({
            "status": "ok",
            "n_train": len(config["train_indices"]),
            "n_test": len(config["test_indices"]),
            "test_auroc": best["test_auroc"],
            "test_acc": best["test_acc"],
            "test_sens": best["test_sens"],
            "test_spec": best["test_spec"],
            "best_epoch": best["epoch"],
        })
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        if os.path.isdir(ckpt_dir) and not job.get("keep_checkpoints", False):
            shutil.rmtree(ckpt_dir)
    except Exception as e:
        row.update({"status": "failed", "error": str(e)})
        print(f"[worker gpu={args.gpu}] FAILED {tag}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)

    rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

print(f"[worker gpu={args.gpu}] done, {len(rows)} results -> {args.out_csv}", flush=True)
