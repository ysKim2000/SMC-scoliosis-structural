# Reproduction notes

## Before running

1. Place the data as described in [`data.md`](data.md).
2. Run every command from the repository root. The scripts resolve `data/`,
   `cache/` and `experiments_lumbar/` relative to the working directory.
3. The first run builds the preprocessing cache (a few minutes for 155 patients ×
   2 views). Subsequent runs load it from `cache/`.

## Training the proposed model

```bash
python src/multimodal/run_cv_sweep.py \
    --clinical_sets lumbar_only \
    --fusion_types film \
    --model_names resnet101.tv_in1k \
    --img_size 512 \
    --n_splits 5 \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 5e-4 \
    --step_size 10 \
    --gamma 0.5 \
    --dropout_p 0.4 \
    --hidden_dim 256 \
    --seed 42 \
    --use_amp \
    --use_cache
```

Note the defaults: `--batch_size` defaults to 32 and `--use_amp` / `--use_cache` are
off. The published run used **batch size 16** with AMP enabled, as above. The sweep
arguments are lists, so widening them trains the cartesian product with the same
folds and schedule.

Output lands in
`experiments_lumbar/scoliosis_structural/cv5_<timestamp>_512/<clinical>__<fusion>__<backbone>/fold{1..5}/`,
each fold holding `checkpoints/best_auroc.pth`, `config.json`, and
`logs/train_log.csv` plus `logs/best_auroc_test_metrics.json`.

## Evaluation, Grad-CAM, and analyses

```bash
python src/multimodal/evaluate.py            # per-fold metrics, ROC, calibration
python src/multimodal/gradcam.py             # Grad-CAM overlays  (Figure 4)
python src/ml/ml_baseline.py                 # conventional classifiers (Table S2)
python src/common/subgroup_analysis.py       # performance by lumbar Cobb stratum (Table 3)
python src/common/error_analysis.py          # TP/TN/FP/FN characterization (Table 4)
python src/common/film_attribution.py        # FiLM γ/β vs. clinical variables
python src/common/stat_significance.py       # DeLong comparisons between models
python src/common/preprocessing_demo.py      # stage-by-stage preprocessing figure
```

`evaluate.py` and `gradcam.py` select a checkpoint by a registry key near the top of
each file; point it at the run directory produced above before running them.
`subgroup_analysis.py` and `stat_significance.py` read from `results/`, so copy the
per-fold prediction tables there first.

## Ablations

Both ablation drivers rebuild the arms on the **proposed model's exact fold indices**,
read back from each fold's `config.json`, so every arm is scored on identical held-out
patients. Point `FINAL_DIR` at your own run directory first.

```bash
python src/multimodal/run_modality_ablation.py   # Table 2 — six reduced-modality arms
python src/multimodal/run_fusion_ablation.py     # Table S3 — fusion strategies
```

Each driver shards jobs across GPUs via `src/multimodal/cv_job_worker.py`. The
`GPU_WORKERS` dictionary at the top of each file assumes four NVIDIA RTX A6000s;
adjust it for a different machine — the worker counts are sized to 48 GB cards.

The modality ablation uses a separate model class
([`ablation_model.py`](../src/multimodal/ablation_model.py)) that can drop an entire
branch: with no clinical input the FiLM module is bypassed for an image-only head,
and `clinical_only` routes through a dedicated clinical head. This is why those rows
are not simply the proposed model with zeroed inputs.

## What is and is not deterministic

The fold split is fixed by `StratifiedShuffleSplit(n_splits=5, test_size=0.2,
random_state=42)`, and each fold's `train_indices` / `test_indices` are written into
its `config.json`, so a run can be reconstructed exactly without re-deriving the
split. `set_seed(42)` seeds `random`, `numpy`, and `torch`.

cuDNN is **not** put in deterministic mode, so GPU kernel non-determinism still moves
per-fold metrics slightly between runs. Reported figures are the means across the five
folds of a single run; the paper's repeated-CV analysis (seeds 42/1/2/3/4) quantifies
this — it shrinks the headline AUROC from 0.902 to 0.889 ± 0.013, which is the more
honest estimate of the model's stability.

## Cross-validation, not a held-out test set

There is no independent test split. All reported metrics are held-out fold metrics
aggregated across the five folds, and checkpoint selection (highest validation AUROC,
early stopping patience 7) happens on the same fold being scored. For a 155-patient
cohort this is the protocol described in the paper, and it is optimistic relative to a
held-out test set. External validation was not performed — see the paper's Limitations.

## Hardware

Trained on NVIDIA RTX A6000 (48 GB) GPUs. Two ResNet-101 branches at 512 × 512 with
batch size 16 and AMP fit comfortably on one card; the ablation drivers pack several
single-branch jobs onto each GPU.
