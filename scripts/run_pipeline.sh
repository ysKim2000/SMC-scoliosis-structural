#!/usr/bin/env bash
#
# End-to-end driver for the published dual-view FiLM model.
#
# Data must already be in place (see docs/data.md). Checkpoint registry keys in
# evaluate.py / gradcam.py must point at the training run produced below; the
# ablation drivers need FINAL_DIR set to the same run. See docs/reproduction.md.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Preprocessing demo (stage-by-stage figure)"
python src/common/preprocessing_demo.py

echo "==> Training: dual-view ResNet-101 + FiLM, 5-fold CV"
python src/multimodal/run_cv_sweep.py \
    --clinical_sets lumbar_only \
    --fusion_types film \
    --model_names resnet101.tv_in1k \
    --img_size 512 --n_splits 5 --epochs 50 --batch_size 16 \
    --lr 1e-4 --weight_decay 5e-4 --step_size 10 --gamma 0.5 \
    --dropout_p 0.4 --hidden_dim 256 --seed 42 --use_amp --use_cache

echo "==> Evaluation from saved checkpoints"
python src/multimodal/evaluate.py

echo "==> Input-modality ablation (Table 2)"
python src/multimodal/run_modality_ablation.py

echo "==> Fusion-strategy ablation (Table S3)"
python src/multimodal/run_fusion_ablation.py

echo "==> Conventional machine-learning baselines (Table S2)"
python src/ml/ml_baseline.py

echo "==> Subgroup and error analyses (Tables 3 and 4)"
python src/common/subgroup_analysis.py
python src/common/error_analysis.py

echo "==> Grad-CAM"
python src/multimodal/gradcam.py

echo "==> Done."
