#!/usr/bin/env bash
set -e

echo "============================================================"
echo "CAMELS-SIMBA 500U kNN Ablation: k=8 retraining with hidden_dim=64"
echo "Purpose: make k=8 consistent with k=4, k=6, and k=12 h64 runs."
echo "============================================================"

mkdir -p logs/knn_ablation_500u_h64

SEEDS=(42 123 2025)

STATIC_DATASET="data/processed/static_500u_logmass_minmax_top100_periodic_knn/camels_500u_static_logmass_minmax_top100_periodic_knn.pt"
TEMPORAL_DATASET="data/processed/temporal_500u_minmax/camels_500u_temporal_logmass_minmax_top100_periodic_knn.pt"

echo "Static k=8 dataset:   ${STATIC_DATASET}"
echo "Temporal k=8 dataset: ${TEMPORAL_DATASET}"

if [ ! -f "${STATIC_DATASET}" ]; then
  echo "ERROR: Static k=8 dataset not found."
  echo "Checked path: ${STATIC_DATASET}"
  exit 1
fi

if [ ! -f "${TEMPORAL_DATASET}" ]; then
  echo "ERROR: Temporal k=8 dataset not found."
  echo "Checked path: ${TEMPORAL_DATASET}"
  exit 1
fi

# ============================================================
# Static GCN k=8 h64
# ============================================================

for SEED in "${SEEDS[@]}"; do

  EXP_NAME="static_gcn_500u_k8_h64_seed${SEED}"
  METRICS_PATH="experiments/${EXP_NAME}/metrics.json"

  if [ -f "${METRICS_PATH}" ]; then
    echo "Skipping ${EXP_NAME}: metrics.json already exists."
    continue
  fi

  echo "============================================================"
  echo "Training Static GCN | k=8 | hidden_dim=64 | seed=${SEED}"
  echo "Experiment: ${EXP_NAME}"
  echo "============================================================"

  python -m src.training.train_static_gcn \
    --dataset_path "${STATIC_DATASET}" \
    --experiment_name "${EXP_NAME}" \
    --output_root experiments \
    --seed ${SEED} \
    --batch_size 8 \
    --epochs 300 \
    --patience 40 \
    --learning_rate 0.001 \
    --weight_decay 0.00001 \
    --hidden_dim 64 \
    --num_layers 3 \
    --dropout 0.2 \
    --graph_pooling mean \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --grad_clip_norm 1.0 \
    --device auto \
    2>&1 | tee logs/knn_ablation_500u_h64/${EXP_NAME}.log

done

# ============================================================
# EvolveGCN-H k=8 h64
# ============================================================

for SEED in "${SEEDS[@]}"; do

  EXP_NAME="evolvegcn_h_500u_k8_h64_seed${SEED}"
  METRICS_PATH="experiments/${EXP_NAME}/metrics.json"

  if [ -f "${METRICS_PATH}" ]; then
    echo "Skipping ${EXP_NAME}: metrics.json already exists."
    continue
  fi

  echo "============================================================"
  echo "Training EvolveGCN-H | k=8 | hidden_dim=64 | seed=${SEED}"
  echo "Experiment: ${EXP_NAME}"
  echo "============================================================"

  python -m src.training.train_evolvegcn_h \
    --dataset_path "${TEMPORAL_DATASET}" \
    --experiment_name "${EXP_NAME}" \
    --output_root experiments \
    --seed ${SEED} \
    --batch_size 4 \
    --epochs 300 \
    --patience 40 \
    --learning_rate 0.001 \
    --weight_decay 0.00001 \
    --hidden_dim 64 \
    --num_layers 2 \
    --dropout 0.2 \
    --temporal_pooling mean \
    --graph_pooling mean \
    --add_self_loops \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --grad_clip_norm 1.0 \
    --device auto \
    2>&1 | tee logs/knn_ablation_500u_h64/${EXP_NAME}.log

done

echo "============================================================"
echo "k=8 h64 retraining complete."
echo "============================================================"

echo "New k=8 h64 metrics:"
find experiments -path "*500u_k8_h64_seed*/metrics.json" | sort

echo "Count:"
find experiments -path "*500u_k8_h64_seed*/metrics.json" | wc -l
