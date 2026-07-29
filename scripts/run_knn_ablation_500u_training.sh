#!/usr/bin/env bash
set -e

echo "============================================================"
echo "CAMELS-SIMBA 500U kNN Ablation Training"
echo "Models: Static GCN + EvolveGCN-H"
echo "k values: 4, 6, 12"
echo "Seeds: 42, 123, 2025"
echo "============================================================"

mkdir -p logs/knn_ablation_500u

SEEDS=(42 123 2025)
KS=(4 6 12)

# ============================================================
# Static GCN training
# ============================================================

for K in "${KS[@]}"; do
  for SEED in "${SEEDS[@]}"; do

    EXP_NAME="static_gcn_500u_k${K}_seed${SEED}"
    METRICS_PATH="experiments/${EXP_NAME}/metrics.json"

    if [ -f "${METRICS_PATH}" ]; then
      echo "============================================================"
      echo "Skipping ${EXP_NAME} because metrics.json already exists."
      echo "============================================================"
      continue
    fi

    echo "============================================================"
    echo "Training Static GCN | k=${K} | seed=${SEED}"
    echo "Experiment: ${EXP_NAME}"
    echo "============================================================"

    python -m src.training.train_static_gcn \
      --dataset_path data/processed/static_500u_logmass_minmax_top100_periodic_knn_k${K}/camels_500u_static_logmass_minmax_top100_periodic_knn_k${K}.pt \
      --experiment_name ${EXP_NAME} \
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
      2>&1 | tee logs/knn_ablation_500u/${EXP_NAME}.log

  done
done


# ============================================================
# EvolveGCN-H training
# ============================================================

for K in "${KS[@]}"; do
  for SEED in "${SEEDS[@]}"; do

    EXP_NAME="evolvegcn_h_500u_k${K}_seed${SEED}"
    METRICS_PATH="experiments/${EXP_NAME}/metrics.json"

    if [ -f "${METRICS_PATH}" ]; then
      echo "============================================================"
      echo "Skipping ${EXP_NAME} because metrics.json already exists."
      echo "============================================================"
      continue
    fi

    echo "============================================================"
    echo "Training EvolveGCN-H | k=${K} | seed=${SEED}"
    echo "Experiment: ${EXP_NAME}"
    echo "============================================================"

    python -m src.training.train_evolvegcn_h \
      --dataset_path data/processed/temporal_500u_logmass_minmax_top100_periodic_knn_k${K}/camels_500u_temporal_logmass_minmax_top100_periodic_knn_k${K}.pt \
      --experiment_name ${EXP_NAME} \
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
      2>&1 | tee logs/knn_ablation_500u/${EXP_NAME}.log

  done
done

echo "============================================================"
echo "All kNN ablation training runs completed."
echo "============================================================"

echo "Metrics files found:"
find experiments -path "*500u_k*_seed*/metrics.json" | sort

echo "Total metrics files:"
find experiments -path "*500u_k*_seed*/metrics.json" | wc -l
