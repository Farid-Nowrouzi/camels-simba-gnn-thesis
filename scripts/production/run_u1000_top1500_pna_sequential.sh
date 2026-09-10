#!/usr/bin/env bash
set -Eeuo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# -eq 2 ]] || fail "usage: $0 EXPECTED_HEAD EXPECTED_PROTOCOL_SHA256"
expected_head="$1"
expected_protocol_sha="$2"
root="/home/ml/thesis-camels"
python="/home/ml/thesis-camels/envs/camels-gnn/bin/python"
dataset="data/processed/temporal_1000u_none_top1500_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1500_periodic_knn_sparse.pt"
dataset_sha="ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113"
protocol="reports/experiment_registry/u1000_top1500_static_pna_protocol.json"

cd "$root"
[[ "$(git rev-parse HEAD)" == "$expected_head" ]] || fail "wrong PNA preparation HEAD"
[[ "$(sha256sum "$protocol" | awk '{print $1}')" == "$expected_protocol_sha" ]] || fail "protocol SHA mismatch"
[[ "$(sha256sum "$dataset" | awk '{print $1}')" == "$dataset_sha" ]] || fail "dataset SHA mismatch"
"$python" -B scripts/validation/prepare_u1000_top1500_pna_training.py --audit --expected-head "$expected_head"

run_seed() {
  local seed="$1"
  local split="configs/splits/u1000_top1500_none_k8_sparse/seed${seed}_train700.json"
  local degree="reports/experiment_registry/pna_degree_histograms/u1000_top1500_k8_seed${seed}_train700.json"
  local name="static_pna_u1000_top1500_sparse_train700_seed${seed}_none_h32_l3_mean_mlp_final"
  local output="experiments/${name}"
  [[ ! -e "$output" ]] || fail "run collision: $output"
  "$python" -B -m src.training.train_static_pna \
    --dataset_path "$dataset" --dataset_format temporal_final_snapshot \
    --split_manifest_path "$split" --dataset_identity "$dataset_sha" \
    --degree_histogram_path "$degree" --experiment_name "$name" --output_root experiments \
    --seed "$seed" --batch_size 8 --epochs 300 --patience 40 \
    --learning_rate 0.001 --weight_decay 0.00001 --hidden_dim 32 --num_layers 3 \
    --dropout 0.2 --graph_pooling mean --train_ratio 0.7 --val_ratio 0.099 \
    --test_ratio 0.201 --grad_clip_norm 1.0 --device cuda
  for artifact in config.json metrics.json checkpoints/best_model.pt predictions/val_predictions.csv; do
    [[ -s "$output/$artifact" ]] || fail "seed $seed missing required artifact: $artifact"
  done
  [[ -s "$output/predictions/test_predictions.csv" ]] || fail "seed $seed missing trainer-standard test predictions"
  printf 'PASS TRAINED PNA seed=%s output=%s\n' "$seed" "$output"
}

run_seed 42
run_seed 123
run_seed 2025
printf 'THREE-SEED PNA TRAINING COMPLETE\n'
