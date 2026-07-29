#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: no active virtualenv or conda environment detected." >&2
  echo "Activate the repository's Python environment, then rerun this script." >&2
  exit 2
fi

command -v python3 >/dev/null 2>&1 || {
  echo "ERROR: python3 is not available in the active environment." >&2
  exit 2
}

if [[ -x /usr/bin/time ]]; then
  TIME_PREFIX=(/usr/bin/time -v)
else
  TIME_PREFIX=()
  echo "WARNING: /usr/bin/time unavailable; continuing without GNU resource statistics."
fi

python3 -c 'import torch; import src.training.train_evolvegcn_h' || {
  echo "ERROR: torch or src.training.train_evolvegcn_h failed to import." >&2
  exit 2
}

LOG_ROOT="${REPO_ROOT}/logs/canonical_evolvegcn_scaling"
mkdir -p "${LOG_ROOT}"

run_one() {
  local experiment_name="$1"
  local dataset_path="$2"
  local seed="$3"
  local experiment_dir="${REPO_ROOT}/experiments/${experiment_name}"
  local log_path="${LOG_ROOT}/${experiment_name}.log"
  local -a required=(
    "config.json"
    "metrics.json"
    "train_log.csv"
    "predictions/test_predictions.csv"
    "checkpoints/best_model.pt"
  )
  local complete=1
  local artifact

  [[ -f "${REPO_ROOT}/${dataset_path}" ]] || {
    echo "ERROR: required dataset does not exist: ${REPO_ROOT}/${dataset_path}" >&2
    exit 3
  }

  if [[ -e "${experiment_dir}" ]]; then
    for artifact in "${required[@]}"; do
      if [[ ! -f "${experiment_dir}/${artifact}" ]]; then
        complete=0
      fi
    done
    if [[ "${complete}" -eq 1 ]]; then
      echo "SKIP: complete planned replacement already exists: ${experiment_name}"
      return 0
    fi
    echo "ERROR: partial planned experiment folder exists: ${experiment_dir}" >&2
    echo "No files were changed. Inspect it manually before retrying." >&2
    exit 4
  fi

  local -a command=(
    python3 -m src.training.train_evolvegcn_h
    --dataset_path "${dataset_path}"
    --experiment_name "${experiment_name}"
    --output_root experiments
    --seed "${seed}"
    --batch_size 4
    --epochs 300
    --patience 40
    --learning_rate 0.001
    --weight_decay 0.00001
    --hidden_dim 32
    --num_layers 2
    --dropout 0.2
    --activation relu
    --temporal_pooling mean
    --graph_pooling mean
    --head_type mlp
    --add_self_loops
    --train_ratio 0.70
    --val_ratio 0.15
    --test_ratio 0.15
    --grad_clip_norm 1.0
    --device auto
  )

  {
    echo
    echo "Experiment: ${experiment_name}"
    echo "Start UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Disk before:"
    df -h "${REPO_ROOT}"
    echo "RAM before:"
    free -h
    printf 'Command:'
    printf ' %q' "${TIME_PREFIX[@]}"
    printf ' %q' "${command[@]}"
    printf '\n'
  } 2>&1 | tee "${log_path}"

  set +e
  "${TIME_PREFIX[@]}" "${command[@]}" 2>&1 | tee -a "${log_path}"
  local run_status="${PIPESTATUS[0]}"
  set -e

  {
    echo "Finish UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Exit status: ${run_status}"
    echo "Disk after:"
    df -h "${REPO_ROOT}"
    echo "RAM after:"
    free -h
  } 2>&1 | tee -a "${log_path}"

  if [[ "${run_status}" -ne 0 ]]; then
    echo "ERROR: training failed for ${experiment_name}; stopping sequential runner." >&2
    exit "${run_status}"
  fi

  for artifact in "${required[@]}"; do
    [[ -f "${experiment_dir}/${artifact}" ]] || {
      echo "ERROR: run exited successfully but required artifact is missing: ${artifact}" >&2
      exit 5
    }
  done
}

run_one "evolvegcn_h_u20_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed42" \
  "data/processed/temporal_20u_minmax/camels_20u_temporal_logmass_minmax_top100_periodic_knn.pt" 42
run_one "evolvegcn_h_u20_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed2025" \
  "data/processed/temporal_20u_minmax/camels_20u_temporal_logmass_minmax_top100_periodic_knn.pt" 2025
run_one "evolvegcn_h_u50_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed42" \
  "data/processed/temporal_50u_minmax/camels_50u_temporal_logmass_minmax_top100_periodic_knn.pt" 42
run_one "evolvegcn_h_u50_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed2025" \
  "data/processed/temporal_50u_minmax/camels_50u_temporal_logmass_minmax_top100_periodic_knn.pt" 2025
run_one "evolvegcn_h_u100_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed42" \
  "data/processed/temporal_100u_minmax/camels_100u_temporal_logmass_minmax_top100_periodic_knn.pt" 42
run_one "evolvegcn_h_u100_top100_norm-minmax_h32_l2_gpool-mean_tpool-mean_head-mlp_canonical300_seed2025" \
  "data/processed/temporal_100u_minmax/camels_100u_temporal_logmass_minmax_top100_periodic_knn.pt" 2025
