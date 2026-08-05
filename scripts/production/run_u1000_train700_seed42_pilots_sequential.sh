#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/envs/camels-gnn/bin/activate"

DATASET="data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt"
METADATA="${DATASET%.pt}.metadata.json"
COMPLETE="${DATASET%.pt}.complete"
MANIFEST="configs/splits/u1000_top1000_none_k8_sparse/seed42_train700.json"
IDENTITY="6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a"
EVOLVE_NAME="evolvegcn_h_u1000_top1000_sparse_train700_seed42_none_h32_l2_mean_temporal_mean_linear"
STATIC_NAME="static_gcn_u1000_top1000_sparse_train700_seed42_none_h32_l3_mean_mlp_final"
LOG_DIR="${REPO_ROOT}/logs/u1000_train700_seed42_pilots"
STATE_FILE="${LOG_DIR}/current_state.env"
PREFLIGHT_ONLY=false
[[ "${1:-}" == "--preflight-only" ]] && PREFLIGHT_ONLY=true
[[ $# -le 1 ]] || { echo "FAILURE: usage: $0 [--preflight-only]"; exit 2; }

EVOLVE_CMD=(python -m src.training.train_evolvegcn_h --dataset_path "${DATASET}" --split_manifest_path "${MANIFEST}" --dataset_identity "${IDENTITY}" --experiment_name "${EVOLVE_NAME}" --output_root experiments --seed 42 --batch_size 4 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 0.00001 --hidden_dim 32 --num_layers 2 --dropout 0.2 --activation relu --temporal_pooling mean --graph_pooling mean --head_type linear --add_self_loops --train_ratio 0.7 --val_ratio 0.099 --test_ratio 0.201 --grad_clip_norm 1.0 --device cuda)
STATIC_CMD=(python -m src.training.train_static_gcn --dataset_path "${DATASET}" --dataset_format temporal_final_snapshot --split_manifest_path "${MANIFEST}" --dataset_identity "${IDENTITY}" --experiment_name "${STATIC_NAME}" --output_root experiments --seed 42 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 0.00001 --hidden_dim 32 --num_layers 3 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.7 --val_ratio 0.099 --test_ratio 0.201 --grad_clip_norm 1.0 --device cuda)

fail() { echo "FAILURE: $*" >&2; exit 1; }
hash_is() { [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || fail "hash mismatch: $1"; }

[[ "$(git branch --show-current)" == "thesis-sparse-integrity-hardening" ]] || fail "wrong branch"
git diff --quiet || fail "tracked working tree is not clean"
git diff --cached --quiet || fail "staged working tree is not clean"
[[ -f "${DATASET}" && -f "${METADATA}" && -f "${COMPLETE}" && -f "${MANIFEST}" ]] || fail "required input missing"
hash_is "${DATASET}" "${IDENTITY}"
hash_is "${METADATA}" "d4ea0ba0c3a1abc6f49d6856be86c7fc1226090daac8924eb6b72262d22753b9"
hash_is "${COMPLETE}" "4eea1a4bbbfc57d0c3420a115ae436240e0dcb1588cf47588ab2ee5809edd85a"
hash_is "${MANIFEST}" "b56f35e4cbf1307344beaaf5b26cf181004d23d04fd719b678f1edb9e9924571"
python scripts/validation/validate_u1000_top1000_sparse_dataset.py
python scripts/validation/validate_u1000_train700_seed42_pilots.py --preflight
python -c 'import torch,sys; ok=torch.cuda.is_available() and torch.cuda.get_device_name(0)=="NVIDIA L40"; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable"); sys.exit(0 if ok else 1)' || fail "NVIDIA L40 CUDA gate failed"
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
[[ "${FREE_MIB}" =~ ^[0-9]+$ && "${FREE_MIB}" -ge 40000 ]] || fail "less than 40000 MiB GPU memory free"
CONFLICTS="$(ps -eo pid=,cmd= | awk -v self="$$" '$1 != self && $0 !~ /awk -v self=/ && $0 ~ /(src\.training\.train_|train_(static_gcn|evolvegcn)|build_temporal_sequences|run_u1000_top1000_sparse_build)/ {print}')"
[[ -z "${CONFLICTS}" ]] || fail "conflicting builder/trainer active: ${CONFLICTS}"
for name in "${EVOLVE_NAME}" "${STATIC_NAME}"; do
  [[ ! -e "experiments/${name}/metrics.json" && ! -e "experiments/${name}/checkpoints/best_model.pt" ]] || fail "completed experiment already exists: ${name}"
done

printf 'Evolve command:'; printf ' %q' "${EVOLVE_CMD[@]}"; printf '\n'
printf 'Static command:'; printf ' %q' "${STATIC_CMD[@]}"; printf '\n'
echo "Preflight PASS: branch, clean tracked/staged tree, immutable inputs, dataset, split, CUDA L40, free memory, conflicts, destinations"
${PREFLIGHT_ONLY} && exit 0
mkdir -p "${LOG_DIR}"

sample_gpu() {
  local trainer_pid="$1" output="$2"
  echo "utc,pid,used_gpu_memory_mib,gpu_utilization_percent,free_gpu_memory_mib" > "${output}"
  while kill -0 "${trainer_pid}" 2>/dev/null; do
    local used util free
    used="$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$2); if ($2+0 > peak) peak=$2+0} END {if (peak > 0) print peak}')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    echo "$(date -u +%FT%TZ),${trainer_pid},${used:-},${util:-},${free:-}" >> "${output}"
    sleep 5
  done
}

run_model() {
  local model="$1" exp_name="$2"; shift 2
  local timestamp log_file time_file telemetry_file start end status trainer_pid monitor_pid max_rss
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log_file="${LOG_DIR}/${model}_${timestamp}.log"
  time_file="${LOG_DIR}/${model}_${timestamp}.time"
  telemetry_file="${LOG_DIR}/${model}_gpu_telemetry.csv"
  start="$(date +%s)"
  /usr/bin/time -v -o "${time_file}" "$@" > "${log_file}" 2>&1 &
  trainer_pid=$!
  printf 'CURRENT_MODEL=%q\nTRAINER_PID=%q\nSTART_EPOCH=%q\nLOG_FILE=%q\nEXPERIMENT_DIR=%q\n' \
    "${model}" "${trainer_pid}" "${start}" "${log_file}" "${REPO_ROOT}/experiments/${exp_name}" > "${STATE_FILE}"
  sample_gpu "${trainer_pid}" "${telemetry_file}" & monitor_pid=$!
  set +e
  wait "${trainer_pid}"; status=$?
  set -e
  wait "${monitor_pid}" 2>/dev/null || true
  end="$(date +%s)"
  max_rss="$(awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/,"",$2); print $2}' "${time_file}")"
  python - "${LOG_DIR}/${model}_runtime.json" "$((end-start))" "${max_rss:-null}" "${status}" "${log_file}" <<'PY'
import json,sys
path,runtime,rss,status,log=sys.argv[1:]
json.dump({"runtime_seconds":int(runtime),"max_host_rss_kib":None if rss=="null" else int(rss),"exit_status":int(status),"log_file":log},open(path,"w"),indent=2)
PY
  [[ "${status}" -eq 0 ]] || fail "${model} trainer exited ${status}; log=${log_file}"
  echo "${model} trainer completed successfully; log=${log_file}"
}

run_model evolve "${EVOLVE_NAME}" "${EVOLVE_CMD[@]}"
python scripts/validation/validate_u1000_train700_seed42_pilots.py --model evolve || fail "Evolve artifact validation failed"
run_model static "${STATIC_NAME}" "${STATIC_CMD[@]}"
python scripts/validation/validate_u1000_train700_seed42_pilots.py --model static || fail "Static artifact validation failed"
python scripts/validation/validate_u1000_train700_seed42_pilots.py --finalize || fail "final report validation failed"
printf 'CURRENT_MODEL=complete\nTRAINER_PID=\nSTART_EPOCH=\nLOG_FILE=\nEXPERIMENT_DIR=\n' > "${STATE_FILE}"
echo "SUCCESS: exactly the authorized EvolveGCN-H and Static GCN pilots completed sequentially and validated"
