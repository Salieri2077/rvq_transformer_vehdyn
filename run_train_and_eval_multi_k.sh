#!/usr/bin/env bash
set -euo pipefail

# ======================
# Config (edit if needed)
# ======================
PROJECT_ROOT="${PROJECT_ROOT:-/home/an.huang3/VQ-VAE}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_PATH="${DATA_PATH:-/home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy}"
DATA_TYPE="${DATA_TYPE:-pred}"
K_LIST_CSV="${K_LIST_CSV:-15,12,9,6,3}"
IFS=',' read -r -a K_LIST <<< "${K_LIST_CSV}"

# 训练参数
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-500}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"

# 输出根目录（每个 K 的模型都会放在这里）
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/rvq_transformer_vehdyn/work_dirs/tokenizer}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${MODEL_ROOT}/multi_k_${RUN_TAG}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/master.log"

echo "Run root: ${RUN_ROOT}" | tee -a "${MASTER_LOG}"
echo "K list: ${K_LIST[*]}" | tee -a "${MASTER_LOG}"
echo "Data path: ${DATA_PATH}" | tee -a "${MASTER_LOG}"

run_cmd() {
  local step_name="$1"
  shift
  local step_log="${LOG_DIR}/${step_name}.log"

  echo "" | tee -a "${MASTER_LOG}"
  echo "[$(date '+%F %T')] STEP=${step_name}" | tee -a "${MASTER_LOG}"
  echo "CMD: $*" | tee -a "${MASTER_LOG}"

  # shellcheck disable=SC2068
  "$@" 2>&1 | tee "${step_log}"
  local rc=${PIPESTATUS[0]}
  if [[ ${rc} -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAILED ${step_name}, exit=${rc}" | tee -a "${MASTER_LOG}"
    exit ${rc}
  fi
  echo "[$(date '+%F %T')] DONE ${step_name}" | tee -a "${MASTER_LOG}"
}

# 1) Train each K model
MODEL_PATH_MAP=""
for K in "${K_LIST[@]}"; do
  SAVE_DIR_K="${RUN_ROOT}/rvq_tfm_kin_k${K}"
  mkdir -p "${SAVE_DIR_K}"

  run_cmd "train_k${K}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/rvq_transformer_vehdyn/train_tfm.py" \
    --data-path "${DATA_PATH}" \
    --save-dir "${SAVE_DIR_K}" \
    --data-type "${DATA_TYPE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-layers "${K}" \
    --epochs "${EPOCHS}" \
    --max-samples "${MAX_SAMPLES}"

  MODEL_PATH_K="${SAVE_DIR_K}/${DATA_TYPE}_rvq_taae_model.pth"
  if [[ ! -f "${MODEL_PATH_K}" ]]; then
    echo "Missing model checkpoint: ${MODEL_PATH_K}" | tee -a "${MASTER_LOG}"
    exit 1
  fi
  MODEL_PATH_MAP+="${K}:${MODEL_PATH_K},"
done
MODEL_PATH_MAP="${MODEL_PATH_MAP%,}"

echo "MODEL_PATH_MAP=${MODEL_PATH_MAP}" | tee -a "${MASTER_LOG}"

# 2) Evaluate multi-K with model mode
# experiment_token_reduction.py 会从 --save-dir 读取 norm params；
# 这里使用第一个 K 的目录（通常 K=15）作为 norm params 来源。
NORM_SAVE_DIR="${RUN_ROOT}/rvq_tfm_kin_k${K_LIST[0]}"
EVAL_OUT_DIR="${RUN_ROOT}/token_reduction_exp_modelK"

TOKEN_COUNTS_CSV="$(IFS=,; echo "${K_LIST[*]}")"
run_cmd "eval_multi_k" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/rvq_transformer_vehdyn/experiment_token_reduction.py" \
  --reduction-mode model \
  --data-type "${DATA_TYPE}" \
  --token-counts "${TOKEN_COUNTS_CSV}" \
  --data-path "${DATA_PATH}" \
  --save-dir "${NORM_SAVE_DIR}" \
  --model-path-map "${MODEL_PATH_MAP}" \
  --max-samples "${EVAL_MAX_SAMPLES}" \
  --output-dir "${EVAL_OUT_DIR}"

echo "" | tee -a "${MASTER_LOG}"
echo "All done." | tee -a "${MASTER_LOG}"
echo "Overview CSV: ${EVAL_OUT_DIR}/token_reduction_overview.csv" | tee -a "${MASTER_LOG}"
echo "Heatmap compare: ${EVAL_OUT_DIR}/token_reduction_control_heatmap_compare.png" | tee -a "${MASTER_LOG}"
echo "Top1 map compare: ${EVAL_OUT_DIR}/token_reduction_control_top1_map_compare.png" | tee -a "${MASTER_LOG}"
echo "Master log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
