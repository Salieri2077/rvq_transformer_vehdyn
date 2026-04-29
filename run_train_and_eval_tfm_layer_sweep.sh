#!/usr/bin/env bash
set -euo pipefail

# ======================
# Config (can override via env)
# ======================
PROJECT_ROOT="${PROJECT_ROOT:-/home/an.huang3/VQ-VAE}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_PATH="${DATA_PATH:-/home/an.huang3/find_bin/work_dirs/dxdydyaw/all_datas_augmented_reverse_detour_directuturn_hs120.npy}"
DATA_TYPE="${DATA_TYPE:-pred}"
TF_LAYERS_CSV="${TF_LAYERS_CSV:-2,3,4,5,6}"
IFS=',' read -r -a TF_LAYERS <<< "${TF_LAYERS_CSV}"

# Train params
NUM_LAYERS="${NUM_LAYERS:-15}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EPOCHS="${EPOCHS:-500}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

# Eval params
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
NUM_VAR_PLOTS="${NUM_VAR_PLOTS:-1}"
NUM_WORST_PLOTS="${NUM_WORST_PLOTS:-1}"

# Output
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/rvq_transformer_vehdyn/work_dirs/tokenizer}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT_DEFAULT="${MODEL_ROOT}/multi_tfm_layers_${RUN_TAG}"
RUN_ROOT="${RUN_ROOT:-${RUN_ROOT_DEFAULT}}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/master.log"

SUMMARY_CSV="${RUN_ROOT}/layer_sweep_summary.csv"
echo "transformer_layers,save_dir,model_path,scenario_eval_dir,scenario_metrics_csv" > "${SUMMARY_CSV}"

echo "Run root: ${RUN_ROOT}" | tee -a "${MASTER_LOG}"
echo "Transformer layers: ${TF_LAYERS[*]}" | tee -a "${MASTER_LOG}"
echo "Data path: ${DATA_PATH}" | tee -a "${MASTER_LOG}"
echo "Data type: ${DATA_TYPE}" | tee -a "${MASTER_LOG}"
echo "Master log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"

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
    exit "${rc}"
  fi
  echo "[$(date '+%F %T')] DONE ${step_name}" | tee -a "${MASTER_LOG}"
}

for TF_LAYER in "${TF_LAYERS[@]}"; do
  SAVE_DIR="${RUN_ROOT}/rvq_tfm_kin_tf${TF_LAYER}"
  mkdir -p "${SAVE_DIR}"

  run_cmd "train_tf${TF_LAYER}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/rvq_transformer_vehdyn/train_tfm.py" \
    --data-path "${DATA_PATH}" \
    --save-dir "${SAVE_DIR}" \
    --data-type "${DATA_TYPE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-layers "${NUM_LAYERS}" \
    --num-transformer-layers "${TF_LAYER}" \
    --epochs "${EPOCHS}" \
    --max-samples "${MAX_SAMPLES}"

  MODEL_PATH="${SAVE_DIR}/${DATA_TYPE}_rvq_taae_model.pth"
  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Missing model checkpoint: ${MODEL_PATH}" | tee -a "${MASTER_LOG}"
    exit 1
  fi

  EVAL_DIR="${SAVE_DIR}/scenario_eval"
  run_cmd "eval_tf${TF_LAYER}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/rvq_transformer_vehdyn/eval_tokenizer_by_scenario.py" \
    --data-path "${DATA_PATH}" \
    --save-dir "${SAVE_DIR}" \
    --data-type "${DATA_TYPE}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --model-type taae \
    --num-var-plots "${NUM_VAR_PLOTS}" \
    --num-worst-plots "${NUM_WORST_PLOTS}" \
    --num-transformer-layers "${TF_LAYER}" \
    --output-dir "${EVAL_DIR}"

  METRICS_CSV="${EVAL_DIR}/scenario_metrics.csv"
  if [[ ! -f "${METRICS_CSV}" ]]; then
    echo "Missing scenario metrics CSV: ${METRICS_CSV}" | tee -a "${MASTER_LOG}"
    exit 1
  fi

  echo "${TF_LAYER},${SAVE_DIR},${MODEL_PATH},${EVAL_DIR},${METRICS_CSV}" >> "${SUMMARY_CSV}"
done

echo "" | tee -a "${MASTER_LOG}"
echo "All done." | tee -a "${MASTER_LOG}"
echo "Summary CSV: ${SUMMARY_CSV}" | tee -a "${MASTER_LOG}"

