#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/storage/yuchen/conda_envs/approx_triton_py312/bin/python}"
OUT_DIR="${OUT_DIR:-/storage/yuchen/approx_sglang_pruning_runs/e2e_rtn_mlp_verify_$(date +%Y%m%d_%H%M%S)}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
ARTIFACT="${APPROX_SGLANG_PRUNE_ARTIFACT:-/storage/yuchen/approx_sglang_pruning_runs/artifacts/rtn_mlp_3b_all_s20.json}"
PLUGIN="${TRITON_PASS_PLUGIN_PATH:-/storage/yuchen/approx_triton_work/build/approx-triton-plugin-faechlo/lib/libApproxTritonPlugin.so}"
SOURCE_TRITON="${APPROX_SOURCE_TRITON_PYTHON:-/storage/yuchen/approx_triton_work/src/triton/python}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OUT_DIR
export MODEL_PATH
export APPROX_SGLANG_PRUNE_ARTIFACT="${ARTIFACT}"
export APPROX_SOURCE_TRITON_PYTHON="${SOURCE_TRITON}"
export TRITON_PASS_PLUGIN_PATH="${PLUGIN}"
export TRITON_PLUGIN_PATHS="${TRITON_PLUGIN_PATHS:-${TRITON_PASS_PLUGIN_PATH}}"

export BATCH_SIZE="${BATCH_SIZE:-1}"
export LONG_PROMPT_REPEAT="${LONG_PROMPT_REPEAT:-256}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
export WARMUP_RUNS="${WARMUP_RUNS:-2}"
export MEASURE_RUNS="${MEASURE_RUNS:-10}"
export ACCURACY_MAX_NEW_TOKENS="${ACCURACY_MAX_NEW_TOKENS:-256}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.80}"

if [[ ! -f "${ARTIFACT}" ]]; then
  echo "Missing pruning artifact: ${ARTIFACT}" >&2
  echo "Build one first with build_rtn_mlp_artifact.py, or set APPROX_SGLANG_PRUNE_ARTIFACT." >&2
  exit 2
fi

echo "[rtn-mlp-e2e] out_dir=${OUT_DIR}"
echo "[rtn-mlp-e2e] model=${MODEL_PATH}"
echo "[rtn-mlp-e2e] artifact=${ARTIFACT}"
echo "[rtn-mlp-e2e] source_triton=${APPROX_SOURCE_TRITON_PYTHON}"
echo "[rtn-mlp-e2e] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/sweep_sglang_pruning.py"
