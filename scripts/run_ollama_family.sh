#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_ollama_family.sh <family> [stage]

Families:
  qwen | deepseek | gemma | gptoss | all

Stages:
  smoke | core | heldout | all

Examples:
  scripts/run_ollama_family.sh qwen
  scripts/run_ollama_family.sh deepseek smoke
  KEEP_ALIVE=30m scripts/run_ollama_family.sh gemma all

Environment overrides:
  OLLAMA_HOST          Default: http://127.0.0.1:11434
  KEEP_ALIVE           Default: 15m
  OLLAMA_THINK         Default: off
  TASKS                Default: task3_startup_week,task6_incident_response_week,task7_quarterly_headcount_plan
  SMOKE_SEEDS          Default: 100:3
  FULL_SEEDS           Default: 100:10
  OUT_ROOT             Default: outputs/ollama-families
  DEFAULT_NUM_CTX      Default: 8192
  BIG_MODEL_NUM_CTX    Default: 4096
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

FAMILY="${1:-}"
STAGE="${2:-all}"

if [[ -z "$FAMILY" ]]; then
  usage
  exit 1
fi

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
KEEP_ALIVE="${KEEP_ALIVE:-15m}"
OLLAMA_THINK="${OLLAMA_THINK:-off}"
TASKS="${TASKS:-task3_startup_week,task6_incident_response_week,task7_quarterly_headcount_plan}"
SMOKE_SEEDS="${SMOKE_SEEDS:-100:3}"
FULL_SEEDS="${FULL_SEEDS:-100:10}"
OUT_ROOT="${OUT_ROOT:-outputs/ollama-families}"
DEFAULT_NUM_CTX="${DEFAULT_NUM_CTX:-8192}"
BIG_MODEL_NUM_CTX="${BIG_MODEL_NUM_CTX:-4096}"

case "$FAMILY" in
  qwen)
    MODELS=("qwen3.5:2b" "qwen3.5:9b" "qwen3.5:27b")
    ;;
  deepseek)
    MODELS=("deepseek-r1:8b" "deepseek-r1:14b")
    ;;
  gemma)
    MODELS=("gemma4:e4b" "gemma4:26b")
    ;;
  gptoss)
    MODELS=("gpt-oss:20b")
    ;;
  all)
    MODELS=(
      "qwen3.5:2b"
      "qwen3.5:9b"
      "qwen3.5:27b"
      "deepseek-r1:8b"
      "deepseek-r1:14b"
      "gemma4:e4b"
      "gemma4:26b"
      "gpt-oss:20b"
    )
    ;;
  *)
    echo "Unknown family: $FAMILY" >&2
    usage
    exit 1
    ;;
esac

case "$STAGE" in
  smoke|core|heldout|all)
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    usage
    exit 1
    ;;
esac

if ! curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null; then
  echo "Ollama is not reachable at ${OLLAMA_HOST}." >&2
  echo "Start or expose ollama serve before running this script." >&2
  exit 2
fi

run_stage() {
  local model="$1"
  local split="$2"
  local seeds="$3"
  local output_dir="$4"
  local num_ctx="$5"
  local cmd=(
    uv run baseline
    --provider ollama
    --ollama-host "$OLLAMA_HOST"
    --ollama-keep-alive "$KEEP_ALIVE"
    --ollama-think "$OLLAMA_THINK"
    --policy model
    --model "$model"
    --tasks "$TASKS"
    --seeds "$seeds"
    --scenario-split "$split"
    --paper-eval
    --output-dir "$output_dir"
  )
  if [[ -n "$num_ctx" ]]; then
    cmd+=(--ollama-num-ctx "$num_ctx")
  fi
  echo
  echo "[run] model=${model} split=${split} seeds=${seeds} output=${output_dir}"
  "${cmd[@]}"
}

best_num_ctx() {
  local model="$1"
  case "$model" in
    qwen3.5:27b|gemma4:26b)
      printf '%s' "$BIG_MODEL_NUM_CTX"
      ;;
    *)
      printf '%s' "$DEFAULT_NUM_CTX"
      ;;
  esac
}

print_summary() {
  local summary_path="$1"
  python - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(0)
summary = json.loads(path.read_text(encoding="utf-8"))
overall = summary.get("overall_mean_score")
strict_native = summary.get("strict_native_overall_mean_score")
episodes = summary.get("episode_count")
run_complete = summary.get("run_complete")
print(f"[summary] path={path}")
print(f"[summary] overall_mean_score={overall} strict_native_overall_mean_score={strict_native} episode_count={episodes} run_complete={run_complete}")
PY
}

for model in "${MODELS[@]}"; do
  safe_name="${model//:/-}"
  num_ctx="$(best_num_ctx "$model")"

  echo
  echo "============================================================"
  echo "[pull] ${model}"
  ollama pull "$model"

  if [[ "$STAGE" == "smoke" || "$STAGE" == "all" ]]; then
    smoke_dir="${OUT_ROOT}/${safe_name}/smoke-core"
    run_stage "$model" "core" "$SMOKE_SEEDS" "$smoke_dir" "$num_ctx"
    print_summary "${smoke_dir}/summary.json"
  fi

  if [[ "$STAGE" == "core" || "$STAGE" == "all" ]]; then
    core_dir="${OUT_ROOT}/${safe_name}/core"
    run_stage "$model" "core" "$FULL_SEEDS" "$core_dir" "$num_ctx"
    print_summary "${core_dir}/summary.json"
  fi

  if [[ "$STAGE" == "heldout" || "$STAGE" == "all" ]]; then
    heldout_dir="${OUT_ROOT}/${safe_name}/heldout"
    run_stage "$model" "heldout" "$FULL_SEEDS" "$heldout_dir" "$num_ctx"
    print_summary "${heldout_dir}/summary.json"
  fi

  echo "[cleanup] unloading ${model}"
  ollama stop "$model" >/dev/null 2>&1 || true
done

echo
echo "[done] family=${FAMILY} stage=${STAGE} outputs=${OUT_ROOT}"
