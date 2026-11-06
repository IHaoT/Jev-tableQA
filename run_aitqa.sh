#!/usr/bin/env bash
# AIT-QA 本地复现（表格内嵌在 questions.json，无需外部 table 文件）
# 用法: bash run_aitqa.sh [START] [END]     默认 [0,20)
set -euo pipefail
cd "$(dirname "$0")"

if [ -f /Users/mibo/paper/code/ASTRA/.env ]; then
  set -a; source /Users/mibo/paper/code/ASTRA/.env; set +a
fi

START="${1:-0}"
END="${2:-20}"
MODEL="${GRAPHOTTER_MODEL:-Pro/deepseek-ai/DeepSeek-V3}"
BASE_URL="${GRAPHOTTER_BASE_URL:-${SILICONFLOW_BASE_URL:-}}"
API_KEY="${GRAPHOTTER_API_KEY:-${SILICONFLOW_API_KEY:-}}"
EMBEDDER="${GRAPHOTTER_EMBEDDER:-/Users/mibo/paper/code/baseline_data/embedders/gte-base}"
DEPTH="${GRAPHOTTER_MAX_ITERATION_DEPTH:-4}"

BASE_URL="${BASE_URL%/}/"
if [ -z "$API_KEY" ] || [ "$BASE_URL" = "/" ]; then
  echo "缺少 API 凭据：请设置 GRAPHOTTER_API_KEY / GRAPHOTTER_BASE_URL" >&2
  exit 1
fi
if [ ! -d "$EMBEDDER" ]; then
  echo "找不到 gte-base 权重目录: $EMBEDDER" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
mkdir -p embed_cache/ait-qa results
LOG="results/aitqa_${START}-${END}.log"
echo "[run] model=$MODEL  range=[$START,$END)  log=$LOG"

.venv/bin/python -u main.py \
  --model "$MODEL" \
  --base_url "$BASE_URL" \
  --key "$API_KEY" \
  --dataset ait-qa \
  --qa_path dataset/AIT-QA/aitqa_clean_questions.json \
  --embedder_path "$EMBEDDER" \
  --embed_cache_dir embed_cache/ait-qa/ \
  --temperature 0.0 \
  --max_iteration_depth "$DEPTH" \
  --start "$START" --end "$END" \
  --seed 42 \
  --output_path "results/aitqa_${START}-${END}.json" 2>&1 | tee "$LOG"
