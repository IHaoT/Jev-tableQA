#!/usr/bin/env bash
# HiTab 本地复现
# 用法: bash run_hitab.sh [START] [END] [INDEX_FILE]     默认 [0,20)
#   给了 INDEX_FILE 时忽略 START/END，按文件里的下标选题。
#
# 可用环境变量覆盖:
#   GRAPHOTTER_MODEL / GRAPHOTTER_BASE_URL / GRAPHOTTER_API_KEY
#   GRAPHOTTER_EMBEDDER / GRAPHOTTER_MAX_ITERATION_DEPTH
set -euo pipefail
cd "$(dirname "$0")"

# 复用 ASTRA 项目里已有的 API 凭据；外部 export 的 GRAPHOTTER_* 优先。
if [ -f /Users/mibo/paper/code/ASTRA/.env ]; then
  set -a; source /Users/mibo/paper/code/ASTRA/.env; set +a
fi

START="${1:-0}"
END="${2:-20}"
INDEX_FILE="${3:-}"
RUN_TAG="${START}-${END}"
if [ -n "$INDEX_FILE" ]; then
  RUN_TAG="$(basename "$INDEX_FILE" .json)"
fi
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
mkdir -p embed_cache/hitab results
LOG="results/hitab_${RUN_TAG}.log"
echo "[run] model=$MODEL  range=[$START,$END)  index=${INDEX_FILE:-none}  log=$LOG"

.venv/bin/python -u main.py \
  --model "$MODEL" \
  --base_url "$BASE_URL" \
  --key "$API_KEY" \
  --dataset hitab \
  --qa_path dataset/hitab/test_samples.jsonl \
  --table_folder dataset/hitab/raw/ \
  --embedder_path "$EMBEDDER" \
  --embed_cache_dir embed_cache/hitab/ \
  --temperature 0.0 \
  --max_iteration_depth "$DEPTH" \
  --start "$START" --end "$END" \
  --index_file "$INDEX_FILE" \
  --seed 42 \
  --output_path "results/hitab_${RUN_TAG}.json" 2>&1 | tee "$LOG"
