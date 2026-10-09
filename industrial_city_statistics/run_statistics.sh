#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ENV_FILE=${1:-"$SCRIPT_DIR/statistics.local.env"}
if [[ ! -f "$ENV_FILE" ]]; then
  printf '配置不存在：%s\n请先复制 statistics.example.env 并填写真实路径。\n' "$ENV_FILE" >&2
  exit 2
fi
cd "$REPO_ROOT"
set -a
source "$ENV_FILE"
set +a
: "${TAP_RUN_ROOT:?请填写已完成的 TAP-SID 运行目录}"
: "${GNPR_RUN_ROOT:?请填写已完成的 GNPR 运行目录}"
: "${PROCESSED_ROOT:?请填写两种方法共用的已有预处理目录}"
PYTHON_BIN=${PYTHON_BIN:-python}
CITY_STATS_ROOT=${CITY_STATS_ROOT:-"$REPO_ROOT/outputs/city_statistics"}
RUN_ID=$(date +%Y%m%d_%H%M%S)_$$
OUTPUT_DIR="$CITY_STATS_ROOT/$RUN_ID"
mkdir -p "$CITY_STATS_ROOT"
mkdir "$OUTPUT_DIR"
# 运行器日志与统计输出分开放置；统计器独占创建 results，拒绝覆盖旧结果。
LOG="$OUTPUT_DIR/run.log"
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 PYTHONUTF8=1
CITY_ARGS=()
if [[ -n "${POI_CITY_JSONL:-}" ]]; then
  CITY_ARGS+=(--poi-city-jsonl "$POI_CITY_JSONL")
fi
if "$PYTHON_BIN" "$SCRIPT_DIR/analyze_cities.py" \
    --tap-run-root "$TAP_RUN_ROOT" \
    --gnpr-run-root "$GNPR_RUN_ROOT" \
    --processed-root "$PROCESSED_ROOT" \
    --output-dir "$OUTPUT_DIR/results" \
    "${CITY_ARGS[@]}" >"$LOG" 2>&1; then
  printf '统计完成，结果已写入：%s/results\n' "$OUTPUT_DIR"
else
  printf '统计失败，请查看：%s 和 results/analysis.log\n' "$LOG" >&2
  exit 1
fi
