#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${1:-"$REPO_ROOT/configs/local.env"}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "配置文件不存在：$ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

: "${TAP_RUN_ROOT:?请在 local.env 中设置 TAP_RUN_ROOT}"
: "${RESIDUAL_RUN_ROOT:?请在 local.env 中设置 RESIDUAL_RUN_ROOT}"
: "${PROCESSED_ROOT:?请在 local.env 中设置 PROCESSED_ROOT}"
: "${CITY_ANALYSIS_ROOT:?请在 local.env 中设置 CITY_ANALYSIS_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-python}
CITY_TOP_N=${CITY_TOP_N:-8}
CITY_BOOTSTRAP=${CITY_BOOTSTRAP:-2000}
CITY_SEED=${CITY_SEED:-42}
CITY_FIGURE_DPI=${CITY_FIGURE_DPI:-300}
PROVINCE_GEOJSON=${PROVINCE_GEOJSON:-"$SCRIPT_DIR/boundaries/province_full.json"}
CITY_BOUNDARY_DIR=${CITY_BOUNDARY_DIR:-"$SCRIPT_DIR/boundaries/cities"}

cd "$REPO_ROOT"
mkdir -p "$CITY_ANALYSIS_ROOT"

"$PYTHON_BIN" "$SCRIPT_DIR/analyze_city_r5.py" \
  --tap-run-root "$TAP_RUN_ROOT" \
  --residual-run-root "$RESIDUAL_RUN_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --province-geojson "$PROVINCE_GEOJSON" \
  --city-boundary-dir "$CITY_BOUNDARY_DIR" \
  --output-dir "$CITY_ANALYSIS_ROOT" \
  --bootstrap "$CITY_BOOTSTRAP" \
  --seed "$CITY_SEED"

"$PYTHON_BIN" "$SCRIPT_DIR/plot_city_r5.py" \
  --input "$CITY_ANALYSIS_ROOT/city_r5_metrics_all.csv" \
  --output-dir "$CITY_ANALYSIS_ROOT" \
  --top-cities "$CITY_TOP_N" \
  --dpi "$CITY_FIGURE_DPI"

"$PYTHON_BIN" -m json.tool \
  "$CITY_ANALYSIS_ROOT/city_r5_metrics_all.json" >/dev/null
"$PYTHON_BIN" -m json.tool \
  "$CITY_ANALYSIS_ROOT/validation_report.json" >/dev/null

for suffix in png pdf svg; do
  output="$CITY_ANALYSIS_ROOT/top${CITY_TOP_N}_city_r5_relative_improvement.$suffix"
  if [[ ! -s "$output" ]]; then
    echo "缺少绘图产物：$output" >&2
    exit 1
  fi
done

echo "CITY_R5_ANALYSIS_OK"
