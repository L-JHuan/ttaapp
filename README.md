# TAP-SID

Mainline implementation of **Task-Aligned Prefix Semantic ID (TAP-SID)** for
generative next-POI recommendation.

## Structure

```text
tap_sid/
  convert_tsmc2014.py        # raw eight-column TSV -> event/POI tables
  prepare_realworld_data.py  # raw CSV -> chronological samples
  build_tap_sid.py           # POI metadata -> TAP-SID codebook
  build_llm_data.py          # POI sequences -> SFT JSON
  train_tap_sid.py           # LoRA training
  catalog_trie.py            # catalog prefix constraint
  evaluate_tap_sid.py        # beam evaluation
scripts/
  prepare_data.sh
  train.sh
  evaluate.sh
configs/
  example.env
  foursquare_category_l1_map.json
docs/
  QUICKSTART.md
  QUICKSTART_ZH.md
  DATA_REQUIREMENTS_ZH.md
  DATA_PROTOCOL_ZH.md
baselines/
  gnpr_industrial/           # matched residual-quantization SID baseline
  geogr_full_pipeline/       # complete GeoGR P2P/RQ/EM/CPT/SFT baseline
industrial_city_r5_delivery/
  README_ZH.md               # city-level R@5 analysis and plotting
```

## Input

For industrial behavior logs, set `INDUSTRIAL_JSONL` to a JSONL file containing
`user_id`, `poiid`, `new_key_type`, `longitude`, `latitude`, and `log_time`.
Consecutive reports from the same user at the same POI are treated as one
observed location state before next-POI samples are constructed.
For a train/test-only industrial split, set `NO_VALIDATION=1`, leave
`VALIDATION_END` empty, and use `TRAIN_END` as the inclusive training cutoff.
For a three-way split, set `NO_VALIDATION=0` and provide both time boundaries.
The trainer saves every epoch and selects the checkpoint with the lowest
validation teacher-forcing language-model loss; the test set is never used for
checkpoint selection.

The simplest input is one TSMC2014-compatible eight-column TSV file containing
user ID, POI ID, category ID/name, coordinates, timezone offset, and UTC time.
Set `TSMC_FILE` in the configuration and the pipeline will generate the event
and POI tables automatically. A fixed Foursquare taxonomy maps fine categories
to coarse categories.

The existing two-table input remains available when `TSMC_FILE` is empty:

- `events.csv`: `user_id`, `poi_id`, `timestamp`;
- `pois.csv`: `poi_id`, `latitude`, `longitude`, `category_l1`, `category_l2`.

Start with [`docs/QUICKSTART.md`](docs/QUICKSTART.md), or use the detailed
Chinese guide [`docs/QUICKSTART_ZH.md`](docs/QUICKSTART_ZH.md). The remaining
documents describe the input schema and chronological split protocol.
For an industrial handoff, use
[`docs/INDUSTRIAL_RUN_ZH.md`](docs/INDUSTRIAL_RUN_ZH.md), which includes
configurable multi-GPU sharded evaluation.

## Run

Install dependencies and expose the repository as a Python package:

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

Copy and edit the example configuration:

```bash
cp configs/example.env configs/local.env
set -a
source configs/local.env
set +a
```

For an industrial JSONL run, start from the dedicated template:

```bash
cp configs/industrial.example.env configs/local.env
```

Run the three stages separately:

```bash
bash scripts/prepare_data.sh
bash scripts/train.sh
bash scripts/evaluate.sh
```

## City-level R@5 Analysis

After both TAP-SID and the matched Residual SID baseline have produced their
formal test predictions, use
[`industrial_city_r5_delivery`](industrial_city_r5_delivery) to aggregate
R@5 by the target POI's city and plot the relative improvements for the eight
cities with the largest test sets. The analysis reuses the standard `RUN_ROOT`
and `PROCESSED_ROOT` layouts and performs strict sample, target, and metric
alignment checks before plotting.

See
[`industrial_city_r5_delivery/README_ZH.md`](industrial_city_r5_delivery/README_ZH.md)
for the required `local.env` entries and the one-command workflow.

For industrial inputs at hundreds of millions of rows, use the Spark preparation
entry instead:

```bash
bash scripts/prepare_data_spark.sh
```

It reads JSONL directly, keeps each user's trajectory together during distributed
processing, and writes partitioned Parquet/JSONL data rather than event-level CSV.

Run the stages in this order. Each stage validates its required inputs and
writes to `PROCESSED_ROOT` or `RUN_ROOT`; generated data, checkpoints, logs, and
predictions are excluded from Git.

The training script defaults to one process. To use multiple GPUs, set
`CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` in the execution environment. See
the quick-start guide for configuration examples and expected outputs.

Generated data, checkpoints, logs, and predictions are excluded by `.gitignore`.

## Matched GNPR Baseline

The industrial GNPR residual-SID baseline is provided under
[`baselines/gnpr_industrial`](baselines/gnpr_industrial). It reuses the exact
TAP preprocessing output, time boundaries, histories, train/validation/test
targets, Llama backbone, LoRA settings, and evaluation protocol. Only the POI
identifier construction is replaced by GNPR's residual-quantization codebook.

Run it after TAP preprocessing:

```bash
cd baselines/gnpr_industrial
cp configs/industrial.example.env configs/local.env
# Set CATEGORY_MODEL and other machine-specific paths.
set -a
source configs/local.env
set +a
export PYTHONPATH=$PWD

bash scripts/build_codebook.sh
bash scripts/train.sh
bash scripts/evaluate.sh
```

## Complete GeoGR Baseline

The second industrial baseline is provided under
[`baselines/geogr_full_pipeline`](baselines/geogr_full_pipeline). It implements
the paper-guided P2P, three-level RQ, EM-style SID refinement, CPT, SFT, and
catalog-constrained evaluation workflow. Its industrial entry reuses the exact
TAP-SID Spark/CSV preprocessing output and supports the train/test-only protocol
used in the handoff:

```bash
cd baselines/geogr_full_pipeline
cp configs/industrial.example.env configs/local.env
# 修改 configs/local.env 中的真实路径后直接运行
bash scripts/run_industrial_pipeline.sh
```

如只保留模板文件，也可以直接修改并显式传入：

```bash
bash scripts/run_industrial_pipeline.sh "$PWD/configs/industrial.example.env"
```

This repository therefore exposes exactly three industrial methods: TAP-SID,
Residual SID, and GeoGR full. Other public-benchmark identifier adaptations are
maintained in the separate TAP-SID open-source repository.
