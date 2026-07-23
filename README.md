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
```

## Input

For industrial behavior logs, set `INDUSTRIAL_JSONL` to a JSONL file containing
`user_id`, `poiid`, `new_key_type`, `longitude`, `latitude`, and `log_time`.
Consecutive reports from the same user at the same POI are treated as one
observed location state before next-POI samples are constructed.
For a train/test-only industrial split, set `NO_VALIDATION=1`, leave
`VALIDATION_END` empty, and use `TRAIN_END` as the inclusive training cutoff.

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

Run the stages in this order. Each stage validates its required inputs and
writes to `PROCESSED_ROOT` or `RUN_ROOT`; generated data, checkpoints, logs, and
predictions are excluded from Git.

The training script defaults to one process. To use multiple GPUs, set
`CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` in the execution environment. See
the quick-start guide for configuration examples and expected outputs.

Generated data, checkpoints, logs, and predictions are excluded by `.gitignore`.
