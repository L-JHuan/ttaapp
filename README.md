# TAP-SID

Mainline implementation of **Task-Aligned Prefix Semantic ID (TAP-SID)** for
generative next-POI recommendation.

## Structure

```text
tap_sid/
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
docs/
  QUICKSTART.md
  QUICKSTART_ZH.md
  DATA_REQUIREMENTS_ZH.md
  DATA_PROTOCOL_ZH.md
```

## Input

Two files are required:

- `events.csv`: `user_id`, `poi_id`, `timestamp`;
- `pois.csv`: `poi_id`, `latitude`, `longitude`, `category_l1`, `category_l2`.

Start with [`docs/QUICKSTART.md`](docs/QUICKSTART.md), or use the detailed
Chinese guide [`docs/QUICKSTART_ZH.md`](docs/QUICKSTART_ZH.md). The remaining
documents describe the input schema and chronological split protocol.

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

Run the three stages separately:

```bash
bash scripts/prepare_data.sh
bash scripts/train.sh
bash scripts/evaluate.sh
```

The training script defaults to one process. To use multiple GPUs, set
`CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE` in the execution environment. See
the quick-start guide for configuration examples and expected outputs.

Generated data, checkpoints, logs, and predictions are excluded by `.gitignore`.
