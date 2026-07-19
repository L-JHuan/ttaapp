# Quick Start

## 1. Install

```bash
git clone https://github.com/L-JHuan/ttaapp.git
cd ttaapp
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

## 2. Prepare data

Place the two input files at:

```text
data/raw/events.csv
data/raw/pois.csv
```

Required columns:

- `events.csv`: `user_id`, `poi_id`, `timestamp`
- `pois.csv`: `poi_id`, `latitude`, `longitude`, `category_l1`, `category_l2`

## 3. Configure

```bash
cp configs/example.env configs/local.env
```

Edit `configs/local.env` to set the input paths, chronological split cutoffs,
base-model path, and output directories. Then load it:

```bash
set -a
source configs/local.env
set +a
```

The split rule is:

```text
target_time <= TRAIN_END                         -> train
TRAIN_END < target_time <= VALIDATION_END       -> validation
target_time > VALIDATION_END                    -> test
```

## 4. Run

```bash
bash scripts/prepare_data.sh
bash scripts/train.sh
bash scripts/evaluate.sh
```

For two-GPU training:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
bash scripts/train.sh
```

Predictions and metrics are written to `$RUN_ROOT/eval/`. See
[`QUICKSTART_ZH.md`](QUICKSTART_ZH.md) for detailed configuration, expected
outputs, and troubleshooting.
