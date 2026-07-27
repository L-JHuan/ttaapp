# GNPR Industrial Baseline

This repository is the standalone residual-quantization SID baseline used for
matched industrial comparison with TAP-SID.

## What changes relative to TAP-SID

The data protocol, histories, train/validation/test targets, Llama backbone,
LoRA configuration, constrained beam search, and metrics remain matched.
Only the POI identifier is replaced:

1. category text embedding (64 dimensions);
2. spherical geographic coordinates (3 dimensions);
3. train-period Fourier time statistics (12 dimensions);
4. three-layer residual quantization with codebook sizes 64/64/64;
5. a deterministic collision suffix only for duplicated complete residual paths.

## Reusing TAP preprocessing

For a fair comparison, point `PROCESSED_ROOT` to the completed TAP preprocessing
directory and set `REUSE_PREPROCESSED=1`. Do not construct a second split.

- CSV-scale TAP output: `id_mappings.json` and `v1_sequence/`.
- Spark-scale TAP output: `mappings/`, `metadata/catalog`, `_spark_stages/`,
  and `sequence_parquet/`.

GNPR still needs train-period POI metadata to build its own continuous
representations. CSV-scale runs use the converted `events.csv` and `pois.csv`.
Spark-scale runs reuse TAP's Parquet states and catalog directly.

## Run

Run the following commands from `baselines/gnpr_industrial`. The default
template points `PROCESSED_ROOT` to TAP's shared preprocessing output and writes
all GNPR-specific artifacts to a separate output directory.

```bash
cd baselines/gnpr_industrial
cp configs/industrial.example.env configs/local.env
set -a
source configs/local.env
set +a
export PYTHONPATH=$PWD

# TAP preprocessing already exists; do not rebuild the split.
bash scripts/build_codebook.sh
bash scripts/train.sh
bash scripts/evaluate.sh
```

When the shared TAP preprocessing output does not exist, first run TAP's
top-level preparation entry. The GNPR baseline should not create an independent
split for a matched comparison.

See [docs/INDUSTRIAL_RUN_ZH.md](docs/INDUSTRIAL_RUN_ZH.md) for the detailed
Chinese protocol and artifact list.
