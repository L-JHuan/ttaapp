# GeoGR-SID current-protocol adaptation

This directory implements the publicly specified SID-construction stages from
**GeoGR: Enabling Spatio-Temporal Aware Industrial-scale Generative POI
Recommendations** (arXiv:2602.10411) under the matched TAP-SID protocol.

This is a paper-guided reimplementation, not official author code. No official
repository was available when this baseline was prepared.

## Implemented scope

- training-only user/POI sets reconstructed from the chronological train split;
- Swing-scored co-visited POI pairs filtered by geographic distance;
- POI-to-POI contrastive learning with in-batch negatives;
- three-layer residual K-means tokenization;
- deterministic collision leaf for a unique catalog mapping;
- the same downstream Llama backbone, LoRA settings, SFT samples, constrained
  beam search, and metrics as TAP-SID.

The public NYC/TKY files do not contain GeoGR's industrial name, brand, address,
consumption, active-period, weather, query, or action-type fields. The available
description therefore contains category names and WGS84 coordinates. The paper
fine-tunes its Qwen encoder; for a controlled and tractable identifier comparison,
this implementation trains a linear contrastive adapter over fixed encoder
embeddings. A precomputed embedding matrix can be supplied through
`GEOGR_INITIAL_EMBEDDINGS_NPZ`.

The paper's EM-style section does not publish the collision-free assignment
procedure, iteration count, optimization corpus, or stopping rule. It is not
silently approximated here. Every report explicitly records this boundary, so
results must be named **GeoGR-SID current-protocol adaptation**, not official
GeoGR.

For the matched extension that additionally includes EM-style SID refinement,
four-template CPT, and SFT, see
[`../geogr_full_pipeline`](../geogr_full_pipeline). Keeping the two directories
separate makes it possible to distinguish identifier-only effects from the
additional training pipeline.

## Run

```bash
cp baselines/geogr/configs/example.env baselines/geogr/configs/local.env
set -a
source baselines/geogr/configs/local.env
set +a

bash baselines/geogr/scripts/build_codebook.sh
bash baselines/geogr/scripts/train.sh
bash baselines/geogr/scripts/evaluate.sh
```

Use `GEOGR_CODEBOOK_SIZE=32` for NYC and `64` for TKY, matching the paper.
Generated artifacts are isolated under `RUN_ROOT`.
