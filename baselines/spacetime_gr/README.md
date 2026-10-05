# Spacetime-GR current-protocol adaptation

This directory implements the geographic hierarchical identifier described in
**Spacetime-GR: A Spacetime-Aware Generative Model for Large Scale Online POI
Recommendation** (arXiv:2508.16126) under the matched TAP-SID protocol.

This is a paper-guided reimplementation, not official author code. No official
repository was available when this baseline was prepared.

## Implemented scope

- two-level POI identifier: `geographic block -> inner POI`;
- 5 km by 5 km blocks, as specified by the paper;
- inner IDs assigned from 1 to K within each block;
- the same chronological samples, history text, Llama backbone, LoRA settings,
  constrained beam search, and metrics as TAP-SID.

The paper does not specify the geographic projection or grid origin. This
implementation uses an equirectangular projection with one fixed reference
latitude per dataset and a global longitude/latitude origin. The exact choice is
recorded in `spacetime_gr_report.json`.

Industrial pretraining, action-intent filtering, user-location tokens,
multimodal POI embeddings, curriculum learning, ranking SFT, and DPO are outside
this matched identifier comparison because the required industrial fields and
author implementation are not public.

## Run

Prepare TAP-SID data once, then reuse its `PROCESSED_ROOT`:

```bash
cp baselines/spacetime_gr/configs/example.env baselines/spacetime_gr/configs/local.env
set -a
source baselines/spacetime_gr/configs/local.env
set +a

bash baselines/spacetime_gr/scripts/build_codebook.sh
bash baselines/spacetime_gr/scripts/train.sh
bash baselines/spacetime_gr/scripts/evaluate.sh
```

The three commands write the codebook, SFT data, checkpoint, predictions, and
metrics under `RUN_ROOT` without modifying TAP-SID outputs.
