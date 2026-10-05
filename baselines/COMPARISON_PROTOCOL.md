# Geographic hierarchical SID comparison protocol

## Purpose

The comparison isolates whether TAP-SID's gains come merely from adding an
explicit geographic hierarchy. Every method uses the same chronological split,
closed train-seen catalog, history text, Llama backbone, LoRA hyperparameters,
training samples, catalog-constrained beam search, and evaluation metrics.
Only the POI identifier is replaced.

## Methods and permitted names

- **TAP-SID**: task-aligned `coarse -> fine -> L1 -> L2 -> leaf` identifier.
- **Spacetime-GR hierarchical-ID adaptation**: paper-defined 5 km geographic
  block followed by an inner-POI index. It is not the full industrial
  Spacetime-GR training system.
- **GeoGR-SID current-protocol adaptation**: training-only geographically
  constrained co-visitation pairs, contrastive representation adaptation, and
  three-layer RQ-Kmeans. This identifier-only variant isolates the SID
  construction stages that can be implemented directly from the paper.
- **GeoGR matched full-pipeline adaptation**: extends the identifier-only
  variant with LLM-driven EM-style SID refinement, four-template CPT, and
  matched SFT. It reports both `GeoGR-EM + SFT` and
  `GeoGR-EM + CPT + SFT`.
- **Residual SID**: the matched GNPR residual-quantization identifier already
  provided in `baselines/gnpr_industrial`.

Do not label any of these adaptations as an official reproduction.

## Required checks before training

1. All methods must cover exactly the same POI catalog.
2. Complete SID paths must be unique.
3. GeoGR pairs and representations must use the training split only.
4. Generated train/validation/test JSON files must have identical sample counts
   and ordering across methods.
5. Backbone, LoRA rank, effective batch size, epochs, cutoff length, beam size,
   catalog constraints, and random seeds must be recorded.

## Result aggregation

After all formal evaluations finish:

```bash
python -m baselines.compare_geo_hierarchical_results \
  --result TAP-SID=/path/to/tap/eval/test_metrics.json \
  --result "Spacetime-GR adapted=/path/to/spacetime/eval/test_metrics.json" \
  --result "GeoGR-SID adapted=/path/to/geogr/eval/test_metrics.json" \
  --result "GeoGR-EM+SFT=/path/to/geogr_full/sft_only/eval/test_metrics.json" \
  --result "GeoGR-EM+CPT+SFT=/path/to/geogr_full/cpt_sft/eval/test_metrics.json" \
  --result "Residual SID=/path/to/residual/eval/test_metrics.json" \
  --reference TAP-SID \
  --output_csv /path/to/comparison.csv \
  --output_json /path/to/comparison.json
```

The reported deltas are `method - reference`; negative values therefore mean
that the method is below TAP-SID when TAP-SID is selected as the reference.
