# GFP representation transfer

Downstream code for GFP single-mutant fitness prediction with feature arrays from a frozen MSA Transformer recipient and frozen ESM-C and Protenix teachers.

The repository implements five feature arms:

| arm | what it does |
| --- | --- |
| `base_msa` | RBF-SVR on the MSA Transformer mutation-site vector |
| `concat` | train-standardized MSA, ESM-C and Protenix blocks joined end-to-end |
| `balanced_concat` | the same concatenation with each block divided by `sqrt(width)` |
| `reconstruction` | MLPs trained to reconstruct MSA embeddings from teacher embeddings |
| `predictor_guided` | MLPs trained through frozen MSA-SVR guides to make fitness-useful corrections |

The runner starts after feature extraction: run frozen MSA Transformer, ESM-C and Protenix encoders on the GFP variant sequences, export the feature arrays as `.npz` files, and pass those files to the downstream experiment.

## Inputs

`configs/gfp_transfer.json` names four input files. The paths are examples; edit them to point at the CSV and embeddings you generate locally:

- a 1,084-row GFP single-mutant CSV with these columns:
  - `variant_id`
  - `mutant`
  - `mutated_sequence`
  - `DMS_score`
  - `fold_random_5`
  - `fold_modulo_5`
  - `fold_contiguous_5`
- `msat.npz`
  - `site`: MSA Transformer site features in CSV row order, shape `N x 768`
  - `variant_ids` or `sequences` for row-order checks
- `esmc600m_features.npz`
  - `site_l12`: ESM-C layer-12 mutation-site features, shape `N x 1152`
  - `variant_ids` in CSV row order
- `protenix_pairformer_features.npz`
  - `sequences`: sequences for each Protenix feature row
  - `pair_mean`: Protenix pair summary, shape `N x 128`
  - `pair_site`: Protenix pair-site summary, shape `N x 128`

ESM-C and MSA features are checked in CSV order. Protenix features are aligned by `mutated_sequence` because the Protenix `.npz` can be stored in a different order.

## Generate the feature files

Produce the arrays with frozen upstream encoders and export them to the `.npz` files named in the config.

For every variant in the CSV:

1. run MSA Transformer and store the layer-12 feature vector at the mutated GFP position;
2. run ESM-C and store the layer-12 feature vector at the mutated GFP position;
3. run Protenix with the standard inference settings you want to use, stop after the pairformer, and summarize the pairformer state into:
   - `pair_mean`: the mean over all residue-pair states;
   - `pair_site`: the mean over pair states touching the mutated residue.

The downstream runner only needs the exported arrays:

```python
import numpy as np

np.savez_compressed(
    "features/msat.npz",
    variant_ids=np.asarray(variant_ids, dtype=str),
    site=msa_site.astype("float32"),          # shape: N x 768
)

np.savez_compressed(
    "features/esmc600m_features.npz",
    variant_ids=np.asarray(variant_ids, dtype=str),
    site_l12=esmc_site_l12.astype("float32"), # shape: N x 1152
)

np.savez_compressed(
    "features/protenix_pairformer_features.npz",
    sequences=np.asarray(sequences, dtype=str),
    pair_mean=protenix_pair_mean.astype("float32"), # shape: N x 128
    pair_site=protenix_pair_site.astype("float32"), # shape: N x 128
)
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Use a CUDA-enabled PyTorch build for the full predictor-guided run.

## Run

Copy the config and fill in the input paths:

```bash
cp configs/gfp_transfer.json configs/local.json
```

Run all three split schemes:

```bash
gfp-transfer run --config configs/local.json --out runs/gfp_transfer --scheme all --device cuda
```

Restrict the run to one split scheme or a subset of feature arms:

```bash
gfp-transfer run --config configs/local.json --out runs/contiguous_transfer --scheme fold_contiguous_5 --arms base_msa balanced_concat predictor_guided --device cuda
```

## Evaluation

For each split scheme, the runner fits and predicts each of five held-out folds. Every scaler, mapper, SVR guide and final regressor is fit using the active outer-training rows only. The headline score is the mean of the pooled Spearman correlations from:

- `fold_random_5`
- `fold_modulo_5`
- `fold_contiguous_5`

Final SVR settings are selected by inner validation on the four training folds for each outer fold. The outer held-out labels are never passed to mapper training or hyperparameter selection.

## Outputs

Each scheme writes:

- `predictions.csv` — held-out predictions with `scheme`, `fold`, `arm`, `variant_id`, `mutant`, `y_true`, and `y_pred`
- `summary.json` — pooled Spearman per arm for that split scheme
- `fold_summaries.json` — per-fold Spearman values
- `mapper_diagnostics.json` — reconstruction and predictor-guided training curves
- `head_selections.json` — SVR hyperparameters selected inside each outer fold
- `complete.json` — SHA-256 hashes for the committed outputs

## Reproducibility choices

- Feature standardizers are fit on the active training rows only.
- The MLP mapper seed defaults to `42`.
- MLP final layers are zero-initialized so predictor-guided transfer starts at the unmodified MSA representation.
- Predictor-guided mappers train through cross-fit frozen SVR guides: a guide never scores a row it was fit on.
- Outer held-out labels are replaced by `NaN` before fitting any mapper.
- Output directories must be new; the runner refuses to overwrite previous results.
