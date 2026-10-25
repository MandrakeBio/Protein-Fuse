# Protein Fuse

**Connecting sequence, structure, and evolution through latent translation.**

Protein Fuse combines complementary representations from frozen protein models for supervised fitness prediction. Lightweight translators map features from ESM-C and Protenix into the latent space of an MSA Transformer base model. The repository also includes single-model and concatenation baselines for comparison.

The initial study uses 1,084 GFP single mutants. The downstream pipeline accepts additional rows with the same input schema; extensions to other proteins or model combinations require suitable feature extraction and evaluation. Performance on larger datasets has not been established by this release.

The repository implements five comparison methods:

| Method (`--arms`) | Description |
| --- | --- |
| `base_msa` | RBF-SVR on the MSA Transformer mutation-site vector |
| `concat` | Training-standardized MSA, ESM-C and Protenix blocks joined end-to-end |
| `balanced_concat` | The same concatenation with each block divided by `sqrt(width)` |
| `reconstruction` | MLPs trained to reconstruct MSA embeddings from teacher embeddings |
| `predictor_guided` | MLPs trained through frozen MSA-SVR guides to produce fitness-relevant residual corrections |

**The assay CSV, precomputed embeddings, model weights, MSAs, and upstream extraction scripts are not included.** The runner starts after feature extraction. Use the upstream implementations to generate embeddings, export them as `.npz` files, and configure the downstream experiment to load those files. Installing Protein Fuse does not install or run the upstream encoders.

## Inputs

`configs/protein_fuse.json` names four input files. The paths are examples; edit them to point at the CSV and embeddings you generate locally:

- a CSV with one row per variant (1,084 rows for the initial GFP benchmark) and these columns:
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

The loader requires unique `variant_id` and `mutated_sequence` values, finite numeric `DMS_score` values, and all three fold columns. Fold IDs must be integers from 0 to 4; populate all five folds in each evaluated scheme. The runner does not generate splits or infer mutation positions.

For the original experiment, use the Sarkisyan avGFP single-mutant subset and corresponding split assignments from [ProteinGym](https://github.com/OATML-Markslab/ProteinGym). Preserve those assignments when comparing with the original benchmark. New datasets require explicitly designed splits.

## Generate the feature files

### Set up the upstream encoders

Follow the official installation and checkpoint instructions:

| Encoder | Upstream implementation | Required output |
| --- | --- | --- |
| MSA Transformer | [FAIR ESM](https://github.com/facebookresearch/esm) | Layer-12 mutation-site vector, width 768 |
| ESM-C 600M | [ESM](https://github.com/evolutionaryscale/esm) | Layer-12 mutation-site vector, width 1152 |
| Protenix | [Protenix](https://github.com/bytedance/Protenix) | Pairformer pair representations with channel width 128 |

Use separate environments when upstream dependencies conflict. FAIR ESM and ESM-C both use the Python import name `esm`; exchange their outputs through feature files.

Keep weights frozen, use evaluation mode, and disable gradient tracking. Extract features without using fitness labels. Record checkpoint identifiers, upstream commit versions, extraction layers, token mappings, seeds, MSA settings, and inference settings alongside the exported files.

**Reproducibility scope:** this release specifies the downstream feature contract, but does not pin the original extraction code, checkpoints, MSA construction, or pair-pooling implementation. These instructions produce compatible inputs; matching shapes alone does not guarantee reproduction of the reported scores.

### MSA Transformer

1. Prepare a homologous-sequence alignment for the target protein using the upstream MSA workflow. Record its source, filtering, depth, and subsampling policy.
2. For each single-mutant variant, use the variant as the query row in the aligned context. Preserve the mapping from ungapped residue positions to alignment columns.
3. Run the frozen model and obtain layer-12 representations. The upstream `esm_msa1b_t12_100M_UR50S` is a compatible 12-layer MSA Transformer.
4. Select the query-row vector at the mutated residue. Account for alignment gaps and tokenizer special tokens; a residue number is not necessarily a tensor index.
5. Stack the vectors in CSV row order to form `msa_site` with shape `(N, 768)`.

### ESM-C

1. Load the ESM-C 600M checkpoint using its upstream implementation.
2. Encode each full variant sequence and capture the layer-12 hidden state using an intermediate-layer output or a forward hook. A default final-layer embedding is not a substitute.
3. Select the mutation-site vector, excluding padding and special tokens. Record the layer-indexing and normalization conventions used by the extractor.
4. Stack vectors in CSV row order to form `esmc_site_l12` with shape `(N, 1152)`.

### Protenix

1. Prepare each variant as a protein input using the upstream preprocessing and inference instructions. Use a consistent checkpoint, MSA/template policy, and recycling settings.
2. Instrument the upstream forward pass to capture the pair representation after the pairformer at the chosen final recycle. An extraction adapter may return at this point, before coordinate generation; standard structure-output files are not these embeddings.
3. Select valid residue-pair states for the target protein using its token-to-residue mapping. Exclude padding and unrelated chains or tokens. Verify channel width 128; other widths require updating the downstream loader.
4. Compute `pair_mean`, an average over the selected residue-pair states, and `pair_site`, an average over selected pair states involving the mutated residue. Each is a 128-dimensional vector.
5. Stack the summaries and save the full variant sequence associated with each row. The loader realigns these features by sequence.

Pair states are directional: the mutation-site row and column need not be equal. Document whether `pair_site` pools a row, a column, or their union, and how the diagonal is handled. The original repository does not resolve that convention. Choose one consistently for a new extraction; obtain the original extraction settings before claiming exact replication.

### Export the arrays

The following is an export template **after** extraction. Its arrays come from your upstream extraction code; Protein Fuse does not create them. `variant_ids` follows CSV order, while `protenix_sequences` follows the order of the Protenix arrays.

The downstream runner only needs the exported arrays:

```python
from pathlib import Path
import numpy as np

Path("features").mkdir(parents=True, exist_ok=True)

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
    sequences=np.asarray(protenix_sequences, dtype=str),
    pair_mean=protenix_pair_mean.astype("float32"), # shape: N x 128
    pair_site=protenix_pair_site.astype("float32"), # shape: N x 128
)
```

All matrices must contain finite values. Save IDs and sequences as NumPy string arrays, not object arrays: the loader uses `allow_pickle=False`. MSA and ESM-C rows must match CSV order exactly. Protenix sequences must be unique and cover every dataset sequence; additional Protenix rows are allowed if each matrix matches its sequence array.

## Install

Use Python 3.10 or newer. Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
protein-fuse --help
```

Use a CUDA-enabled PyTorch build for GPU runs, or select `--device cpu` for CPU execution. Upstream embedding generation has separate dependency and hardware requirements.

The distribution and CLI are now `protein-fuse`, and the Python package is `protein_fuse`. These replace `gfp-representation-transfer`, `gfp-transfer`, and `gfp_transfer`. Reinstall from this checkout and update existing commands and imports when migrating. The configuration template is now `configs/protein_fuse.json`.

## Run

Copy the config and fill in the input paths:

```bash
cp configs/protein_fuse.json configs/local.json
```

Edit the following fields in `configs/local.json`, preserving the remaining settings:

```json
{
  "dataset_csv": "data/gfp_benchmark_v1/singles.csv",
  "msa_npz": "features/msat.npz",
  "esmc_npz": "features/esmc600m_features.npz",
  "protenix_npz": "features/protenix_pairformer_features.npz"
}
```

Relative paths are resolved from the directory where you run the command, **not** from the configuration file's directory. Run from the repository root or use absolute paths.

Validate the inputs using the same loaders used by the experiment:

```bash
python - <<'PYVALIDATE'
from pathlib import Path
from protein_fuse.config import load_config
from protein_fuse.data import load_dataset, load_feature_blocks

cfg = load_config(Path("configs/local.json"))
data = load_dataset(cfg.dataset_csv)
features = load_feature_blocks(
    msa_npz=cfg.msa_npz,
    esmc_npz=cfg.esmc_npz,
    protenix_npz=cfg.protenix_npz,
    data=data,
)
for scheme in cfg.schemes:
    if set(data.folds(scheme).tolist()) != set(range(5)):
        raise ValueError(f"{scheme} must populate all five folds")
print(f"Validated {len(data.y)} variants")
print("Feature shapes:", features.msa.shape, features.esmc.shape,
      features.protenix_pair_mean.shape, features.protenix_pair_site.shape)
PYVALIDATE
```

Run all three split schemes:

```bash
protein-fuse run --config configs/local.json --out runs/protein_fuse --scheme all --device cuda
```

Restrict the run to one split scheme or a subset of feature arms:

```bash
protein-fuse run --config configs/local.json --out runs/contiguous_fuse --scheme fold_contiguous_5 --arms base_msa balanced_concat predictor_guided --device cuda
```

Replace `--device cuda` with `--device cpu` to run on a CPU. Choose a fresh output directory for every run.

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

The runner writes per-scheme summaries. To obtain the study's headline metric, average the three per-scheme pooled Spearman correlations for each method; no separate cross-scheme aggregate file is written.

## Reproducibility choices

- Feature standardizers are fit on the active training rows only.
- The MLP mapper seed defaults to `42`.
- MLP final layers are zero-initialized so predictor-guided transfer starts at the unmodified MSA representation.
- Predictor-guided mappers train through cross-fit frozen SVR guides: a guide never scores a row it was fit on.
- Outer held-out labels are replaced by `NaN` before fitting any mapper.
- Output directories must be new; the runner refuses to overwrite previous results.


## Extend Protein Fuse

The row count is inferred from the CSV and is not hard-coded to 1,084. For a larger dataset with compatible single-site features, generate new embeddings, supply the required columns and split assignments, and update the input paths.

- **Other proteins:** construct suitable MSAs, map mutation sites correctly, and choose splits that match the intended generalization question. IDs and sequences must remain unique.
- **Multiple substitutions or indels:** define a feature-selection or pooling rule first. The current contract expects one site vector per variant and does not define how to combine multiple mutation sites.
- **Different models or feature widths:** update the width checks in `src/protein_fuse/data.py`. Changing teacher blocks also requires updating `FeatureBlocks`, coefficient validation in `config.py`, and the experiment configuration. Coefficients are ordered as Protenix `pair_mean`, Protenix `pair_site`, then ESM-C.
- **Larger runs:** the runner loads feature arrays into memory, uses full-batch mapper training, and repeatedly fits kernel SVRs. Larger datasets may require batching, memory-efficient storage, or a different prediction head.

Validate extensions independently. The initial GFP result does not establish performance across proteins or dataset sizes, and additional data does not guarantee improved performance.
