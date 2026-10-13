from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {
    "variant_id",
    "mutant",
    "mutated_sequence",
    "DMS_score",
    "fold_random_5",
    "fold_modulo_5",
    "fold_contiguous_5",
}


@dataclass(frozen=True)
class GFPDataset:
    rows: list[dict[str, str]]
    variant_ids: np.ndarray
    mutants: np.ndarray
    sequences: np.ndarray
    y: np.ndarray

    def folds(self, scheme: str) -> np.ndarray:
        if scheme not in REQUIRED_COLUMNS:
            raise ValueError(f"unknown split scheme {scheme!r}")
        folds = np.asarray([int(row[scheme]) for row in self.rows], dtype=np.int64)
        if folds.shape != self.y.shape or not np.all((folds >= 0) & (folds < 5)):
            raise ValueError(f"{scheme} must contain fold IDs 0..4 for every row")
        return folds


@dataclass(frozen=True)
class FeatureBlocks:
    msa: np.ndarray
    esmc: np.ndarray
    protenix_pair_mean: np.ndarray
    protenix_pair_site: np.ndarray

    @property
    def teachers(self) -> tuple[np.ndarray, ...]:
        # Same order as the published coefficients: pair_mean, pair_site, ESM-C.
        return (self.protenix_pair_mean, self.protenix_pair_site, self.esmc)

    @property
    def teacher_names(self) -> tuple[str, ...]:
        return ("protenix_pair_mean", "protenix_pair_site", "esmc_l12")


def load_dataset(path: Path) -> GFPDataset:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")

    variant_ids = np.asarray([row["variant_id"] for row in rows], dtype=str)
    sequences = np.asarray([row["mutated_sequence"] for row in rows], dtype=str)
    mutants = np.asarray([row["mutant"] for row in rows], dtype=str)
    if len(set(variant_ids)) != len(variant_ids):
        raise ValueError("variant_id values must be unique")
    if len(set(sequences)) != len(sequences):
        raise ValueError("mutated_sequence values must be unique")

    y = np.asarray([float(row["DMS_score"]) for row in rows], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("DMS_score must be finite for every row")
    return GFPDataset(rows, variant_ids, mutants, sequences, y)


def _as_str(handle: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in handle:
        raise ValueError(f"npz is missing {key!r}")
    return np.asarray(handle[key]).astype(str)


def _matrix(handle: np.lib.npyio.NpzFile, key: str, rows: int, width: int) -> np.ndarray:
    if key not in handle:
        raise ValueError(f"npz is missing {key!r}")
    value = np.asarray(handle[key], dtype=np.float64)
    if value.shape != (rows, width) or not np.isfinite(value).all():
        raise ValueError(f"{key} expected shape {(rows, width)}, got {value.shape}")
    return value


def load_feature_blocks(
    *,
    msa_npz: Path,
    esmc_npz: Path,
    protenix_npz: Path,
    data: GFPDataset,
) -> FeatureBlocks:
    n = len(data.y)

    with np.load(msa_npz, allow_pickle=False) as handle:
        if "variant_ids" in handle:
            if not np.array_equal(_as_str(handle, "variant_ids"), data.variant_ids):
                raise ValueError("MSA variant_ids do not match the dataset order")
        elif "sequences" in handle:
            if not np.array_equal(_as_str(handle, "sequences"), data.sequences):
                raise ValueError("MSA sequences do not match the dataset order")
        else:
            raise ValueError("MSA npz needs variant_ids or sequences for alignment checks")
        msa = _matrix(handle, "site", n, 768)

    with np.load(esmc_npz, allow_pickle=False) as handle:
        if not np.array_equal(_as_str(handle, "variant_ids"), data.variant_ids):
            raise ValueError("ESM-C variant_ids do not match the dataset order")
        esmc = _matrix(handle, "site_l12", n, 1152)

    with np.load(protenix_npz, allow_pickle=False) as handle:
        sequences = _as_str(handle, "sequences")
        if len(set(sequences)) != len(sequences):
            raise ValueError("Protenix sequences must be unique")
        index = {sequence: i for i, sequence in enumerate(sequences)}
        try:
            order = np.asarray([index[sequence] for sequence in data.sequences], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"Protenix features are missing sequence {exc.args[0]!r}") from exc
        pair_mean = _matrix(handle, "pair_mean", len(sequences), 128)[order]
        pair_site = _matrix(handle, "pair_site", len(sequences), 128)[order]

    return FeatureBlocks(msa=msa, esmc=esmc, protenix_pair_mean=pair_mean, protenix_pair_site=pair_site)
