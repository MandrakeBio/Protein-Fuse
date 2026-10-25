from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


ALL_ARMS = {"base_msa", "concat", "balanced_concat", "reconstruction", "predictor_guided"}


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_csv: Path
    msa_npz: Path
    esmc_npz: Path
    protenix_npz: Path
    schemes: tuple[str, ...]
    arms: tuple[str, ...]
    coefficients: list[float]
    reconstruction_alpha: float
    seed: int
    epochs: int
    hidden_width: int
    guide_svr: dict
    svr_grid: dict


def load_config(path: Path) -> ExperimentConfig:
    with Path(path).open() as handle:
        raw = json.load(handle)
    arms = tuple(raw.get("arms", sorted(ALL_ARMS)))
    unknown = sorted(set(arms) - ALL_ARMS)
    if unknown:
        raise ValueError(f"unknown arms in config: {unknown}")
    coefficients = [float(x) for x in raw["coefficients"]]
    if len(coefficients) != 3:
        raise ValueError("coefficients must contain pair_mean, pair_site and ESM-C values")
    return ExperimentConfig(
        dataset_csv=Path(raw["dataset_csv"]),
        msa_npz=Path(raw["msa_npz"]),
        esmc_npz=Path(raw["esmc_npz"]),
        protenix_npz=Path(raw["protenix_npz"]),
        schemes=tuple(raw.get("schemes", ("fold_random_5", "fold_modulo_5", "fold_contiguous_5"))),
        arms=arms,
        coefficients=coefficients,
        reconstruction_alpha=float(raw.get("reconstruction_alpha", 0.5)),
        seed=int(raw.get("seed", 42)),
        epochs=int(raw.get("epochs", 60)),
        hidden_width=int(raw.get("hidden_width", 64)),
        guide_svr=dict(raw["guide_svr"]),
        svr_grid=dict(raw["svr_grid"]),
    )
