from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import ExperimentConfig, load_config
from .data import FeatureBlocks, ProteinDataset, load_dataset, load_feature_blocks
from .io import sha256_file, write_csv, write_json
from .methods import (
    Standardizer,
    concatenated_features,
    fit_predictor_guided_transfer,
    fit_reconstruction_transfer,
)
from .svr import SVRParams, fit_svr, predict_svr, select_params, spearman


def final_predictions(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    groups: np.ndarray,
    grid: dict,
    *,
    already_scaled: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    z = x if already_scaled else Standardizer.fit(x[train]).transform(x)
    params, inner = select_params(z, y, train, groups, grid)
    model = fit_svr(z[train], y[train], params)
    pred = predict_svr(model, z[test])
    return pred, {"selected": params.__dict__, "inner": inner}


def arm_features(
    arm: str,
    cfg: ExperimentConfig,
    features: FeatureBlocks,
    y_train_only: np.ndarray,
    train: np.ndarray,
    groups: np.ndarray,
    device: str,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    if arm == "base_msa":
        return features.msa, False, {}
    if arm == "concat":
        return concatenated_features(features.msa, features.teachers, train, balanced=False), True, {}
    if arm == "balanced_concat":
        return concatenated_features(features.msa, features.teachers, train, balanced=True), True, {}
    if arm == "reconstruction":
        x, diag = fit_reconstruction_transfer(
            features.teachers,
            features.msa,
            train,
            alpha=cfg.reconstruction_alpha,
            seed=cfg.seed,
            epochs=cfg.epochs,
            hidden_width=cfg.hidden_width,
            device=device,
        )
        return x, False, diag
    if arm == "predictor_guided":
        guide = SVRParams(
            C=float(cfg.guide_svr["C"]),
            gamma_multiplier=float(cfg.guide_svr["gamma_multiplier"]),
            epsilon=float(cfg.guide_svr["epsilon"]),
        )
        x, diag = fit_predictor_guided_transfer(
            features.teachers,
            features.msa,
            y_train_only,
            train,
            groups,
            coefficients=cfg.coefficients,
            guide_params=guide,
            seed=cfg.seed,
            epochs=cfg.epochs,
            hidden_width=cfg.hidden_width,
            device=device,
        )
        return x, False, diag
    raise ValueError(f"unknown arm {arm}")


def run_scheme(
    cfg: ExperimentConfig,
    data: ProteinDataset,
    features: FeatureBlocks,
    scheme: str,
    arms: tuple[str, ...],
    out: Path,
    device: str
) -> None:
    folds = data.folds(scheme)
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for fold in range(5):
        train = folds != fold
        test = folds == fold
        y_train_only = data.y.copy()
        y_train_only[test] = np.nan

        arm_outputs: dict[str, tuple[np.ndarray, bool]] = {}
        for arm in arms:
            x, already_scaled, diag = arm_features(arm, cfg, features, y_train_only, train, folds, device)
            arm_outputs[arm] = (x, already_scaled)
            if diag:
                diagnostics.append({"scheme": scheme, "fold": fold, "arm": arm, **diag})

        for arm, (x, already_scaled) in arm_outputs.items():
            pred, selection = final_predictions(
                x,
                data.y,
                train,
                test,
                folds,
                cfg.svr_grid,
                already_scaled=already_scaled,
            )
            selections.append({"scheme": scheme, "fold": fold, "arm": arm, **selection})
            score = spearman(data.y[test], pred)
            fold_rows.append({"scheme": scheme, "fold": fold, "arm": arm, "spearman": score})
            for idx, y_pred in zip(np.flatnonzero(test), pred):
                prediction_rows.append({
                    "scheme": scheme,
                    "fold": fold,
                    "arm": arm,
                    "variant_id": data.variant_ids[idx],
                    "mutant": data.mutants[idx],
                    "y_true": float(data.y[idx]),
                    "y_pred": float(y_pred),
                })

    summary = {}
    for arm in arms:
        rows = [row for row in prediction_rows if row["arm"] == arm]
        summary[arm] = {"pooled_spearman": spearman(
            np.asarray([row["y_true"] for row in rows], dtype=float),
            np.asarray([row["y_pred"] for row in rows], dtype=float),
        )}

    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "predictions.csv", prediction_rows)
    write_json(out / "summary.json", summary)
    write_json(out / "fold_summaries.json", fold_rows)
    write_json(out / "head_selections.json", selections)
    write_json(out / "mapper_diagnostics.json", diagnostics)
    complete_files = ["predictions.csv", "summary.json", "fold_summaries.json", "head_selections.json", "mapper_diagnostics.json"]
    write_json(out / "complete.json", {
        "scheme": scheme,
        "arms": list(arms),
        "created_at_unix": time.time(),
        "sha256": {name: sha256_file(out / name) for name in complete_files},
    })


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data = load_dataset(cfg.dataset_csv)
    features = load_feature_blocks(
        msa_npz=cfg.msa_npz,
        esmc_npz=cfg.esmc_npz,
        protenix_npz=cfg.protenix_npz,
        data=data,
    )
    arms = tuple(args.arms) if args.arms else cfg.arms
    schemes = cfg.schemes if args.scheme == "all" else (args.scheme,)
    for scheme in schemes:
        run_scheme(cfg, data, features, scheme, arms, args.out / scheme, args.device)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="protein-fuse")
    sub = p.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--config", type=Path, required=True)
    run_p.add_argument("--out", type=Path, required=True)
    run_p.add_argument("--scheme", default="all")
    run_p.add_argument("--arms", nargs="*")
    run_p.add_argument("--device", default="cuda")
    run_p.set_defaults(func=run)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
