from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.stats import spearmanr
from sklearn.svm import SVR


@dataclass(frozen=True)
class SVRParams:
    C: float
    gamma_multiplier: float
    epsilon: float


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(spearmanr(y_true, y_pred).statistic)


def gamma_scale(x: np.ndarray, multiplier: float) -> float:
    variance = float(np.var(x))
    return float(multiplier / (x.shape[1] * variance)) if variance > 0 else float(multiplier)


def fit_svr(x: np.ndarray, y: np.ndarray, params: SVRParams) -> SVR:
    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) or 1.0
    model = SVR(
        C=float(params.C),
        epsilon=float(params.epsilon),
        gamma=gamma_scale(x, params.gamma_multiplier),
    )
    model.fit(x, (y - y_mean) / y_std)
    model._protein_fuse_y_mean = y_mean
    model._protein_fuse_y_std = y_std
    return model


def predict_svr(model: SVR, x: np.ndarray) -> np.ndarray:
    return model.predict(x) * model._protein_fuse_y_std + model._protein_fuse_y_mean


def candidate_params(grid: dict) -> list[SVRParams]:
    return [
        SVRParams(float(c), float(g), float(grid.get("epsilon", 0.02)))
        for c, g in product(grid["C"], grid["gamma_multiplier"])
    ]


def select_params(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    groups: np.ndarray,
    grid: dict,
) -> tuple[SVRParams, list[dict]]:
    candidates = candidate_params(grid)
    inner_groups = np.unique(groups[train])
    rows: list[dict] = []
    best_score = -np.inf
    best = candidates[0]

    for params in candidates:
        pred = np.full(len(y), np.nan, dtype=np.float64)
        for held_group in inner_groups:
            fit_mask = train & (groups != held_group)
            val_mask = train & (groups == held_group)
            model = fit_svr(x[fit_mask], y[fit_mask], params)
            pred[val_mask] = predict_svr(model, x[val_mask])
        score = spearman(y[train], pred[train])
        row = {
            "C": params.C,
            "gamma_multiplier": params.gamma_multiplier,
            "epsilon": params.epsilon,
            "spearman": score,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best = params
    return best, rows
