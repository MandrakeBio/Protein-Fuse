from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from sklearn.svm import SVR

from .mlp import MultiTeacherMapper, clip_gradients, deterministic_torch, state_hash
from .svr import SVRParams, gamma_scale


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        return cls(mean, scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.mean


class FrozenRBFSVR(nn.Module):
    def __init__(self, model: SVR):
        super().__init__()
        self.gamma = float(model._gamma)
        self.register_buffer(
            "support_vectors", torch.as_tensor(model.support_vectors_, dtype=torch.float32)
        )
        self.register_buffer(
            "dual_coef", torch.as_tensor(model.dual_coef_.ravel(), dtype=torch.float32)
        )
        self.register_buffer(
            "intercept", torch.as_tensor(model.intercept_, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x[:, None, :] - self.support_vectors[None, :, :]
        kernel = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))
        return kernel @ self.dual_coef + self.intercept[0]


def standardized_blocks(
    native: np.ndarray,
    teachers: tuple[np.ndarray, ...],
    train: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    native_z = Standardizer.fit(native[train]).transform(native)
    teacher_z = [Standardizer.fit(block[train]).transform(block) for block in teachers]
    return native_z, teacher_z


def concatenated_features(
    native: np.ndarray,
    teachers: tuple[np.ndarray, ...],
    train: np.ndarray,
    *,
    balanced: bool,
) -> np.ndarray:
    blocks = list(standardized_blocks(native, teachers, train))
    values = [blocks[0], *blocks[1]]
    if balanced:
        values = [value / np.sqrt(value.shape[1]) for value in values]
    return np.concatenate(values, axis=1)


def fit_reconstruction_transfer(
    teachers: tuple[np.ndarray, ...],
    native: np.ndarray,
    train: np.ndarray,
    *,
    alpha: float,
    seed: int,
    epochs: int,
    hidden_width: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    native_scaler = Standardizer.fit(native[train])
    native_z = native_scaler.transform(native)
    teacher_scalers = [Standardizer.fit(block[train]) for block in teachers]
    teacher_z = [scaler.transform(block) for scaler, block in zip(teacher_scalers, teachers)]

    torch.manual_seed(seed)
    mapper = MultiTeacherMapper(
        [block.shape[1] for block in teachers],
        native.shape[1],
        hidden_width,
        [1.0 / len(teachers)] * len(teachers),
    ).to(device)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=1e-3, weight_decay=0.01)
    source_t = [torch.as_tensor(block[train], dtype=torch.float32, device=device) for block in teacher_z]
    target_t = torch.as_tensor(native_z[train], dtype=torch.float32, device=device)
    curve: list[float] = []

    with deterministic_torch(device):
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            prediction = mapper(source_t)
            loss = (prediction - target_t).square().mean()
            loss.backward()
            clip_gradients(mapper.parameters())
            optimizer.step()
            curve.append(float(loss.detach().cpu()))

    with torch.inference_mode():
        all_t = [torch.as_tensor(block, dtype=torch.float32, device=device) for block in teacher_z]
        mapped_z = mapper(all_t).double().cpu().numpy()
    blended_z = (1.0 - float(alpha)) * native_z + float(alpha) * mapped_z
    return native_scaler.inverse_transform(blended_z), {
        "loss": curve,
        "state_hash": state_hash(mapper),
        "alpha": float(alpha),
    }


def fit_guides(
    native_z: np.ndarray,
    y_z: np.ndarray,
    train: np.ndarray,
    groups: np.ndarray,
    params: SVRParams,
    device: str,
) -> dict[int, FrozenRBFSVR]:
    guides: dict[int, FrozenRBFSVR] = {}
    for group in np.unique(groups[train]):
        fit = train & (groups != group)
        model = SVR(
            C=params.C,
            epsilon=params.epsilon,
            gamma=gamma_scale(native_z[fit], params.gamma_multiplier),
        )
        model.fit(native_z[fit], y_z[fit])
        guides[int(group)] = FrozenRBFSVR(model).to(device)
    return guides


def fit_predictor_guided_transfer(
    teachers: tuple[np.ndarray, ...],
    native: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    groups: np.ndarray,
    *,
    coefficients: list[float],
    guide_params: SVRParams,
    seed: int,
    epochs: int,
    hidden_width: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    native_scaler = Standardizer.fit(native[train])
    native_z = native_scaler.transform(native)
    teacher_scalers = [Standardizer.fit(block[train]) for block in teachers]
    teacher_z = [scaler.transform(block) for scaler, block in zip(teacher_scalers, teachers)]
    y_mean = float(np.mean(y[train]))
    y_std = float(np.std(y[train])) or 1.0
    y_z = (y - y_mean) / y_std

    torch.manual_seed(seed)
    mapper = MultiTeacherMapper(
        [block.shape[1] for block in teachers],
        native.shape[1],
        hidden_width,
        coefficients,
    ).to(device)
    guides = fit_guides(native_z, y_z, train, groups, guide_params, device)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=1e-3, weight_decay=0.3)

    source_t = [torch.as_tensor(block[train], dtype=torch.float32, device=device) for block in teacher_z]
    native_t = torch.as_tensor(native_z[train], dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_z[train], dtype=torch.float32, device=device)
    group_t = groups[train]
    curve: list[float] = []

    with deterministic_torch(device):
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            z = native_t + mapper(source_t)
            pred = torch.empty_like(y_t)
            for group, guide in guides.items():
                mask_np = group_t == group
                if np.any(mask_np):
                    mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
                    pred[mask] = guide(z[mask])
            loss = (pred - y_t).square().mean()
            loss.backward()
            clip_gradients(mapper.parameters())
            optimizer.step()
            curve.append(float(loss.detach().cpu()))

    with torch.inference_mode():
        all_t = [torch.as_tensor(block, dtype=torch.float32, device=device) for block in teacher_z]
        delta_z = mapper(all_t).double().cpu().numpy()
    transferred = native_scaler.inverse_transform(native_z + delta_z)
    return transferred, {"loss": curve, "state_hash": state_hash(mapper)}
