from __future__ import annotations

from contextlib import contextmanager
import os

import numpy as np
import torch
from torch import nn


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@contextmanager
def deterministic_torch(device: str):
    old = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(0)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old)


class BranchMLP(nn.Module):
    def __init__(self, in_width: int, out_width: int, hidden_width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, out_width),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiTeacherMapper(nn.Module):
    def __init__(
        self,
        source_widths: list[int],
        out_width: int,
        hidden_width: int,
        coefficients: list[float],
    ):
        super().__init__()
        if len(source_widths) != len(coefficients):
            raise ValueError("one coefficient is required for each source")
        self.coefficients = tuple(float(c) for c in coefficients)
        self.branches = nn.ModuleList(
            BranchMLP(width, out_width, hidden_width) for width in source_widths
        )

    def branch_outputs(self, sources: list[torch.Tensor]) -> list[torch.Tensor]:
        return [branch(source) for branch, source in zip(self.branches, sources)]

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        outputs = self.branch_outputs(sources)
        out = outputs[0] * self.coefficients[0]
        for coefficient, value in zip(self.coefficients[1:], outputs[1:]):
            out = out + coefficient * value
        return out


def clip_gradients(parameters, max_norm: float = 1.0) -> float:
    params = [p for p in parameters if p.grad is not None]
    norm = torch.sqrt(sum(torch.sum(p.grad * p.grad) for p in params))
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("non-finite gradient")
    scale = torch.clamp(torch.as_tensor(max_norm, device=norm.device) / (norm + 1e-6), max=1.0)
    for param in params:
        param.grad.mul_(scale)
    return float(norm.detach().cpu())


def state_hash(module: nn.Module) -> str:
    import hashlib

    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return h.hexdigest()
