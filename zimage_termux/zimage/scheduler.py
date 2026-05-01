"""Z-Image flow-matching scheduler (numpy).

Z-Image-Turbo uses a shifted-sigmoid timestep schedule. The shift constant
`mu` and the base sigmas are read from the model's `scheduler/config.json`
that ships in the artifact bundle.

The numerical work here is microseconds per step — running on the CPU is
cheaper than the QNN buffer roundtrip would be.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SchedulerConfig:
    num_train_timesteps: int
    shift: float                # base sigmoid shift from training
    use_dynamic_shifting: bool  # Z-Image's resolution-aware shift
    base_image_seq_len: int
    max_image_seq_len: int
    base_shift: float
    max_shift: float

    @classmethod
    def from_dir(cls, path: Path) -> "SchedulerConfig":
        cfg = json.loads((path / "scheduler" / "config.json").read_text())
        return cls(
            num_train_timesteps=int(cfg.get("num_train_timesteps", 1000)),
            shift=float(cfg.get("shift", 1.0)),
            use_dynamic_shifting=bool(cfg.get("use_dynamic_shifting", True)),
            base_image_seq_len=int(cfg.get("base_image_seq_len", 256)),
            max_image_seq_len=int(cfg.get("max_image_seq_len", 4096)),
            base_shift=float(cfg.get("base_shift", 0.5)),
            max_shift=float(cfg.get("max_shift", 1.15)),
        )


def _resolution_mu(cfg: SchedulerConfig, image_seq_len: int) -> float:
    """Linear interpolation between base_shift and max_shift by seq len.

    Mirrors the Tongyi-MAI reference implementation: longer sequences
    (higher resolutions) get a larger mu, biasing the schedule toward more
    work in the high-noise region.
    """
    if not cfg.use_dynamic_shifting:
        return float(np.log(cfg.shift))
    span = cfg.max_image_seq_len - cfg.base_image_seq_len
    if span <= 0:
        return cfg.base_shift
    t = (image_seq_len - cfg.base_image_seq_len) / span
    return float(cfg.base_shift + (cfg.max_shift - cfg.base_shift) * t)


def _shift_sigma(sigma: np.ndarray, mu: float) -> np.ndarray:
    """Apply the shifted-sigmoid transform: sigma' = exp(mu) * sigma /
    (1 + (exp(mu) - 1) * sigma)."""
    e = np.exp(mu)
    return e * sigma / (1.0 + (e - 1.0) * sigma)


class FlowMatchScheduler:
    """8-step (default) flow-matching scheduler for Z-Image-Turbo."""

    def __init__(self, cfg: SchedulerConfig, num_steps: int, image_seq_len: int):
        self.cfg = cfg
        self.num_steps = int(num_steps)
        sigmas = np.linspace(1.0, 0.0, self.num_steps + 1, dtype=np.float32)
        mu = _resolution_mu(cfg, image_seq_len)
        self.sigmas = _shift_sigma(sigmas, mu).astype(np.float32)
        # Timesteps in the same units the DiT was trained on (0..1000):
        self.timesteps = (self.sigmas[:-1] * cfg.num_train_timesteps).astype(np.float32)

    def step(self, model_output: np.ndarray, sample: np.ndarray, i: int) -> np.ndarray:
        """One Euler step of the flow ODE: x_{t+1} = x_t + (sigma_{t+1} - sigma_t) * v."""
        sigma_t = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]
        return sample + (sigma_next - sigma_t) * model_output

    def scale_initial_noise(self, noise: np.ndarray) -> np.ndarray:
        return noise * self.sigmas[0]
