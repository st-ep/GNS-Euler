"""
Objectives and loss functions.

=== RESEARCH KNOB ===
- one-step vs multi-step rollout loss
- robust losses (Huber/Charbonnier)
- auxiliary losses (constraints, regularizers)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    kind: str = "mse"
    huber_delta: float = 1.0
    rollout_weight: float = 0.0
    rollout_horizon: int = 0


def one_step_loss(pred: torch.Tensor, target: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    if cfg.kind == "mse":
        return F.mse_loss(pred, target)
    if cfg.kind == "huber":
        return F.huber_loss(pred, target, delta=cfg.huber_delta)
    raise ValueError(f"Unknown loss kind: {cfg.kind}")


def rollout_position_loss(
    pred_pos: torch.Tensor, target_pos: torch.Tensor, cfg: LossConfig
) -> torch.Tensor:
    """
    Rollout loss helper. Uses the same robust loss family as one-step loss.
    """
    return one_step_loss(pred_pos, target_pos, cfg)
