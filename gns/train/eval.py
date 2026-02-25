"""
Rollout evaluation metrics.

=== RESEARCH KNOB ===
- task-specific metrics
- horizon-weighting, per-node-type metrics
"""

from __future__ import annotations

import torch


def rollout_position_mse_error(pred_pos: torch.Tensor, true_pos: torch.Tensor) -> torch.Tensor:
    """
    Args:
      pred_pos: [H+1, N, D] or [B, H+1, N, D]
      true_pos: [H+1, N, D] or [B, H+1, N, D]

    Returns:
      err: [B, H+1], where each element is mean over node and space dims.
    """
    if pred_pos.shape != true_pos.shape:
        raise ValueError(f"pred_pos and true_pos must match, got {pred_pos.shape} vs {true_pos.shape}")
    if pred_pos.dim() == 3:
        pred_pos = pred_pos.unsqueeze(0)
        true_pos = true_pos.unsqueeze(0)
    if pred_pos.dim() != 4:
        raise ValueError(f"Expected [B,H+1,N,D] or [H+1,N,D], got {tuple(pred_pos.shape)}")

    return (pred_pos - true_pos).pow(2).mean(dim=(-1, -2))


def rollout_position_mse_from_error(err: torch.Tensor) -> dict[str, float]:
    """
    Args:
      err: [B, H+1]
    """
    if err.dim() != 2:
        raise ValueError(f"err must be [B, H+1], got {tuple(err.shape)}")
    if err.numel() == 0:
        raise ValueError("err is empty")

    err_t = err.mean(dim=0)  # [H+1]
    metrics = {f"pos_mse_t{i}": float(err_t[i].item()) for i in range(err_t.numel())}
    metrics["pos_mse_mean"] = float(err_t.mean().item())
    metrics["num_rollouts"] = float(err.shape[0])
    return metrics


def rollout_position_mse(pred_pos: torch.Tensor, true_pos: torch.Tensor) -> dict[str, float]:
    err = rollout_position_mse_error(pred_pos, true_pos)
    return rollout_position_mse_from_error(err)
