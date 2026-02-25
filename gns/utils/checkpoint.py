"""
Checkpointing utilities.

Saves:
- model/optim/scheduler states
- step
- RNG states
- config snapshot (dict)

Only rank0 should save in DDP.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def save_checkpoint(
    path: str,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    cfg_dict: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and hasattr(scaler, "state_dict") else None,
        "cfg": cfg_dict,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra_state": extra_state,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(payload["scaler"])

    if "rng_state" in payload:
        try:
            rng = payload["rng_state"]
            if not isinstance(rng, torch.ByteTensor):
                rng = rng.byte().cpu()
            torch.set_rng_state(rng)
        except Exception:
            pass
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        try:
            states = payload["cuda_rng_state"]
            torch.cuda.set_rng_state_all([s.byte().cpu() for s in states])
        except Exception:
            pass

    return payload
