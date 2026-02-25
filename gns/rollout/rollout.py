"""
Rollout loop.

This is where:
- graph building (neighbor search),
- feature construction,
- model prediction,
- and the integrator
meet in a tight loop.

Design goals:
- compact and readable
- GPU-first (keeps tensors on device; no accidental CPU sync)
- stable semantics for long horizons (predict -> integrate; caller controls grads)

=== RESEARCH KNOBS ===
- Teacher forcing vs open-loop (requires providing ground truth; add it here).
- Graph rebuild frequency (every step vs cached edges).
- Noise injection (state noise, target noise, denoising objectives).
- Integrator conventions (predict accel vs delta-vel vs next-state).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from gns.graphs.base import GraphBuilder
from gns.rollout.integrators import Integrator
from gns.typing import GraphBatch


@dataclass(frozen=True)
class RolloutResult:
    pos: torch.Tensor  # [H+1, N, D]
    vel: torch.Tensor  # [H+1, N, D]


def _maybe_move_to_device(x: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.device == device:
        return x
    return x.to(device)


def rollout(
    *,
    model: torch.nn.Module,
    graph_builder: GraphBuilder,
    integrator: Integrator,
    x_builder,  # callable(pos, vel, node_type?) -> x
    pos0: torch.Tensor,
    vel0: torch.Tensor,
    dt: float,
    horizon: int,
    batch_index: Optional[torch.Tensor] = None,
    cache_edges: bool = False,
    edge_rebuild_interval: int = 1,
) -> RolloutResult:
    """
    Open-loop rollout for `horizon` steps.

    Args:
      pos0, vel0: [N, D] (or flattened across batch if using batch_index)
      batch_index: [N] long graph id per node (optional). If provided, graph building
                   should not connect across different graph ids.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    if pos0.shape != vel0.shape:
        raise ValueError(f"pos0 and vel0 must have same shape, got {pos0.shape} vs {vel0.shape}")
    if edge_rebuild_interval < 0:
        raise ValueError(f"edge_rebuild_interval must be >= 0, got {edge_rebuild_interval}")

    # Keep everything on the model's device by default (GPU-first ergonomics).
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = pos0.device

    pos = _maybe_move_to_device(pos0, device)
    vel = _maybe_move_to_device(vel0, device)
    batch_index = _maybe_move_to_device(batch_index, device)
    assert pos is not None and vel is not None

    # Preserve caller's mode (important if rollout is used inside training code).
    was_training = model.training
    model.eval()

    debug_finite = os.getenv("GNS_DEBUG_FINITE", "0").strip() == "1"

    # Two paths:
    # - If grads enabled, store as a list and stack (keeps autograd graph intact).
    # - If no grads, preallocate for speed.
    track_grad = torch.is_grad_enabled()
    if track_grad:
        pos_list = [pos]
        vel_list = [vel]
    else:
        pos_out = pos.new_empty((horizon + 1,) + pos.shape)
        vel_out = vel.new_empty((horizon + 1,) + vel.shape)
        pos_out[0] = pos
        vel_out[0] = vel

    try:
        cached_edge_index: Optional[torch.Tensor] = None
        cached_edge_attr: Optional[torch.Tensor] = None

        for t in range(horizon):
            if cache_edges:
                should_rebuild = cached_edge_index is None
                if edge_rebuild_interval > 0 and (t % edge_rebuild_interval == 0):
                    should_rebuild = True
                if should_rebuild:
                    cached_edge_index, cached_edge_attr = graph_builder.build(pos, batch=batch_index)
                edge_index = cached_edge_index
                edge_attr = cached_edge_attr
                assert edge_index is not None
            else:
                # Neighbor search / edges
                edge_index, edge_attr = graph_builder.build(pos, batch=batch_index)

            # Features (user-supplied; keep it simple)
            x = x_builder(pos, vel, None)  # === RESEARCH KNOB === feature construction

            g = GraphBatch(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch_index,
            )

            accel = model(g)

            if debug_finite:
                if not torch.isfinite(accel).all():
                    raise FloatingPointError("Non-finite accel encountered during rollout.")

            pos, vel = integrator.step(pos, vel, accel, dt)

            if track_grad:
                pos_list.append(pos)
                vel_list.append(vel)
            else:
                pos_out[t + 1] = pos
                vel_out[t + 1] = vel
    finally:
        # Restore mode
        model.train(was_training)

    if track_grad:
        return RolloutResult(pos=torch.stack(pos_list, dim=0), vel=torch.stack(vel_list, dim=0))
    return RolloutResult(pos=pos_out, vel=vel_out)
