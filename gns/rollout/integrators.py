"""
Integrator interface + a stable default.

In this repo, the model typically predicts acceleration (or an acceleration-like quantity),
and the integrator defines how we advance (pos, vel).

=== RESEARCH KNOBS ===
- Interpret the model output differently (delta-vel, delta-pos, next-pos, etc.).
- Add damping, constraints, or projection steps for stability.
- Replace with higher-order schemes (requires additional model evaluations).

Note:
- We keep the interface extremely small: integrators only see (pos, vel, accel, dt).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Integrator(ABC):
    @abstractmethod
    def step(
        self,
        pos: torch.Tensor,  # [N, D]
        vel: torch.Tensor,  # [N, S]
        accel: torch.Tensor,  # [N, S] (or "delta-vel" depending on convention)
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance one step.

        Keep this tiny: higher-order methods that require multiple accel evaluations
        should live in the rollout loop (where the model is available), not here.
        """
        raise NotImplementedError


class SemiImplicitEuler(Integrator):
    """
    Semi-implicit (symplectic) Euler:
      v_{t+1} = v_t + a_t * dt
      x_{t+1} = x_t + v_{t+1} * dt

    This is a good default for long-horizon stability compared to explicit Euler.

    === RESEARCH KNOB ===
    - clamp accel or vel here for stability (careful: can hide model bugs)
    - apply damping: vel_next = (1-gamma)*vel_next
    """

    def step(
        self, pos: torch.Tensor, vel: torch.Tensor, accel: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vel.shape != accel.shape:
            raise ValueError(f"vel and accel must match, got {vel.shape} vs {accel.shape}")
        # Static-position mode: useful for field rollouts on fixed grids.
        if pos.shape[-1] != vel.shape[-1]:
            vel_next = vel + accel * dt
            return pos, vel_next
        vel_next = vel + accel * dt
        pos_next = pos + vel_next * dt
        return pos_next, vel_next


class ExplicitEuler(Integrator):
    """
    Explicit Euler (less stable, but sometimes useful as a baseline):
      x_{t+1} = x_t + v_t * dt
      v_{t+1} = v_t + a_t * dt
    """

    def step(
        self, pos: torch.Tensor, vel: torch.Tensor, accel: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vel.shape != accel.shape:
            raise ValueError(f"vel and accel must match, got {vel.shape} vs {accel.shape}")
        if pos.shape[-1] != vel.shape[-1]:
            vel_next = vel + accel * dt
            return pos, vel_next
        pos_next = pos + vel * dt
        vel_next = vel + accel * dt
        return pos_next, vel_next
