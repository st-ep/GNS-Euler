"""
Minimal trainer.

Design goals:
- explicit step() and eval()
- easy to read, easy to edit
- DDP/AMP friendly structure, but no heavy framework

=== RESEARCH KNOB ===
- how batches are sampled (random t vs fixed)
- objective definition (accel vs delta-v)
- normalization (feature/target)
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from typing import Any

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from gns.config import Config
from gns.graphs.base import GraphBuilder
from gns.nn.normalizer import RunningNorm
from gns.nn.spectral import omega_to_velocity
from gns.rollout.integrators import Integrator
from gns.train.eval import rollout_position_mse_error, rollout_position_mse_from_error
from gns.train.losses import LossConfig, one_step_loss, rollout_position_loss
from gns.typing import GraphBatch, TrajectoryBatch
from gns.utils.checkpoint import load_checkpoint, save_checkpoint
from gns.utils.distributed import DistInfo, is_rank0


def default_x_builder(pos: torch.Tensor, vel: torch.Tensor, node_type: Any) -> torch.Tensor:
    """
    === RESEARCH KNOB === Feature construction.
    Minimal: concatenate pos and vel -> x dim = 2D
    """

    return torch.cat([pos, vel], dim=-1)


class Trainer:
    def __init__(
        self,
        *,
        cfg: Config,
        model: nn.Module,
        graph_builder: GraphBuilder,
        integrator: Integrator,
        dist: DistInfo,
        run_dir: str,
    ):
        self.cfg = cfg
        self.dist = dist
        self.run_dir = run_dir

        if cfg.device == "cuda":
            if torch.cuda.is_available():
                if dist.enabled:
                    self.device = torch.device(f"cuda:{dist.local_rank}")
                else:
                    self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(cfg.device)
        model = model.to(self.device)

        if dist.enabled:
            model = DDP(model, device_ids=[dist.local_rank] if self.device.type == "cuda" else None)

        self.model = model
        self.graph_builder = graph_builder
        self.integrator = integrator

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        amp_enabled = cfg.train.amp and self.device.type == "cuda"
        self.scaler = _make_grad_scaler(amp_enabled) if amp_enabled else _NoopGradScaler()
        self.loss_cfg = LossConfig(
            kind=cfg.loss.kind,
            huber_delta=cfg.loss.huber_delta,
            rollout_weight=cfg.loss.rollout_weight,
            rollout_horizon=cfg.loss.rollout_horizon,
        )

        self.norm_enabled = bool(cfg.norm.enabled)
        if self.norm_enabled:
            self.x_norm = RunningNorm(dim=cfg.model.node_in_dim).to(self.device)
            self.target_norm = RunningNorm(dim=cfg.model.out_dim).to(self.device)
        else:
            self.x_norm = None
            self.target_norm = None

        self.grid_n = int(cfg.data.grid_n)
        self.domain_L = float(cfg.data.domain_L)

    def _augment_vel_with_velocity(
        self, vel_flat: torch.Tensor, bsz: int, num_nodes: int,
    ) -> torch.Tensor:
        """If grid_n > 0, compute (u, v) from omega and concatenate."""
        if self.grid_n <= 0:
            return vel_flat
        n = self.grid_n
        omega_grid = vel_flat[:, 0].view(bsz, n, n)
        u, v = omega_to_velocity(omega_grid, L=self.domain_L)
        u_flat = u.reshape(bsz * num_nodes, 1)
        v_flat = v.reshape(bsz * num_nodes, 1)
        return torch.cat([vel_flat, u_flat, v_flat], dim=-1)

    @torch.no_grad()
    def warmup_normalization(self, dataloader, num_batches: int = 50) -> None:
        """Pre-compute normalization statistics before training begins."""
        if not self.norm_enabled:
            return
        assert self.x_norm is not None and self.target_norm is not None
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            pos = batch.pos.to(self.device)
            vel = batch.vel.to(self.device)
            bsz, seq_len, num_nodes, pos_dim = pos.shape
            vel_dim = int(vel.shape[-1])
            dt = float(batch.dt)
            t = i % (seq_len - 1)
            pos_flat = pos[:, t].reshape(bsz * num_nodes, pos_dim)
            vel_flat = vel[:, t].reshape(bsz * num_nodes, vel_dim)
            vel_aug = self._augment_vel_with_velocity(vel_flat, bsz, num_nodes)
            target = ((vel[:, t + 1] - vel[:, t]) / dt).reshape(bsz * num_nodes, vel_dim)
            x = default_x_builder(pos_flat, vel_aug, None)
            self.x_norm.update(x)
            self.target_norm.update(target)

    def train_step(self, batch: TrajectoryBatch, step: int) -> dict[str, float]:
        self.model.train()
        pos = batch.pos.to(self.device)
        vel = batch.vel.to(self.device)

        bsz, seq_len, num_nodes, pos_dim = pos.shape
        vel_dim = int(vel.shape[-1])
        dt = float(batch.dt)

        requested_roll_h = int(self.loss_cfg.rollout_horizon) if self.loss_cfg.rollout_weight > 0 else 0
        required_h = max(1, requested_roll_h)
        if seq_len <= required_h:
            raise ValueError(
                f"Sequence length {seq_len} is too short for required horizon {required_h}."
            )
        t = torch.randint(low=0, high=seq_len - required_h, size=(1,), device=pos.device).item()

        pos_t = pos[:, t]
        vel_t = vel[:, t]
        vel_tp1 = vel[:, t + 1]

        target_accel = (vel_tp1 - vel_t) / dt

        if self.cfg.train.noise_std > 0 and self.model.training:
            vel_t = vel_t + torch.randn_like(vel_t) * self.cfg.train.noise_std

        pos_flat = pos_t.reshape(bsz * num_nodes, pos_dim)
        vel_flat = vel_t.reshape(bsz * num_nodes, vel_dim)
        vel_aug = self._augment_vel_with_velocity(vel_flat, bsz, num_nodes)
        target_flat = target_accel.reshape(bsz * num_nodes, vel_dim)
        batch_index = torch.arange(bsz, device=self.device).repeat_interleave(num_nodes)

        edge_index, edge_attr = self.graph_builder.build(pos_flat, batch=batch_index)
        x_raw = default_x_builder(pos_flat, vel_aug, None)

        if self._should_update_norm(step):
            assert self.x_norm is not None and self.target_norm is not None
            self.x_norm.update(x_raw.detach())
            self.target_norm.update(target_flat.detach())

        x = self._normalize_x(x_raw)
        target_for_loss = self._normalize_target(target_flat)

        graph = GraphBatch(
            x=x,
            pos=pos_flat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_index,
        )

        self.optim.zero_grad(set_to_none=True)

        with _autocast(self.scaler.is_enabled()):
            pred = self.model(graph)
            one_step = one_step_loss(pred, target_for_loss, self.loss_cfg)
            total_loss = one_step
            rollout_term = pred.new_zeros(())

            if self.loss_cfg.rollout_weight > 0 and self.loss_cfg.rollout_horizon > 0:
                rollout_term = self._rollout_training_loss(
                    pos=pos,
                    vel=vel,
                    t=t,
                    dt=dt,
                    horizon=min(int(self.loss_cfg.rollout_horizon), seq_len - 1 - t),
                )
                total_loss = total_loss + float(self.loss_cfg.rollout_weight) * rollout_term

        self.scaler.scale(total_loss).backward()

        if self.cfg.train.grad_clip > 0:
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)

        self.scaler.step(self.optim)
        self.scaler.update()

        metrics = {
            "loss": float(total_loss.item()),
            "loss/one_step": float(one_step.item()),
        }
        if self.loss_cfg.rollout_weight > 0 and self.loss_cfg.rollout_horizon > 0:
            metrics["loss/rollout"] = float(rollout_term.item())
        return metrics

    @torch.no_grad()
    def eval_rollout(self, batch: TrajectoryBatch) -> dict[str, float]:
        pred_state, true_state, _, _ = self._rollout_state_sequences(batch)
        err = rollout_position_mse_error(pred_state, true_state)
        return rollout_position_mse_from_error(err)

    @torch.no_grad()
    def eval_rollout_loader(self, dataloader, max_batches: int = 0) -> dict[str, float]:
        err_sum = None
        num_rollouts = 0
        num_batches = 0

        for i, batch in enumerate(dataloader):
            if max_batches > 0 and i >= max_batches:
                break
            pred_state, true_state, _, _ = self._rollout_state_sequences(batch)
            err = rollout_position_mse_error(pred_state, true_state)  # [B, H+1]
            batch_sum = err.sum(dim=0)  # [H+1]
            err_sum = batch_sum if err_sum is None else (err_sum + batch_sum)
            num_rollouts += int(err.size(0))
            num_batches += 1

        if err_sum is None or num_rollouts == 0:
            raise RuntimeError("Eval dataloader produced no batches.")

        mean_err = (err_sum / float(num_rollouts)).unsqueeze(0)  # [1, H+1]
        metrics = rollout_position_mse_from_error(mean_err)
        metrics["num_rollouts"] = float(num_rollouts)
        metrics["num_batches"] = float(num_batches)
        return metrics

    @torch.no_grad()
    def collect_rollout_examples(
        self,
        dataloader,
        *,
        max_samples: int,
        max_batches: int = 0,
    ) -> dict[str, Any]:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be >0, got {max_samples}")

        pred_chunks: list[torch.Tensor] = []
        true_chunks: list[torch.Tensor] = []
        total = 0
        dt_value = 0.0
        state_kind = "pos"

        for i, batch in enumerate(dataloader):
            if max_batches > 0 and i >= max_batches:
                break
            pred_state, true_state, dt_value, state_kind = self._rollout_state_sequences(batch)
            take = min(max_samples - total, int(pred_state.size(0)))
            if take <= 0:
                break
            pred_chunks.append(pred_state[:take].detach().cpu())
            true_chunks.append(true_state[:take].detach().cpu())
            total += take
            if total >= max_samples:
                break

        if total == 0:
            raise RuntimeError("No rollout examples collected.")

        return {
            "pred": torch.cat(pred_chunks, dim=0),
            "true": torch.cat(true_chunks, dim=0),
            "dt": float(dt_value),
            "state_kind": state_kind,
        }

    @torch.no_grad()
    def _rollout_state_sequences(
        self,
        batch: TrajectoryBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, float, str]:
        """
        Rollout state trajectories.

        Returns:
          pred_state: [B, H+1, N, S]
          true_state: [B, H+1, N, S]
          dt: float
          state_kind: "pos" | "vel"
        """
        pos = batch.pos.to(self.device)
        vel = batch.vel.to(self.device)
        bsz, seq_len, num_nodes, pos_dim = pos.shape
        vel_dim = int(vel.shape[-1])
        dt = float(batch.dt)
        horizon = min(int(self.cfg.eval.rollout_horizon), seq_len - 1)

        state_on_vel = pos_dim != vel_dim
        state_kind = "vel" if state_on_vel else "pos"

        pos_cur = pos[:, 0].reshape(bsz * num_nodes, pos_dim)
        vel_cur = vel[:, 0].reshape(bsz * num_nodes, vel_dim)
        true_state = vel[:, : horizon + 1] if state_on_vel else pos[:, : horizon + 1]
        batch_index = torch.arange(bsz, device=self.device).repeat_interleave(num_nodes)

        model = self.model.module if hasattr(self.model, "module") else self.model
        was_training = model.training
        model.eval()

        if state_on_vel:
            pred_state_seq = [vel_cur.view(bsz, num_nodes, vel_dim)]
        else:
            pred_state_seq = [pos_cur.view(bsz, num_nodes, pos_dim)]

        interval = int(self.cfg.graph.edge_rebuild_interval)
        use_cache = bool(self.cfg.graph.cache_edges)
        cached_edge_index = None
        cached_edge_attr = None
        try:
            for k in range(horizon):
                if use_cache:
                    should_rebuild = cached_edge_index is None
                    if interval > 0 and (k % interval == 0):
                        should_rebuild = True
                    if should_rebuild:
                        cached_edge_index, cached_edge_attr = self.graph_builder.build(
                            pos_cur, batch=batch_index
                        )
                    edge_index, edge_attr = cached_edge_index, cached_edge_attr
                else:
                    edge_index, edge_attr = self.graph_builder.build(pos_cur, batch=batch_index)

                vel_aug = self._augment_vel_with_velocity(vel_cur, bsz, num_nodes)
                x_raw = default_x_builder(pos_cur, vel_aug, None)
                x = self._normalize_x(x_raw)
                graph = GraphBatch(
                    x=x,
                    pos=pos_cur,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch_index,
                )
                pred = model(graph)
                accel = self._denormalize_target(pred)
                pos_cur, vel_cur = self.integrator.step(pos_cur, vel_cur, accel, dt)
                if state_on_vel:
                    pred_state_seq.append(vel_cur.view(bsz, num_nodes, vel_dim))
                else:
                    pred_state_seq.append(pos_cur.view(bsz, num_nodes, pos_dim))
        finally:
            model.train(was_training)

        pred_state = torch.stack(pred_state_seq, dim=1)  # [B,H+1,N,S]
        return pred_state, true_state, dt, state_kind

    def maybe_save(self, step: int) -> None:
        if not is_rank0(self.dist):
            return
        if step % self.cfg.train.ckpt_every != 0:
            return

        path = os.path.join(self.run_dir, "checkpoints", f"step_{step:07d}.pt")
        model = self.model.module if hasattr(self.model, "module") else self.model
        save_checkpoint(
            path,
            step=step,
            model=model,
            optimizer=self.optim,
            scheduler=None,
            scaler=self.scaler,
            cfg_dict=asdict(self.cfg),
            extra_state=self._extra_state_for_checkpoint(),
        )

    def maybe_save_best(self, step: int, eval_metric: float) -> None:
        if not is_rank0(self.dist):
            return
        if not hasattr(self, "_best_eval_metric"):
            self._best_eval_metric = float("inf")
        if eval_metric >= self._best_eval_metric:
            return
        self._best_eval_metric = eval_metric
        path = os.path.join(self.run_dir, "checkpoints", "best.pt")
        model = self.model.module if hasattr(self.model, "module") else self.model
        save_checkpoint(
            path,
            step=step,
            model=model,
            optimizer=self.optim,
            scheduler=None,
            scaler=self.scaler,
            cfg_dict=asdict(self.cfg),
            extra_state=self._extra_state_for_checkpoint(),
        )
        print(f"[step {step}] New best model saved (eval={eval_metric:.6f})")

    def load_checkpoint(self, path: str) -> int:
        model = self.model.module if hasattr(self.model, "module") else self.model
        payload = load_checkpoint(
            path,
            model=model,
            optimizer=self.optim,
            scheduler=None,
            scaler=self.scaler,
            map_location=str(self.device),
        )
        self._load_extra_state_from_checkpoint(payload.get("extra_state"))
        return int(payload.get("step", -1)) + 1

    def _rollout_training_loss(
        self,
        *,
        pos: torch.Tensor,
        vel: torch.Tensor,
        t: int,
        dt: float,
        horizon: int,
    ) -> torch.Tensor:
        if horizon <= 0:
            return pos.new_zeros(())

        bsz, _, num_nodes, pos_dim = pos.shape
        vel_dim = int(vel.shape[-1])
        state_on_vel = pos_dim != vel_dim
        batch_index = torch.arange(bsz, device=self.device).repeat_interleave(num_nodes)

        pos_cur = pos[:, t]
        vel_cur = vel[:, t]

        interval = int(self.cfg.graph.edge_rebuild_interval)
        use_cache = bool(self.cfg.graph.cache_edges)
        cached_edge_index = None
        cached_edge_attr = None

        losses = []
        for k in range(horizon):
            pos_flat = pos_cur.reshape(bsz * num_nodes, pos_dim)
            vel_flat = vel_cur.reshape(bsz * num_nodes, vel_dim)

            if use_cache:
                should_rebuild = cached_edge_index is None
                if interval > 0 and (k % interval == 0):
                    should_rebuild = True
                if should_rebuild:
                    cached_edge_index, cached_edge_attr = self.graph_builder.build(
                        pos_flat, batch=batch_index
                    )
                edge_index, edge_attr = cached_edge_index, cached_edge_attr
            else:
                edge_index, edge_attr = self.graph_builder.build(pos_flat, batch=batch_index)

            vel_aug = self._augment_vel_with_velocity(vel_flat, bsz, num_nodes)
            x_raw = default_x_builder(pos_flat, vel_aug, None)
            x = self._normalize_x(x_raw)
            graph = GraphBatch(
                x=x,
                pos=pos_flat,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch_index,
            )

            pred = self.model(graph)
            accel = self._denormalize_target(pred)
            pos_next_flat, vel_next_flat = self.integrator.step(pos_flat, vel_flat, accel, dt)
            pos_next = pos_next_flat.view(bsz, num_nodes, pos_dim)
            vel_next = vel_next_flat.view(bsz, num_nodes, vel_dim)

            if state_on_vel:
                true_state_next = vel[:, t + k + 1]
                pred_state_next = vel_next
            else:
                true_state_next = pos[:, t + k + 1]
                pred_state_next = pos_next
            losses.append(rollout_position_loss(pred_state_next, true_state_next, self.loss_cfg))

            pos_cur, vel_cur = pos_next, vel_next

        return torch.stack(losses, dim=0).mean()

    def _should_update_norm(self, step: int) -> bool:
        if not self.norm_enabled:
            return False
        if self.cfg.norm.update_steps > 0 and step >= int(self.cfg.norm.update_steps):
            return False
        if self.cfg.norm.freeze_after > 0 and step >= int(self.cfg.norm.freeze_after):
            return False
        return True

    def _norm_ready(self) -> bool:
        if not self.norm_enabled or self.x_norm is None or self.target_norm is None:
            return False
        return bool(self.x_norm.count.item() > 0 and self.target_norm.count.item() > 0)

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        if self._norm_ready():
            assert self.x_norm is not None
            return self.x_norm.normalize(x)
        return x

    def _normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        if self._norm_ready():
            assert self.target_norm is not None
            return self.target_norm.normalize(target)
        return target

    def _denormalize_target(self, pred: torch.Tensor) -> torch.Tensor:
        if self._norm_ready():
            assert self.target_norm is not None
            return self.target_norm.denormalize(pred)
        return pred

    def _extra_state_for_checkpoint(self) -> dict[str, dict[str, torch.Tensor]] | None:
        if not self.norm_enabled:
            return None
        assert self.x_norm is not None and self.target_norm is not None
        return {
            "x_norm": self.x_norm.state_dict(),
            "target_norm": self.target_norm.state_dict(),
        }

    def _load_extra_state_from_checkpoint(self, extra_state) -> None:
        if not self.norm_enabled or not extra_state:
            return
        if self.x_norm is None or self.target_norm is None:
            return
        if "x_norm" in extra_state:
            self.x_norm.load_state_dict(extra_state["x_norm"])
        if "target_norm" in extra_state:
            self.target_norm.load_state_dict(extra_state["target_norm"])


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    if enabled:
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


class _NoopGradScaler:
    def is_enabled(self) -> bool:
        return False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer) -> None:
        del optimizer

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def state_dict(self) -> dict[str, bool]:
        return {"enabled": False}

    def load_state_dict(self, state: dict[str, bool]) -> None:
        del state
