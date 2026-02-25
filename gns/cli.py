"""
CLI entrypoint.

Examples:
  python -m gns train --config configs/base.json
  python -m gns train --config configs/base.json --set train.lr=1e-4 --set graph.radius=0.2

DDP:
  torchrun --nproc_per_node=4 -m gns train --config configs/base.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

import torch

from gns.config import apply_overrides, load_config
from gns.data.loader import make_dataloader
from gns.factory import build_dataset, build_graph_builder, build_integrator, build_model
from gns.train.trainer import Trainer
from gns.utils.distributed import (
    DistInfo,
    barrier_if_distributed,
    cleanup_distributed,
    init_distributed,
    is_rank0,
)
from gns.utils.logging import log_metrics, setup_loggers
from gns.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser("gns")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", type=str, required=True)
    p_train.add_argument("--set", dest="overrides", action="append", default=[])
    p_train.add_argument("--resume", type=str, default=None)

    p_eval = sub.add_parser("eval", help="Run rollout evaluation")
    p_eval.add_argument("--config", type=str, required=True)
    p_eval.add_argument("--set", dest="overrides", action="append", default=[])
    p_eval.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    dist = init_distributed()
    set_seed(cfg.seed + (dist.rank if dist.enabled else 0), deterministic=False)

    if is_rank0(dist):
        run_dir = _resolve_run_dir(
            args.cmd,
            args.resume if args.cmd == "train" else args.checkpoint if args.cmd == "eval" else None,
        )
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = "runs/_ddp_nonzero_rank"

    loggers = setup_loggers(run_dir, enable_tb=is_rank0(dist))

    if cfg.device == "cuda":
        use_pin_memory = cfg.data.pin_memory and torch.cuda.is_available()
    else:
        use_pin_memory = False

    train_dataset = build_dataset(cfg, split=cfg.data.split)
    eval_dataset = build_dataset(cfg, split=cfg.data.eval_split)
    train_dl = make_dataloader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=use_pin_memory,
        shuffle=True,
        dist=dist,
    )
    eval_dl = make_dataloader(
        eval_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=use_pin_memory,
        shuffle=False,
        dist=DistInfo(enabled=False),
    )

    model = build_model(cfg)
    graph_builder = build_graph_builder(cfg)
    integrator = build_integrator(cfg)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        graph_builder=graph_builder,
        integrator=integrator,
        dist=dist,
        run_dir=run_dir,
    )

    if args.cmd == "train":
        start_step = 0
        if args.resume:
            start_step = trainer.load_checkpoint(args.resume)
            barrier_if_distributed(dist)
            if is_rank0(dist):
                print(f"Resumed from {args.resume} at step={start_step}")
        elif cfg.norm.enabled:
            trainer.warmup_normalization(train_dl, num_batches=50)
            if is_rank0(dist):
                print(f"Norm warmup done (50 batches)")

        try:
            steps_per_epoch = len(train_dl)
        except TypeError:
            steps_per_epoch = 0
        if steps_per_epoch <= 0:
            raise RuntimeError(
                "Dataloader has zero length. Increase dataset size or reduce batch_size."
            )

        step = start_step
        epoch = step // steps_per_epoch
        _set_sampler_epoch(train_dl, epoch)
        it = iter(train_dl)

        # Best-effort intra-epoch restore: skip already-consumed batches.
        for _ in range(step % steps_per_epoch):
            try:
                _ = next(it)
            except StopIteration:
                epoch += 1
                _set_sampler_epoch(train_dl, epoch)
                it = iter(train_dl)

        while step < cfg.train.steps:
            try:
                batch = next(it)
            except StopIteration:
                epoch += 1
                _set_sampler_epoch(train_dl, epoch)
                it = iter(train_dl)
                batch = next(it)

            metrics = trainer.train_step(batch, step)
            if is_rank0(dist) and (step % cfg.train.log_every == 0):
                log_metrics(loggers, step, metrics)

            if step % cfg.train.eval_every == 0:
                barrier_if_distributed(dist)
                if is_rank0(dist):
                    eval_metrics = trainer.eval_rollout_loader(
                        eval_dl,
                        max_batches=int(cfg.eval.max_batches),
                    )
                    log_metrics(loggers, step, {f"eval/{k}": v for k, v in eval_metrics.items()})
                    trainer.maybe_save_best(step, eval_metrics.get("pos_mse_mean", float("inf")))
                barrier_if_distributed(dist)

            trainer.maybe_save(step)
            step += 1

    elif args.cmd == "eval":
        if args.checkpoint:
            _ = trainer.load_checkpoint(args.checkpoint)
        metrics = trainer.eval_rollout_loader(eval_dl, max_batches=int(cfg.eval.max_batches))
        if is_rank0(dist):
            log_metrics(loggers, 0, {f"eval/{k}": v for k, v in metrics.items()})
            if cfg.eval.save_json:
                _save_eval_json(run_dir, metrics, args.checkpoint)
            if cfg.eval.save_csv:
                _save_eval_csv(run_dir, metrics, args.checkpoint)
            if cfg.eval.save_predictions or cfg.eval.save_plots:
                examples = trainer.collect_rollout_examples(
                    eval_dl,
                    max_samples=int(cfg.eval.prediction_samples),
                    max_batches=int(cfg.eval.max_batches),
                )
                if cfg.eval.save_predictions:
                    _save_eval_predictions_npz(run_dir, examples, args.checkpoint)
                if cfg.eval.save_plots:
                    _maybe_save_ns2d_plots(
                        run_dir,
                        examples,
                        cfg=cfg,
                        checkpoint=args.checkpoint,
                        dataset=eval_dataset,
                    )

    cleanup_distributed(dist)


def _resolve_run_dir(cmd: str, ckpt_path: str | None) -> str:
    if cmd in {"train", "eval"} and ckpt_path:
        ckpt_abs = os.path.abspath(ckpt_path)
        ckpt_dir = os.path.dirname(ckpt_abs)
        if os.path.basename(ckpt_dir) == "checkpoints":
            return os.path.dirname(ckpt_dir)
    return os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))


def _set_sampler_epoch(dl, epoch: int) -> None:
    sampler = getattr(dl, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _save_eval_json(run_dir: str, metrics: dict[str, float], checkpoint: str | None) -> None:
    base = _eval_artifact_base(checkpoint)
    name = f"eval_metrics_{base}.json" if base else "eval_metrics.json"
    path = os.path.join(run_dir, name)
    payload = {"checkpoint": checkpoint, "metrics": metrics}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _save_eval_csv(run_dir: str, metrics: dict[str, float], checkpoint: str | None) -> None:
    base = _eval_artifact_base(checkpoint)
    name = f"eval_metrics_{base}.csv" if base else "eval_metrics.csv"
    path = os.path.join(run_dir, name)
    fields = ["checkpoint"] + sorted(metrics.keys())
    row = {"checkpoint": checkpoint or ""}
    row.update(metrics)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def _save_eval_predictions_npz(
    run_dir: str,
    examples: dict[str, Any],
    checkpoint: str | None,
) -> None:
    base = _eval_artifact_base(checkpoint)
    name = f"eval_rollout_{base}.npz" if base else "eval_rollout.npz"
    path = os.path.join(run_dir, name)
    pred = examples["pred"].cpu().numpy()
    true = examples["true"].cpu().numpy()
    np.savez_compressed(
        path,
        pred=pred,
        true=true,
        dt=np.asarray(float(examples["dt"]), dtype=np.float32),
        state_kind=np.asarray(str(examples["state_kind"])),
    )


def _maybe_save_ns2d_plots(
    run_dir: str,
    examples: dict[str, Any],
    *,
    cfg,
    checkpoint: str | None,
    dataset,
) -> None:
    try:
        from data.ns2d.visualize_trajectory import make_grid_video
    except Exception as e:
        print(f"[eval] skipping plots: unable to import ns2d visualizer ({e})")
        return

    pred = examples["pred"]  # [S, H+1, N, D]
    true = examples["true"]
    if pred.ndim != 4 or pred.shape[-1] != 1:
        print("[eval] skipping plots: expected [S,H+1,N,1] trajectories.")
        return

    n_nodes = int(pred.shape[2])
    side = int(round(n_nodes**0.5))
    if side * side != n_nodes:
        print(f"[eval] skipping plots: node count {n_nodes} is not a square grid.")
        return

    samples = int(pred.shape[0])
    indices = list(range(samples))
    omega_pred = pred[..., 0].reshape(samples, pred.shape[1], side, side).cpu().numpy()
    omega_true = true[..., 0].reshape(samples, true.shape[1], side, side).cpu().numpy()
    dt_save = float(examples["dt"])
    L = _dataset_L(dataset)
    ext = str(cfg.eval.plot_format).lower()
    fps = int(cfg.eval.plot_fps)
    base = _eval_artifact_base(checkpoint)
    suffix = f"_{base}" if base else ""

    out_pred = os.path.join(run_dir, f"eval_pred{suffix}.{ext}")
    out_true = os.path.join(run_dir, f"eval_true{suffix}.{ext}")
    make_grid_video(omega_pred, dt_save, L, indices, "pred", output_path=Path(out_pred), fps=fps)
    make_grid_video(omega_true, dt_save, L, indices, "true", output_path=Path(out_true), fps=fps)


def _dataset_L(dataset) -> float:
    root_dir = getattr(getattr(dataset, "cfg", None), "root_dir", None)
    if not root_dir:
        return 1.0
    meta_path = os.path.join(root_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return 1.0
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return float(meta.get("grid", {}).get("L", 1.0))
    except Exception:
        return 1.0


def _eval_artifact_base(checkpoint: str | None) -> str:
    if checkpoint:
        base = os.path.splitext(os.path.basename(checkpoint))[0]
        return base
    return ""
