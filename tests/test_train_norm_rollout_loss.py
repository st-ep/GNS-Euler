import torch

from gns.config import Config, apply_overrides
from gns.data.loader import make_dataloader
from gns.factory import build_dataset, build_graph_builder, build_integrator, build_model
from gns.train.trainer import Trainer
from gns.utils.distributed import DistInfo


def test_train_step_with_norm_and_rollout_loss(tmp_path):
    cfg = apply_overrides(
        Config(device="cpu"),
        [
            "data.batch_size=2",
            "train.amp=false",
            "norm.enabled=true",
            "loss.rollout_weight=0.5",
            "loss.rollout_horizon=2",
        ],
    )
    dist = DistInfo(enabled=False)
    dataset = build_dataset(cfg)
    dl = make_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        dist=dist,
    )
    batch = next(iter(dl))

    trainer = Trainer(
        cfg=cfg,
        model=build_model(cfg),
        graph_builder=build_graph_builder(cfg),
        integrator=build_integrator(cfg),
        dist=dist,
        run_dir=str(tmp_path),
    )

    metrics = trainer.train_step(batch, step=0)
    assert "loss" in metrics
    assert "loss/one_step" in metrics
    assert "loss/rollout" in metrics
    assert metrics["loss"] >= 0.0
    assert metrics["loss/rollout"] >= 0.0
    assert trainer.x_norm is not None and trainer.target_norm is not None
    assert float(trainer.x_norm.count.item()) > 0.0
    assert float(trainer.target_norm.count.item()) > 0.0


def test_eval_rollout_works_with_norm_enabled(tmp_path):
    cfg = apply_overrides(
        Config(device="cpu"),
        [
            "data.batch_size=2",
            "norm.enabled=true",
        ],
    )
    dist = DistInfo(enabled=False)
    dataset = build_dataset(cfg)
    dl = make_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        dist=dist,
    )
    batch = next(iter(dl))

    trainer = Trainer(
        cfg=cfg,
        model=build_model(cfg),
        graph_builder=build_graph_builder(cfg),
        integrator=build_integrator(cfg),
        dist=dist,
        run_dir=str(tmp_path),
    )

    _ = trainer.train_step(batch, step=0)  # updates running stats
    with torch.no_grad():
        metrics = trainer.eval_rollout(batch)
    assert "pos_mse_mean" in metrics
    assert metrics["pos_mse_mean"] >= 0.0
