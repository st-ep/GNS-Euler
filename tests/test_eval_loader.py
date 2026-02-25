import torch

from gns.config import Config, apply_overrides
from gns.data.loader import make_dataloader
from gns.factory import build_dataset, build_graph_builder, build_integrator, build_model
from gns.train.trainer import Trainer
from gns.utils.distributed import DistInfo


@torch.no_grad()
def test_eval_rollout_loader_aggregates_batches(tmp_path):
    cfg = apply_overrides(
        Config(device="cpu"),
        [
            "data.batch_size=2",
            "eval.rollout_horizon=3",
            "eval.max_batches=2",
        ],
    )

    dist = DistInfo(enabled=False)
    dataset = build_dataset(cfg)
    eval_dl = make_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        dist=dist,
    )

    trainer = Trainer(
        cfg=cfg,
        model=build_model(cfg),
        graph_builder=build_graph_builder(cfg),
        integrator=build_integrator(cfg),
        dist=dist,
        run_dir=str(tmp_path),
    )

    metrics = trainer.eval_rollout_loader(eval_dl, max_batches=2)
    assert "pos_mse_mean" in metrics
    assert "num_rollouts" in metrics
    assert "num_batches" in metrics
    assert metrics["num_batches"] == 2.0
    assert metrics["num_rollouts"] == 4.0
    assert metrics["pos_mse_mean"] >= 0.0
