from gns.config import Config, apply_overrides
from gns.data.loader import make_dataloader
from gns.factory import build_dataset, build_graph_builder, build_integrator, build_model
from gns.train.trainer import Trainer
from gns.utils.distributed import DistInfo


def test_ns2d_trainer_train_and_eval_smoke(tmp_path):
    cfg = apply_overrides(
        Config(device="cpu"),
        [
            "data.dataset=ns2d",
            "data.root_dir=data/ns2d/ns2d_traj",
            "data.split=train",
            "data.eval_split=test",
            "data.batch_size=2",
            "data.seq_len=16",
            "data.spatial_stride=8",
            "data.max_samples=6",
            "data.eval_max_samples=4",
            "data.dt=0.0",
            "graph.radius=0.2",
            "graph.max_neighbors=8",
            "graph.cache_edges=true",
            "graph.edge_rebuild_interval=0",
            "model.node_in_dim=3",
            "model.edge_in_dim=3",
            "model.out_dim=1",
            "train.amp=false",
            "norm.enabled=true",
            "loss.rollout_weight=0.2",
            "loss.rollout_horizon=2",
            "eval.rollout_horizon=6",
            "eval.max_batches=1",
        ],
    )
    dist = DistInfo(enabled=False)

    train_ds = build_dataset(cfg, split=cfg.data.split)
    eval_ds = build_dataset(cfg, split=cfg.data.eval_split)
    train_dl = make_dataloader(
        train_ds,
        batch_size=cfg.data.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        dist=dist,
    )
    eval_dl = make_dataloader(
        eval_ds,
        batch_size=cfg.data.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        dist=DistInfo(enabled=False),
    )

    trainer = Trainer(
        cfg=cfg,
        model=build_model(cfg),
        graph_builder=build_graph_builder(cfg),
        integrator=build_integrator(cfg),
        dist=dist,
        run_dir=str(tmp_path),
    )

    batch = next(iter(train_dl))
    metrics = trainer.train_step(batch, step=0)
    assert "loss" in metrics
    assert "loss/one_step" in metrics
    assert metrics["loss"] >= 0.0

    eval_metrics = trainer.eval_rollout_loader(eval_dl, max_batches=1)
    assert "pos_mse_mean" in eval_metrics
    assert eval_metrics["num_batches"] == 1.0

    examples = trainer.collect_rollout_examples(eval_dl, max_samples=2, max_batches=1)
    assert examples["pred"].shape[0] == 2
    assert examples["pred"].shape[-1] == 1
    assert examples["state_kind"] == "vel"
