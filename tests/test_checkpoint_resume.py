from pathlib import Path

import torch

from gns.config import Config, apply_overrides
from gns.data.loader import make_dataloader
from gns.factory import build_dataset, build_graph_builder, build_integrator, build_model
from gns.train.trainer import Trainer
from gns.utils.distributed import DistInfo


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def test_checkpoint_resume_restores_step_and_state(tmp_path: Path):
    cfg = apply_overrides(
        Config(device="cpu"),
        [
            "train.ckpt_every=1",
            "train.amp=false",
            "data.batch_size=2",
            "norm.enabled=true",
        ],
    )

    dist = DistInfo(enabled=False)
    run_dir = str(tmp_path)

    dataset = build_dataset(cfg)
    dl = make_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        shuffle=True,
        dist=dist,
    )
    batch = next(iter(dl))

    trainer1 = Trainer(
        cfg=cfg,
        model=build_model(cfg),
        graph_builder=build_graph_builder(cfg),
        integrator=build_integrator(cfg),
        dist=dist,
        run_dir=run_dir,
    )

    _ = trainer1.train_step(batch, step=0)
    trainer1.maybe_save(step=0)

    ckpt_path = tmp_path / "checkpoints" / "step_0000000.pt"
    assert ckpt_path.exists()

    trainer2 = Trainer(
        cfg=cfg,
        model=build_model(cfg),
        graph_builder=build_graph_builder(cfg),
        integrator=build_integrator(cfg),
        dist=dist,
        run_dir=run_dir,
    )
    start_step = trainer2.load_checkpoint(str(ckpt_path))
    assert start_step == 1

    m1 = _unwrap_model(trainer1.model)
    m2 = _unwrap_model(trainer2.model)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)

    assert trainer2.optim.state_dict()["state"] != {}
    assert trainer1.x_norm is not None and trainer2.x_norm is not None
    assert trainer1.target_norm is not None and trainer2.target_norm is not None
    assert torch.allclose(trainer1.x_norm.count, trainer2.x_norm.count)
    assert torch.allclose(trainer1.target_norm.count, trainer2.target_norm.count)
