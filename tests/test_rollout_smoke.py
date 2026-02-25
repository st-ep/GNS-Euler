import torch

from gns.config import Config
from gns.factory import build_graph_builder, build_integrator, build_model
from gns.rollout.rollout import rollout
from gns.train.trainer import default_x_builder


@torch.no_grad()
def test_rollout_smoke_cpu():
    cfg = Config(device="cpu")
    model = build_model(cfg).to("cpu")
    graph_builder = build_graph_builder(cfg)
    integrator = build_integrator(cfg)

    n_nodes, dim = 8, cfg.data.dim
    pos0 = torch.randn(n_nodes, dim)
    vel0 = torch.randn(n_nodes, dim)

    res = rollout(
        model=model,
        graph_builder=graph_builder,
        integrator=integrator,
        x_builder=default_x_builder,
        pos0=pos0,
        vel0=vel0,
        dt=cfg.data.dt,
        horizon=3,
        batch_index=None,
    )
    assert res.pos.shape == (4, n_nodes, dim)
    assert torch.isfinite(res.pos).all()
