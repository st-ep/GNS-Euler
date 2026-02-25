import torch
import torch.nn as nn

from gns.rollout.integrators import SemiImplicitEuler
from gns.rollout.rollout import rollout
from gns.train.trainer import default_x_builder


class _CountingGraphBuilder:
    def __init__(self):
        self.calls = 0

    def build(self, pos: torch.Tensor, batch: torch.Tensor | None = None):
        del batch
        self.calls += 1
        n, dim = pos.shape
        if n <= 1:
            return (
                torch.empty((2, 0), dtype=torch.long, device=pos.device),
                torch.empty((0, dim + 1), dtype=pos.dtype, device=pos.device),
            )
        src, dst = torch.where(
            ~torch.eye(n, dtype=torch.bool, device=pos.device)
        )  # fully connected directed without self loops
        edge_index = torch.stack([src, dst], dim=0)
        dx = pos[dst] - pos[src]
        dist = torch.norm(dx, dim=-1, keepdim=True)
        edge_attr = torch.cat([dx, dist], dim=-1)
        return edge_index, edge_attr


class _ZeroAccelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, g):
        return torch.zeros_like(g.pos)


@torch.no_grad()
def test_rollout_cache_edges_once_for_static_graph():
    model = _ZeroAccelModel()
    builder = _CountingGraphBuilder()
    integrator = SemiImplicitEuler()

    pos0 = torch.randn(8, 2)
    vel0 = torch.randn(8, 2)
    _ = rollout(
        model=model,
        graph_builder=builder,
        integrator=integrator,
        x_builder=default_x_builder,
        pos0=pos0,
        vel0=vel0,
        dt=0.01,
        horizon=5,
        cache_edges=True,
        edge_rebuild_interval=0,
    )
    assert builder.calls == 1


@torch.no_grad()
def test_rollout_cache_edges_rebuild_interval():
    model = _ZeroAccelModel()
    builder = _CountingGraphBuilder()
    integrator = SemiImplicitEuler()

    pos0 = torch.randn(8, 2)
    vel0 = torch.randn(8, 2)
    _ = rollout(
        model=model,
        graph_builder=builder,
        integrator=integrator,
        x_builder=default_x_builder,
        pos0=pos0,
        vel0=vel0,
        dt=0.01,
        horizon=5,
        cache_edges=True,
        edge_rebuild_interval=2,
    )
    assert builder.calls == 3  # t=0,2,4
