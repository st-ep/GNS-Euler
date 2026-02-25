import torch

from gns.config import Config
from gns.factory import build_graph_builder, build_model
from gns.train.trainer import default_x_builder
from gns.typing import GraphBatch
from gns.utils.seed import set_seed


def test_cpu_determinism_same_seed_same_output():
    # Force CPU determinism only (GPU scatter ops may be nondeterministic).
    set_seed(999, deterministic=False)

    cfg = Config(device="cpu")
    model = build_model(cfg).eval()
    graph_builder = build_graph_builder(cfg)

    n_nodes, dim = 8, cfg.data.dim
    pos = torch.randn(n_nodes, dim)
    vel = torch.randn(n_nodes, dim)
    batch = torch.zeros(n_nodes, dtype=torch.long)

    edge_index, edge_attr = graph_builder.build(pos, batch=batch)
    x = default_x_builder(pos, vel, None)
    graph = GraphBatch(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    out1 = model(graph).detach().clone()

    set_seed(999, deterministic=False)
    model2 = build_model(cfg).eval()
    out2 = model2(graph).detach().clone()

    assert torch.allclose(out1, out2, atol=0, rtol=0)
