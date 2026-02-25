import torch

from gns.config import Config
from gns.factory import build_graph_builder, build_model
from gns.train.trainer import default_x_builder
from gns.typing import GraphBatch


def test_forward_shapes_and_device_cpu():
    cfg = Config(device="cpu")
    model = build_model(cfg).to("cpu")
    graph_builder = build_graph_builder(cfg)

    n_nodes, dim = 8, cfg.data.dim
    pos = torch.randn(n_nodes, dim)
    vel = torch.randn(n_nodes, dim)
    batch = torch.zeros(n_nodes, dtype=torch.long)

    edge_index, edge_attr = graph_builder.build(pos, batch=batch)
    x = default_x_builder(pos, vel, None)

    graph = GraphBatch(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    out = model(graph)

    assert out.shape == (n_nodes, cfg.model.out_dim)
    assert out.device.type == "cpu"


def test_forward_shapes_cuda_if_available():
    if not torch.cuda.is_available():
        return

    cfg = Config(device="cuda")
    model = build_model(cfg).to("cuda")
    graph_builder = build_graph_builder(cfg)

    n_nodes, dim = 8, cfg.data.dim
    pos = torch.randn(n_nodes, dim, device="cuda")
    vel = torch.randn(n_nodes, dim, device="cuda")
    batch = torch.zeros(n_nodes, dtype=torch.long, device="cuda")

    edge_index, edge_attr = graph_builder.build(pos, batch=batch)
    x = default_x_builder(pos, vel, None)

    graph = GraphBatch(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    out = model(graph)

    assert out.shape == (n_nodes, cfg.model.out_dim)
    assert out.device.type == "cuda"
