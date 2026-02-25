"""
Tiny factories to keep CLI clean and avoid a big registry.

Add your new components here with minimal diffs.
"""

from __future__ import annotations

from gns.config import Config
from gns.data.dummy import DummyDatasetConfig, DummyTrajectoryDataset
from gns.data.ns2d import NS2DDatasetConfig, NS2DTrajectoryDataset
from gns.graphs.radius import RadiusGraphBuilder, RadiusGraphConfig
from gns.models.gns import GNSModel, GNSModelConfig
from gns.rollout.integrators import ExplicitEuler, SemiImplicitEuler


def build_dataset(cfg: Config, split: str | None = None):
    split_name = split or cfg.data.split
    max_samples = int(cfg.data.max_samples)
    if split is not None and split_name == cfg.data.eval_split and int(cfg.data.eval_max_samples) > 0:
        max_samples = int(cfg.data.eval_max_samples)

    if cfg.data.dataset == "dummy":
        dcfg = DummyDatasetConfig(
            seq_len=cfg.data.seq_len,
            num_nodes=cfg.data.num_nodes,
            dim=cfg.data.dim,
            dt=cfg.data.dt,
            seed=cfg.seed,
        )
        return DummyTrajectoryDataset(dcfg)
    if cfg.data.dataset == "ns2d":
        dcfg = NS2DDatasetConfig(
            root_dir=cfg.data.root_dir,
            split=split_name,
            seq_len=cfg.data.seq_len,
            frame_stride=cfg.data.frame_stride,
            spatial_stride=cfg.data.spatial_stride,
            dt_override=cfg.data.dt,
            max_samples=max_samples,
        )
        return NS2DTrajectoryDataset(dcfg)
    raise ValueError(f"Unknown dataset: {cfg.data.dataset}")


def build_graph_builder(cfg: Config):
    if cfg.graph.builder == "radius":
        gcfg = RadiusGraphConfig(radius=cfg.graph.radius, max_neighbors=cfg.graph.max_neighbors)
        return RadiusGraphBuilder(gcfg)
    raise ValueError(f"Unknown graph builder: {cfg.graph.builder}")


def build_model(cfg: Config):
    mcfg = GNSModelConfig(
        node_in_dim=cfg.model.node_in_dim,
        edge_in_dim=cfg.model.edge_in_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_message_passing_steps=cfg.model.num_message_passing_steps,
        out_dim=cfg.model.out_dim,
    )
    return GNSModel(mcfg)


def build_integrator(cfg: Config):
    kind = cfg.integrator.kind.lower().strip()
    if kind in {"semi_implicit_euler", "semi_implicit", "symplectic_euler"}:
        return SemiImplicitEuler()
    if kind in {"explicit_euler", "explicit"}:
        return ExplicitEuler()
    raise ValueError(f"Unknown integrator kind: {cfg.integrator.kind}")
