"""
Shared lightweight dataclasses to keep interfaces explicit.

These mirror common graph-learning conventions (PyG-like) without depending on PyG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class GraphBatch:
    """
    A batch of one or more graphs, flattened into a single node list.

    Attributes:
      x: [num_nodes, node_feat_dim]
      pos: [num_nodes, spatial_dim]
      edge_index: [2, num_edges] (src, dst) indices into the flattened node list
      edge_attr: [num_edges, edge_feat_dim] or None
      batch: [num_nodes] graph id per node (0..B-1), or None if single graph
    """

    x: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor]
    batch: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TrajectoryBatch:
    """
    A batch of trajectories for time-dependent rollouts.

    Minimal assumption: fixed num_nodes per example for the default collate.
    You can replace collate to support ragged/variable-node trajectories later.

    Attributes:
      pos: [B, T, N, D]
      vel: [B, T, N, D]
      node_type: [B, N] or None
      dt: float (shared across batch for simplicity)
    """

    pos: torch.Tensor
    vel: torch.Tensor
    dt: float
    node_type: Optional[torch.Tensor] = None
