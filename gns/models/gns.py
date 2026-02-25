"""
GNS-style model wrapper: encode -> process -> decode.

=== RESEARCH KNOB ===
- feature construction in encode()
- processor in gns/models/processor.py
- decode head (accel vs delta-v vs next-pos)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from gns.models.processor import MPNNProcessor, ProcessorConfig
from gns.nn.mlp import MLP
from gns.typing import GraphBatch


@dataclass(frozen=True)
class GNSModelConfig:
    node_in_dim: int = 4
    edge_in_dim: int = 3
    hidden_dim: int = 128
    num_message_passing_steps: int = 3
    out_dim: int = 2


class GNSModel(nn.Module):
    def __init__(self, cfg: GNSModelConfig):
        super().__init__()
        hidden = cfg.hidden_dim
        self.node_enc = MLP(cfg.node_in_dim, [hidden, hidden], hidden)
        self.edge_enc = MLP(cfg.edge_in_dim, [hidden, hidden], hidden)
        self.processor = MPNNProcessor(
            ProcessorConfig(
                hidden_dim=hidden,
                num_message_passing_steps=cfg.num_message_passing_steps,
            ),
            edge_dim=hidden,
        )
        self.node_dec = MLP(hidden, [hidden, hidden], cfg.out_dim)

    def forward(self, g: GraphBatch) -> torch.Tensor:
        """
        Returns:
          pred: [num_nodes, out_dim] (e.g., acceleration)
        """
        h = self.node_enc(g.x)
        e = self.edge_enc(g.edge_attr) if g.edge_attr is not None else None
        h = self.processor(h, g.edge_index, e)
        return self.node_dec(h)
