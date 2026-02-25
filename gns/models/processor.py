"""
Message passing core(s).

This file is deliberately the main "research knob" for GNS-style work:
- swap MPNN blocks, attention, equivariant operators, edge updates, norms, etc.
- keep interfaces small and the dataflow obvious.

Current implementation:
- Weight-shared MPNN repeated for `num_message_passing_steps`.
- Message:  m_ij = f_msg([h_i, h_j, e_ij])
- Aggregate to dst: agg_j = sum_{i->j} m_ij
- Update:   h_j = LN(h_j + f_upd([h_j, agg_j]))

Notes on stability:
- LayerNorm after each residual update is a simple, effective default for longer rollouts.
- Summation aggregation is standard; replace with mean/max if needed.

=== RESEARCH KNOBS ===
- Change message function inputs (remove h_dst, add relative vel, node types, globals).
- Change aggregation (mean, attention-weighted, normalize by degree).
- Add edge latent updates (phi_e) and return updated edges (would require interface change).
- Add dropout/gating/residual scaling for stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from gns.nn.mlp import MLP


@dataclass(frozen=True)
class ProcessorConfig:
    hidden_dim: int = 128
    num_message_passing_steps: int = 3


class MPNNProcessor(nn.Module):
    """
    Minimal, hackable MPNN processor with:
    - sum aggregation
    - residual updates
    - LayerNorm for stability
    """

    def __init__(self, cfg: ProcessorConfig, edge_dim: int):
        super().__init__()
        H = int(cfg.hidden_dim)
        self.steps = int(cfg.num_message_passing_steps)

        # === RESEARCH KNOB === activation / depth / width
        self.msg = MLP(in_dim=2 * H + edge_dim, hidden_dims=[H, H], out_dim=H, act="relu")
        self.upd = MLP(in_dim=2 * H, hidden_dims=[H, H], out_dim=H, act="relu")

        # Simple stabilization for longer horizons.
        self.norm = nn.LayerNorm(H)

        # === RESEARCH KNOB === edge dropout (set >0 to regularize)
        self.edge_dropout_p: float = 0.0

    def forward(
        self,
        h: torch.Tensor,  # [N, H]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: Optional[torch.Tensor],  # [E, edge_dim] or None
    ) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError(
                "MPNNProcessor requires edge_attr (provide at least relative pos, encoded)."
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")
        if h.dim() != 2:
            raise ValueError(f"h must be [N, H], got {tuple(h.shape)}")

        src_all = edge_index[0].to(torch.long)
        dst_all = edge_index[1].to(torch.long)

        N = int(h.size(0))
        H = int(h.size(1))
        E = int(src_all.numel())

        # Reuse aggregation buffer across steps to avoid reallocations.
        agg = h.new_zeros((N, H))

        for _ in range(self.steps):
            if E == 0:
                agg.zero_()
            else:
                # === RESEARCH KNOB === edge dropout / subsampling
                if self.training and self.edge_dropout_p > 0.0:
                    keep = torch.rand(E, device=h.device) >= self.edge_dropout_p
                    src = src_all[keep]
                    dst = dst_all[keep]
                    e = edge_attr[keep]
                else:
                    src = src_all
                    dst = dst_all
                    e = edge_attr

                m_in = torch.cat([h[src], h[dst], e], dim=-1)
                m = self.msg(m_in)  # [E, H]

                agg.zero_()
                # Sum messages into destination nodes.
                agg.index_add_(0, dst, m)

            upd = self.upd(torch.cat([h, agg], dim=-1))
            h = self.norm(h + upd)

        return h
