"""
Tiny DDP helpers.

Assumes torchrun-style environment variables:
- RANK, WORLD_SIZE, LOCAL_RANK
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0


def init_distributed() -> DistInfo:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistInfo(enabled=False)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return DistInfo(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def is_rank0(info: DistInfo) -> bool:
    return (not info.enabled) or info.rank == 0


def barrier_if_distributed(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()


def cleanup_distributed(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()
