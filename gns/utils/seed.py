"""
Seeding helpers.

Determinism note:
- Full GPU determinism can be tricky depending on ops (atomic adds, etc.).
- Tests enforce determinism on CPU only by default.
"""

from __future__ import annotations

import os
import random
import warnings

import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    _safe_cuda_manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _safe_cuda_manual_seed_all(seed: int) -> None:
    # Some CPU-only environments with CUDA builds emit initialization warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization: .*")
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            return
