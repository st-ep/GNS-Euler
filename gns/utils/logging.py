"""
Minimal logging:
- console printing
- JSONL metrics file
- optional TensorBoard if installed

Keep it dead simple for clean diffs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Loggers:
    run_dir: str
    jsonl_path: str
    tb: Any | None = None


def setup_loggers(run_dir: str, enable_tb: bool = True) -> Loggers:
    os.makedirs(run_dir, exist_ok=True)
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")

    tb = None
    if enable_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            tb = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
        except Exception:
            tb = None

    return Loggers(run_dir=run_dir, jsonl_path=jsonl_path, tb=tb)


def log_metrics(loggers: Loggers, step: int, metrics: dict[str, float]) -> None:
    rec = {"step": step, "time": time.time(), **metrics}
    msg = " ".join([f"{k}={v:.6g}" for k, v in metrics.items()])
    print(f"[step {step}] {msg}")

    with open(loggers.jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    if loggers.tb is not None:
        for k, v in metrics.items():
            loggers.tb.add_scalar(k, v, step)
