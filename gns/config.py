"""
Dataclass-based config with JSON load + CLI overrides.

Why not Hydra?
- Hydra is great for large sweeps, but adds heavier dependency + structure.
- This repo prioritizes compact diffs and fewer moving parts.

Usage:
  cfg = load_config("configs/base.json")
  cfg = apply_overrides(cfg, ["train.lr=1e-4", "graph.radius=0.2"])
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import json


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "dummy"
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True
    root_dir: str = "data/ns2d/ns2d_traj"
    split: str = "train"
    eval_split: str = "test"
    frame_stride: int = 1
    spatial_stride: int = 1
    max_samples: int = 0
    eval_max_samples: int = 0
    seq_len: int = 8
    num_nodes: int = 16
    dim: int = 2
    dt: float = 0.01
    grid_n: int = 0
    domain_L: float = 1.0


@dataclass(frozen=True)
class GraphConfig:
    builder: str = "radius"
    radius: float = 0.5
    max_neighbors: int = 32
    cache_edges: bool = False
    edge_rebuild_interval: int = 1


@dataclass(frozen=True)
class ModelConfig:
    node_in_dim: int = 4
    edge_in_dim: int = 3
    hidden_dim: int = 128
    num_message_passing_steps: int = 3
    out_dim: int = 2


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 100
    ckpt_every: int = 100
    amp: bool = True
    noise_std: float = 0.0


@dataclass(frozen=True)
class LossConfig:
    kind: str = "mse"
    huber_delta: float = 1.0
    rollout_weight: float = 0.0
    rollout_horizon: int = 0


@dataclass(frozen=True)
class NormConfig:
    enabled: bool = False
    update_steps: int = 0
    freeze_after: int = 0


@dataclass(frozen=True)
class IntegratorConfig:
    kind: str = "semi_implicit_euler"


@dataclass(frozen=True)
class EvalConfig:
    rollout_horizon: int = 5
    max_batches: int = 0
    save_json: bool = True
    save_csv: bool = True
    save_predictions: bool = True
    prediction_samples: int = 8
    save_plots: bool = False
    plot_format: str = "gif"
    plot_fps: int = 10


@dataclass(frozen=True)
class Config:
    seed: int = 123
    device: str = "cuda"
    dtype: str = "float32"
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _from_dict(raw)


def save_config(cfg: Config, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """
    Overrides are dot.path=value, e.g. train.lr=1e-4.

    Minimal implementation:
    - supports only scalar JSON-ish values
    - TODO: support lists/dicts if you need them
    """

    d = asdict(cfg)
    for item in overrides:
        key, value = _split_override(item)
        _set_by_dotpath(d, key, _parse_scalar(value))
    return _from_dict(d)


# ----------------- internal helpers -----------------

def _from_dict(d: dict[str, Any]) -> Config:
    data = DataConfig(**d.get("data", {}))
    graph = GraphConfig(**d.get("graph", {}))
    model = ModelConfig(**d.get("model", {}))
    train = TrainConfig(**d.get("train", {}))
    loss = LossConfig(**d.get("loss", {}))
    norm = NormConfig(**d.get("norm", {}))
    integrator = IntegratorConfig(**d.get("integrator", {}))
    eval_ = EvalConfig(**d.get("eval", {}))
    return Config(
        seed=d.get("seed", 123),
        device=d.get("device", "cuda"),
        dtype=d.get("dtype", "float32"),
        data=data,
        graph=graph,
        model=model,
        train=train,
        loss=loss,
        norm=norm,
        integrator=integrator,
        eval=eval_,
    )


def _split_override(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise ValueError(f"Override must be key=value, got: {s}")
    key, value = s.split("=", 1)
    return key.strip(), value.strip()


def _set_by_dotpath(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = d
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _parse_scalar(v: str) -> Any:
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"

    try:
        if v.startswith("0") and len(v) > 1 and v[1].isdigit():
            raise ValueError
        return int(v)
    except ValueError:
        pass

    try:
        return float(v)
    except ValueError:
        pass

    return v.strip('"\'')
