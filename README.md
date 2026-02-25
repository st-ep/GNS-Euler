# GNS Euler Skeleton

Compact, hackable PyTorch GNS-style rollout scaffold focused on fast research iteration.

## Quick start

Create environment (venv):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

Create environment (Conda):

```bash
conda env create -f environment.yml
conda activate gns-euler
```

Train:

```bash
python -m gns train --config configs/base.json
```

Train with overrides:

```bash
python -m gns train --config configs/base.json --set train.lr=1e-4 --set graph.radius=0.2 --set loss.kind=huber --set integrator.kind=explicit_euler --set graph.cache_edges=true --set graph.edge_rebuild_interval=0 --set norm.enabled=true --set loss.rollout_weight=0.5 --set loss.rollout_horizon=3
```

Resume training from a checkpoint:

```bash
python -m gns train --config configs/base.json --resume runs/<run_id>/checkpoints/step_0000100.pt
```

Eval:

```bash
python -m gns eval --config configs/base.json
```

Eval a checkpoint:

```bash
python -m gns eval --config configs/base.json --checkpoint runs/<run_id>/checkpoints/step_0000100.pt
```

Eval notes:
- eval now aggregates rollout metrics across the eval dataloader (not just one sample)
- set `eval.max_batches=0` to use the full loader, or a positive integer to cap eval cost
- with `eval.save_json=true`, eval writes `eval_metrics*.json` in the run directory
- with `eval.save_csv=true`, eval also writes `eval_metrics*.csv`
- with `eval.save_predictions=true`, eval writes `eval_rollout*.npz` containing `pred`, `true`, `dt`, `state_kind`
- with `eval.save_plots=true` (NS2D only), eval attempts to render grid videos via `data/ns2d/visualize_trajectory.py`

## NS2D configs

Quick smoke run on bundled NS2D trajectories:

```bash
python -m gns train --config configs/ns2d_smoke.json
python -m gns eval --config configs/ns2d_smoke.json
```

Longer training profile:

```bash
python -m gns train --config configs/ns2d_full.json
```

DDP single-node:

```bash
torchrun --nproc_per_node=4 -m gns train --config configs/base.json
```

## Primary hack points

- `gns/graphs/*` neighbor search and edge construction
- `gns/models/*` encode/process/decode and message passing
- `gns/rollout/*` integration and rollout behavior
- `gns/train/losses.py` objectives
- `gns/nn/normalizer.py` normalization strategy
