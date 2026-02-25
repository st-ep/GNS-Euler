"""
GNS-style rollout skeleton.

Primary hack points:
- gns/graphs/*        (neighbor search + edge building)
- gns/models/*        (encode/process/decode + message passing)
- gns/rollout/*       (integrator + rollout scheme)
- gns/train/losses.py (training objectives)
- gns/nn/normalizer.py (normalization)
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
