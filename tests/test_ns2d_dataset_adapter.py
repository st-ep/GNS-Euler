from gns.config import Config, apply_overrides
from gns.factory import build_dataset


def test_ns2d_dataset_adapter_shapes_and_dt():
    cfg = apply_overrides(
        Config(device="cpu"),
        [
            "data.dataset=ns2d",
            "data.root_dir=data/ns2d/ns2d_traj",
            "data.split=train",
            "data.seq_len=12",
            "data.frame_stride=1",
            "data.spatial_stride=8",
            "data.max_samples=3",
            "data.dt=0.0",
        ],
    )
    ds = build_dataset(cfg, split="train")
    sample = ds[0]

    assert sample.pos.shape == (1, 12, 64, 2)
    assert sample.vel.shape == (1, 12, 64, 1)
    assert abs(sample.dt - 0.02) < 1e-8
