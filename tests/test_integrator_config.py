from gns.config import Config, apply_overrides
from gns.factory import build_integrator
from gns.rollout.integrators import ExplicitEuler, SemiImplicitEuler


def test_build_integrator_default_is_semi_implicit():
    cfg = Config()
    integrator = build_integrator(cfg)
    assert isinstance(integrator, SemiImplicitEuler)


def test_build_integrator_explicit_override():
    cfg = apply_overrides(Config(), ["integrator.kind=explicit_euler"])
    integrator = build_integrator(cfg)
    assert isinstance(integrator, ExplicitEuler)
