import pytest

from gns.utils.seed import set_seed


@pytest.fixture(autouse=True)
def _seed_every_test():
    set_seed(123, deterministic=False)
    yield
