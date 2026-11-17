import pytest

from recsys.data.loaders import load_retailrocket
from recsys.data.synthetic import generate


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("rr") / "raw")


@pytest.fixture(scope="session")
def raw(raw_dir):
    return load_retailrocket(raw_dir)
