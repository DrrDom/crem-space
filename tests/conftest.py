import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TINY = os.path.join(HERE, "..", "examples", "tiny")
TINY_DB = os.path.join(TINY, "tiny.db")


@pytest.fixture(scope="session")
def tiny_db():
    return TINY_DB


@pytest.fixture(scope="session")
def tiny_slice(tmp_path_factory):
    """slice of the tiny example database (radius 5, grow + make_cycle, seeds up to 18 heavy atoms)"""
    from cremspace.prepare import prepare
    out = tmp_path_factory.mktemp("slice") / "tiny_r5.db"
    prepare(TINY_DB, 5, str(out), hac_max=18, ncpu=2)
    return str(out)
