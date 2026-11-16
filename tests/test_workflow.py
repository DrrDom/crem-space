"""Exact enumeration of the tiny space matches the stored counts; a short sampling run reproduces them."""
import json
import math
import os

import pytest

from cremspace.exact import enumerate_space

from conftest import TINY


def expected(mode):
    with open(os.path.join(TINY, f"exact_{mode}_h18.json")) as f:
        return {int(k): v for k, v in json.load(f)["counts"].items()}


@pytest.mark.parametrize("mode", ["grow", "cycle"])
def test_exact_small_cap(tiny_slice, mode):
    """the space at cap 12 is the part of the cap-18 space with <= 12 heavy atoms (edges only add atoms)"""
    r = enumerate_space(tiny_slice, "example", mode, hac_max=12, ncpu=2)
    exp = expected(mode)
    assert r["counts"] == {h: exp[h] for h in range(1, 13)}


@pytest.mark.slow
def test_sample_diagnose_table(tiny_slice, tmp_path):
    from cremspace.diagnose import diagnose
    from cremspace.sampler import sample
    from cremspace.table import table

    run = tmp_path / "run"
    rc = sample(tiny_slice, ["example"], "grow", str(run), walkers=4, hours=0.1, hac_max=16, copy="none")
    assert rc == 0
    diagnose([str(run)], ncpu=1, exact=[os.path.join(TINY, "exact_grow_h18.json")])
    with open(run / "example" / "estimate.json") as f:
        res = json.load(f)
    est = res["estimate"]
    exact = math.log10(sum(v for h, v in expected("grow").items() if h <= 16))
    assert abs(est["total"] - exact) < 0.1
    with open(tmp_path / "t.tsv", "w") as fh:
        table([str(run)], per_hac=True, out=fh)
    rows = open(tmp_path / "t.tsv").read().splitlines()
    assert len(rows) == 2 and rows[1].split("\t")[1] == "example"
