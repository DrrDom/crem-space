"""The slice keeps exactly the rows grow_mol / make_cycle can use, and its packed tables agree with them."""
import sqlite3

import pytest

from cremspace.prepare import prepare
from cremspace.util import read_meta


def test_meta(tiny_slice, tiny_db):
    m = read_meta(tiny_slice)
    assert m["format"] == "cremspace-slice"
    assert m["radius"] == "5" and m["sets"] == "example" and m["modes"] == "grow,cycle"
    assert m["max_atoms"] == "10" and m["hac_max"] == "18" and m["min_freq"] == "1"


def test_rows_match_source(tiny_slice, tiny_db):
    src = sqlite3.connect(tiny_db)
    sl = sqlite3.connect(tiny_slice)
    q_grow = ("SELECT r.env_id, r.core_smi_id FROM radius5 r JOIN frags f USING(core_smi_id) "
              "WHERE r.dist2 = 0 AND r.core_num_atoms BETWEEN 1 AND 10 AND instr(f.core_smi, ':2]') = 0 "
              "AND r.example >= 1")
    q_ring = ("SELECT r.env_id, r.core_smi_id FROM radius5 r JOIN frags f USING(core_smi_id) "
              "WHERE r.core_num_atoms BETWEEN 1 AND 10 AND instr(f.core_smi, '[1*:1]') > 0 "
              "AND instr(f.core_smi, '[1*:2]') > 0 AND instr(f.core_smi, ':3]') = 0 AND r.example >= 1")
    want = set(src.execute(q_grow).fetchall()) | set(src.execute(q_ring).fetchall())
    got = set(sl.execute("SELECT env_id, core_smi_id FROM radius5").fetchall())
    assert got == want
    # packed grow cores = the grow rows, per env and cumulative by size
    n_pq = sl.execute("SELECT count(*) FROM pq_0").fetchone()[0]
    assert n_pq == len(set(src.execute(q_grow).fetchall()))
    n_cq = sl.execute("SELECT count(*) FROM cq_0").fetchone()[0]
    assert n_cq == len(set(src.execute(q_ring).fetchall()))
    for eid, start, *cum in sl.execute("SELECT * FROM pe_0 LIMIT 200"):
        sizes = [n for (n,) in sl.execute(
            "SELECT r.core_num_atoms FROM pq_0 p JOIN radius5 r ON r.core_smi_id = p.core_smi_id AND r.env_id = ? "
            "WHERE p.rid >= ? AND p.rid < ? ORDER BY p.rid", (eid, start, start + cum[-1]))]
        assert sizes == sorted(sizes) and len(sizes) == cum[-1]
        assert [sum(s <= k for s in sizes) for k in range(1, 11)] == cum


def test_min_freq(tiny_db, tmp_path):
    out = tmp_path / "s.db"
    prepare(tiny_db, 5, str(out), hac_max=18, min_freq=2, cycle=False, ncpu=1)
    sl = sqlite3.connect(str(out))
    assert sl.execute("SELECT min(example) FROM radius5").fetchone()[0] >= 2
    assert read_meta(str(out))["modes"] == "grow"


def test_refuses_bad_input(tiny_db, tmp_path):
    with pytest.raises(ValueError):
        prepare(tiny_db, 3, str(tmp_path / "x.db"))           # no radius3 table
    with pytest.raises(ValueError):
        prepare(tiny_db, 5, str(tmp_path / "x.db"), sets=["nope"])
