"""The sampler's edges are CReM's: forward products equal crem.grow_mol / crem.make_cycle on the same database,
every forward edge has exactly one matching in-edge (the bijection the Metropolis ratio relies on), and the
composition-key pre-filter of make_cycle never drops a pair that has an env in the database."""
import random
from collections import Counter
from itertools import combinations

import pytest
from rdkit import Chem

from crem.crem import grow_mol, make_cycle

from cremspace.space import open_space
from cremspace.space import cycle as cl
from cremspace.space.grow import GrowSpace

N_MOLS = 25
HAC_MAX = 18


def walk_molecules(G, n, seed, steps=5):
    rng = random.Random(seed)
    G.rng = rng
    mols = []
    while len(mols) < n:
        m = G.random_seed()
        for _ in range(rng.randrange(0, steps)):
            c = G.sample_child(m)
            if c is None:
                break
            m = c
        mols.append(m)
    return mols


@pytest.fixture(scope="module")
def grow(tiny_slice):
    return open_space(tiny_slice, "example", "grow", hac_max=HAC_MAX)


@pytest.fixture(scope="module")
def cycle(tiny_slice):
    return open_space(tiny_slice, "example", "cycle", hac_max=HAC_MAX)


@pytest.mark.parametrize("db_kind", ["source", "slice"])
def test_grow_products_equal_crem(grow, tiny_db, tiny_slice, db_kind):
    db = tiny_db if db_kind == "source" else tiny_slice
    for m in walk_molecules(grow, N_MOLS, 1):
        k = min(grow.max_atoms, HAC_MAX - grow.hac(m))
        ref = set()
        if k >= 1:
            ref = {Chem.CanonSmiles(s) for s in grow_mol(Chem.AddHs(Chem.MolFromSmiles(m)), db, radius=5,
                                                         min_atoms=1, max_atoms=k, return_mol=False,
                                                         set_names=["example"])}
        assert set(grow.children(m)) == ref, m


def test_cycle_products_equal_crem(cycle, tiny_db):
    for m in walk_molecules(cycle, N_MOLS, 2, steps=6):
        k = min(cycle.max_atoms, HAC_MAX - cycle.hac(m))
        cycle._cyc.pop(m, None)
        got = set(cycle.cycle_children(m))
        ref = set()
        if k >= 1:
            ref = {Chem.CanonSmiles(s) for s in make_cycle(Chem.MolFromSmiles(m), tiny_db, radius=5,
                                                           ring_closures=True, spiro=False, min_atoms=1,
                                                           max_atoms=k, set_names=["example"])}
        assert got == ref, m


def test_grow_bijection(grow):
    for m in walk_molecules(grow, N_MOLS, 3):
        for c, mult in Counter(grow.children(m)).items():
            assert grow.cut_parents(c).count(m) == mult, (m, c)


def test_cycle_bijection(cycle):
    for m in walk_molecules(cycle, N_MOLS, 4, steps=6):
        for c, mult in Counter(cycle.children(m)).items():
            back = GrowSpace.cut_parents(cycle, c).count(m) + sum(mm for p, mm in cycle.cycle_cuts(c) if p == m)
            assert back == mult, (m, c)


def test_cycle_keys(cycle):
    for m in walk_molecules(cycle, N_MOLS, 5, steps=6):
        mol, molh = cycle._prepared(m)
        pk = cl._PairKeys(mol, cycle.radius)
        hs = [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() > 0]
        for a, b in combinations(hs, 2):
            ctx = cycle._pair_context(molh, a, b)
            env = cycle._emit(ctx)[0] if ctx is not None else None
            if env is not None:
                assert cl.env_key(env, cycle.radius) == pk.key(a, b)
        if HAC_MAX - cycle.hac(m) >= 1:
            keyed = {e for e, _ in cycle.cycle_classes(m)[0]}
            full = {e for e, _ in cycle.cycle_classes(m, keyed=False)[0]
                    if (eid := cycle.cenv_id(e)) is not None and cycle.cenv_cores(eid)[1][cycle.max_atoms] > 0}
            assert full <= keyed


def test_degree_by_bin_sums(cycle):
    for m in walk_molecules(cycle, 10, 6, steps=6):
        assert sum(cycle.d_plus_by_bin(m).values()) == cycle.d_plus(m)
        assert sum(cycle.d_minus_by_bin(m).values()) == cycle.d_minus(m)
