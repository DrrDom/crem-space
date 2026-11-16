"""Grow + make_cycle space: GrowSpace (grow.py) extended by the ring-closure edges of crem.make_cycle.

make_cycle(mol, db, radius=R, ring_closures=True, spiro=False, min_atoms=1, max_atoms=K) joins two
distinct H-bearing atoms a, b of P by a 2-attachment ring-arc core c from the DB: both H are cut,
the context (P with two ring-cut-labelled dummies) gives an env, and every core of that env makes
the product P+c@(a,b), which closes a new ring. Like grow it only adds atoms (HAC(P) + #atoms(c)),
so the walk, the root, the bins and the estimators of the walker carry over unchanged once D+ and D-
count both edge types:

    D+(P) = D+_grow(P) + #cycle edges of P
    D-(M) = #grow cut classes + sum over ring-arc cut classes of their multiplicity  (+1 if seed)

Forward (cycle edges of P). For every anchor-pair class (one per canonical context, exactly as
crem's __fragment_mol_ring_closure dedupes them) and every core row of its env with count >= min_freq in the set and
<= min(K, HAC_MAX - HAC(P)) atoms, the product is built with crem's own __frag_replace -- so the
ring-geometry filter of make_cycle (discard_ring_geometry=True) applies -- and only built products
count. Cycle products per molecule are few at radius 5, so they are materialised and cached;
D+ and the transition-matrix counts are then exact.

Most anchor pairs have no env in the DB, and crem's env canonicalisation (~4 ms per pair) would
dominate. A pair is therefore first tested against a composition key that both an env string and a
pair (a, b) of P determine (pair_key / env_key): the heavy atoms within R-1 bonds of a or b counted
by (element, aromatic, charge), the number of boundary dummies (atoms exactly R bonds away) and the
a-b distance if <= 9 (beyond that the two balls are disconnected in the env). Keys of all ring-arc
envs of the slice are stored in table cinv with a set mask; a pair whose key is absent cannot have
a DB env. tests/test_spaces.py checks key(env of the pair) == pair_key on every pair it sees.

Backward (ring-arc cut classes of M). The DB rows came from cutting two non-aromatic single bonds of
one SSSR ring that split the molecule in two, either side being the core (ring_fragments.py), so a
cut class of M is such a bond pair with a chosen core side (1..K atoms), up to automorphism; it is
DB-supported if its (env, core) is a row of the set. Multiplicity: the DB stores every labelling of
an asymmetric core that keeps the env canonical. If the full context has an automorphism swapping
its two attachment points, both labellings rebuild M from the same parent class, so that cut is
reached by two forward edges; otherwise by one. The cut must also survive the geometry filter on M
with its core atoms marked as inserted (the filter is a function of the product and the inserted
atoms only), and be forward-visible: the parent's own pair context at the two anchors must give the
same env (guards against SMILES round-trip artefacts, as GrowSpace.parents does for grow).
"""
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import rdMolTransforms  # noqa: F401  (crem's geometry helpers import lazily)

import crem.crem as _cc
from crem.mol_context import get_canon_context_core, get_std_context_core_permutations
from crem.ring_fragments import RING_CUT_DUMMY_ISOTOPE

from cremspace.space.grow import GrowSpace, ROOT
from cremspace.util import LRU

# crem's module-private helpers, used so that forward products and filters are make_cycle's own
_frag_replace = getattr(_cc, "__frag_replace")
_standardize_context_mol = getattr(_cc, "__standardize_context_mol")
_backup_atom_properties = getattr(_cc, "__backup_atom_properties")
_props_to_backup = getattr(_cc, "__atom_properties_to_backup")
_identify_new_ring = getattr(_cc, "__identify_new_ring")
_ring_topology_ok = getattr(_cc, "__ring_topology_ok")
_ring_reach_ok = getattr(_cc, "__ring_reach_ok")
ATOM_INDEX_PROP = _cc.ATOM_INDEX_PROP
CREM_MARKER_PROP = _cc.CREM_MARKER_PROP

FAKE_CORE = "[1*:1]C[1*:2]"          # crem's stand-in core for a labelled two-point context
H2_CORE = "[H][*:1].[H][*:2]"         # the "old fragment" of a ring closure
FAR = 99                              # anchor distance bucket: env balls disconnected


def _popcount(v):
    return bin(v).count("1")


def _key(dd, n_boundary, counts):
    return f"{dd}|{n_boundary}|" + ",".join(f"{t}x{n}" for t, n in sorted(counts.items()))


def _atype(a):
    return f"{a.GetAtomicNum()}{'a' if a.GetIsAromatic() else ''}{a.GetFormalCharge():+d}"


def env_key(env, radius=5):
    """composition key of a labelled two-point env string (see module doc)"""
    m = Chem.MolFromSmiles(env, sanitize=False)
    aps, nb, counts = [], 0, {}
    for a in m.GetAtoms():
        z = a.GetAtomicNum()
        if z == 0:
            if a.GetAtomMapNum():
                aps.append(a.GetIdx())
            else:
                nb += 1
        elif z > 1:
            t = _atype(a)
            counts[t] = counts.get(t, 0) + 1
    if len(aps) != 2:
        return None
    # BFS distance between the two attachment dummies
    dist, front, seen = 0, {aps[0]}, {aps[0]}
    dd = FAR
    while front:
        if aps[1] in front:
            dd = dist - 2
            break
        nxt = set()
        for i in front:
            for n in m.GetAtomWithIdx(i).GetNeighbors():
                if n.GetIdx() not in seen:
                    seen.add(n.GetIdx())
                    nxt.add(n.GetIdx())
        front, dist = nxt, dist + 1
    if dd > 9:
        dd = FAR
    return _key(dd, nb, counts)


class _PairKeys:
    """composition keys of all anchor pairs of one molecule, from its distance matrix"""

    def __init__(self, mol, radius):
        n = mol.GetNumAtoms()
        D = Chem.GetDistanceMatrix(mol)
        self.D = D
        r = radius - 1                       # heavy atoms within R bonds of the dummy
        self.ball = [sum(1 << v for v in range(n) if D[u][v] <= r) for u in range(n)]
        self.shell = [sum(1 << v for v in range(n) if D[u][v] == r + 1) for u in range(n)]
        types = {}
        for a in mol.GetAtoms():
            if a.GetAtomicNum() > 1:
                t = _atype(a)
                types[t] = types.get(t, 0) | (1 << a.GetIdx())
        self.types = types

    def key(self, a, b):
        S = self.ball[a] | self.ball[b]
        boundary = (self.shell[a] | self.shell[b]) & ~S
        counts = {t: c for t, msk in self.types.items() if (c := _popcount(S & msk))}
        d = int(self.D[a][b])
        return _key(d if d <= 9 else FAR, _popcount(boundary), counts)


def geometry_ok(p):
    """make_cycle's discard_ring_geometry test on product p whose inserted atoms carry the crem
    marker -- the exact code path of crem's __frag_replace, including its fail-open except"""
    try:
        ringinfo = _identify_new_ring(p)
        if ringinfo is not None and (not _ring_topology_ok(p, ringinfo) or not _ring_reach_ok(p, ringinfo)):
            return False
    except Exception:
        pass
    return True


def _canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


class GrowCycleSpace(GrowSpace):
    def __init__(self, db, set_name, cycle_cache_size=50_000, **kw):
        super().__init__(db, set_name, **kw)
        names = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        si = self.set_index
        self.ce, self.cq = f"ce_{si}", f"cq_{si}"
        if "cycle" not in self.meta.get("modes", "").split(",") or not {"cinv", self.ce, self.cq} <= names:
            raise ValueError("the slice has no make_cycle rows (it was prepared with --no-cycle)")
        self._cenv_cores = LRU(cycle_cache_size)
        self._cyc = LRU(cycle_cache_size)        # smi -> list of cycle products (with multiplicity)
        self._ccuts = LRU(cycle_cache_size)      # smi -> [(parent, multiplicity)]
        self._pair_env = LRU(cycle_cache_size)   # parent SMILES with mapped anchors -> env
        self._ckey = LRU(4 * cycle_cache_size)   # composition key -> set mask
        for k in ("cycle_pairs", "cycle_pairs_keyed", "cycle_envs", "cycle_products",
                  "cycle_geometry_rejected", "cycle_cuts", "cycle_cut_mult2", "cycle_reverse_only"):
            self.stats[k] = 0

    # ------------------------------------------------------------------ DB lookups (cycle)
    # ring-arc rows, their envs and cores live in the slice's radius/envs/frags tables next to the grow rows
    def cenv_id(self, env):
        return self.env_id(env)

    def ccore_smi(self, cid):
        return self.core_smi(cid)

    def ccore_id(self, smi):
        return self.core_id(smi)

    def has_cpair(self, eid, cid):
        return self.has_pair(eid, cid)

    def cenv_cores(self, eid):
        """(start rid in cq, cum) for the ring-arc cores of this set at env eid"""
        return self._packed(self._cenv_cores, self.ce, eid)

    def ccore_range(self, eid, c):
        h, _ = self.cenv_cores(eid)
        return [r[0] for r in self.con.execute(
            f"SELECT core_smi_id FROM {self.cq} WHERE rid >= ? AND rid < ? ORDER BY rid", (h, h + c))]

    def keys_present(self, keys):
        """subset of composition keys that some ring-arc env of this set has"""
        out, todo = set(), []
        for k in keys:
            m = self._ckey.get(k)
            if m is None:
                todo.append(k)
            elif m & self.bit:
                out.add(k)
        for i in range(0, len(todo), 500):
            chunk = todo[i:i + 500]
            found = dict(self.con.execute(
                f"SELECT key, mask FROM cinv WHERE key IN ({','.join('?' * len(chunk))})", chunk))
            for k in chunk:
                m = found.get(k, 0)
                self._ckey.put(k, m)
                if m & self.bit:
                    out.add(k)
        return out

    # ------------------------------------------------------------------ forward (make_cycle)
    @staticmethod
    def _pair_context(molh, a, b):
        """molh (explicit H) with one H on a and one on b turned into dummies [*:1], [*:2] -- the
        shape of an MMPA two-H cut core, built in place like crem's same-atom contexts"""
        rw = Chem.RWMol(molh)
        for map_num, anchor in ((1, a), (2, b)):
            h = next(n.GetIdx() for n in rw.GetAtomWithIdx(anchor).GetNeighbors()
                     if n.GetAtomicNum() == 1 and n.GetAtomMapNum() == 0)
            d = rw.GetAtomWithIdx(h)
            d.SetAtomicNum(0)
            d.SetIsotope(0)
            d.SetNoImplicit(True)
            d.SetNumExplicitHs(0)
            d.SetAtomMapNum(map_num)
        ctx = rw.GetMol()
        if Chem.SanitizeMol(ctx, catchErrors=True):
            return None
        return ctx

    def _emit(self, ctx):
        """crem's emit(): labelled variant -> (env, standardised context Mol, its SMILES)"""
        variant = Chem.Mol(ctx)
        for at in variant.GetAtoms():
            if at.GetAtomicNum() == 0 and at.GetAtomMapNum():
                at.SetIsotope(RING_CUT_DUMMY_ISOTOPE)
        env, frag, old_to_new = get_canon_context_core(variant, FAKE_CORE, radius=self.radius,
                                                       keep_stereo=False, return_att_map=True,
                                                       preserve_dummy_isotopes=True)
        if env is None or not frag:
            return None, None, None
        context_smi, std = _standardize_context_mol(variant, old_to_new)
        return env, std, context_smi

    def _prepared(self, smi):
        """(mol, molh) as make_cycle prepares its input: explicit H, atom indices, backups"""
        mol = Chem.MolFromSmiles(smi)
        molh = _backup_atom_properties(Chem.AddHs(mol), _props_to_backup)
        for at in molh.GetAtoms():
            at.SetIntProp(ATOM_INDEX_PROP, at.GetIdx())
        return mol, molh

    def cycle_classes(self, smi, keyed=True):
        """[(env, std context Mol)] one per anchor-pair class with an env in the set's ring-arc
        rows (keyed=False skips the composition pre-filter -- validation only)"""
        mol, molh = self._prepared(smi)
        hs = [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() > 0]
        pairs = list(combinations(hs, 2))
        self.stats["cycle_pairs"] += len(pairs)
        if keyed and pairs:
            pk = _PairKeys(mol, self.radius)
            keys = {pr: pk.key(*pr) for pr in pairs}
            ok = self.keys_present(set(keys.values()))
            pairs = [pr for pr in pairs if keys[pr] in ok]
            self.stats["cycle_pairs_keyed"] += len(pairs)
        out = {}
        for a, b in pairs:
            ctx = self._pair_context(molh, a, b)
            if ctx is None:
                continue
            env, std, context_smi = self._emit(ctx)
            self.stats["cycle_envs"] += 1
            if env is None or (env, context_smi) in out:
                continue
            out[(env, context_smi)] = std
        return [(env, std) for (env, _), std in out.items()], molh

    def cycle_children(self, smi):
        """products of all cycle edges of smi (with multiplicity), as canonical SMILES"""
        if smi == ROOT:
            return []
        v = self._cyc.get(smi)
        if v is not None:
            return v
        hac = Chem.MolFromSmiles(smi).GetNumHeavyAtoms()
        k = min(self.max_atoms, self.hac_max - hac)
        v = []
        if k >= 1:
            classes, molh = self.cycle_classes(smi)
            for env, std in classes:
                eid = self.cenv_id(env)
                if eid is None:
                    continue
                c = self.cenv_cores(eid)[1][k]
                for cid in (self.ccore_range(eid, c) if c else ()):
                    prods = list(_frag_replace(molh, None, H2_CORE, self.ccore_smi(cid), self.radius, std,
                                               discard_ring_geometry=True, clear_index_props=True))
                    if not prods:
                        self.stats["cycle_geometry_rejected"] += 1
                        continue
                    s2 = _canon(prods[0][0])
                    if s2 is None:
                        self.stats["unstable_smiles"] += 1
                        continue
                    if s2 != prods[0][0]:
                        self.stats["unstable_smiles"] += 1
                    if s2 != smi:
                        v.append(s2)
            self.stats["cycle_products"] += len(v)
        self._cyc.put(smi, v)
        return v

    def d_plus(self, smi):
        if smi == ROOT:
            return len(self.seed_ids)
        return super().d_plus(smi) + len(self.cycle_children(smi))

    def sample_child(self, smi):
        if smi == ROOT:
            return self.random_seed()
        ng = super().d_plus(smi)
        cyc = self.cycle_children(smi)
        tot = ng + len(cyc)
        if tot == 0:
            return None
        r = self.rng.randrange(tot)
        if r < ng:
            return super().sample_child(smi)     # uniform over grow edges, i.e. this branch's share
        return cyc[r - ng]

    def children(self, smi):
        if smi == ROOT:
            return super().children(smi)
        return super().children(smi) + list(self.cycle_children(smi))

    def d_plus_by_bin(self, smi):
        out = super().d_plus_by_bin(smi)
        for c in (self.cycle_children(smi) if smi != ROOT else ()):
            h = self.hac(c)
            out[h] = out.get(h, 0) + 1
        return out

    # ------------------------------------------------------------------ backward (ring-arc cuts)
    def _forward_env(self, parent_mapped):
        """env of the parent's own pair context at the two anchors mapped 1 and 2 in parent_mapped"""
        v = self._pair_env.get(parent_mapped)
        if v is None:
            pm = Chem.MolFromSmiles(parent_mapped)
            v = ""
            if pm is not None:
                anchors = {a.GetAtomMapNum(): a.GetIdx() for a in pm.GetAtoms() if a.GetAtomMapNum()}
                for a in pm.GetAtoms():
                    a.SetAtomMapNum(0)
                # AddHs appends the H atoms, so heavy-atom indices (the anchors) are kept
                molh = _backup_atom_properties(Chem.AddHs(pm), _props_to_backup)
                for at in molh.GetAtoms():
                    at.SetIntProp(ATOM_INDEX_PROP, at.GetIdx())
                ctx = self._pair_context(molh, anchors[1], anchors[2]) if len(anchors) == 2 else None
                if ctx is not None:
                    env, _, _ = self._emit(ctx)
                    v = env or ""
            self._pair_env.put(parent_mapped, v)
        return v

    def cycle_cuts(self, smi):
        """[(parent, multiplicity)] over DB-supported, geometry-valid, forward-visible ring-arc cut
        classes of smi (not yet filtered by reachability)"""
        if smi == ROOT:
            return []
        v = self._ccuts.get(smi)
        if v is not None:
            return v
        mol = Chem.MolFromSmiles(smi)
        for at in mol.GetAtoms():
            at.SetIntProp("orig_idx", at.GetIdx())
        v, seen_cls, seen_pairs = [], set(), set()
        for ring in mol.GetRingInfo().BondRings():
            for b1, b2 in combinations(sorted(ring), 2):
                if (b1, b2) in seen_pairs:
                    continue
                seen_pairs.add((b1, b2))
                bonds = [mol.GetBondWithIdx(b) for b in (b1, b2)]
                if any(b.GetBondType() != Chem.BondType.SINGLE or b.GetIsAromatic() for b in bonds):
                    continue
                frag = Chem.FragmentOnBonds(mol, [b1, b2], dummyLabels=[(1, 1), (2, 2)])
                pieces = Chem.GetMolFrags(frag, asMols=True)
                if len(pieces) != 2:
                    continue
                for ci in (0, 1):
                    core, ctx = pieces[ci], pieces[1 - ci]
                    nc = core.GetNumHeavyAtoms()
                    if not (1 <= nc <= self.max_atoms) or ctx.GetNumHeavyAtoms() < 1:
                        continue
                    cut = self._cut_class(mol, core, ctx, b1, b2)
                    if cut is None:
                        continue
                    cls, parent, mult = cut
                    if cls in seen_cls:
                        continue
                    seen_cls.add(cls)
                    if mult:
                        v.append((parent, mult))
        self.stats["cycle_cuts"] += len(v)
        self._ccuts.put(smi, v)
        return v

    def _cut_class(self, mol, core, ctx, b1, b2):
        """(class key, parent, multiplicity) of one oriented ring-arc cut; multiplicity 0 if the cut
        is not a valid reverse of a cycle edge"""
        core_idx = {a.GetIntProp("orig_idx") for a in core.GetAtoms() if a.GetAtomicNum()}
        ends = set()
        for b in (b1, b2):
            bd = mol.GetBondWithIdx(b)
            ends.add(bd.GetEndAtomIdx() if bd.GetBeginAtomIdx() in core_idx else bd.GetBeginAtomIdx())
        if len(ends) == 1:
            return None          # both ends on one context atom: a spiro closure, not made with spiro=False

        # pieces in the DB convention: dummy isotope -> map number, then the ring-cut label
        core, ctx = Chem.RWMol(core), Chem.RWMol(ctx)
        for pc in (core, ctx):
            for a in pc.GetAtoms():
                if a.GetAtomicNum() == 0:
                    a.SetAtomMapNum(a.GetIsotope())
                    a.SetIsotope(RING_CUT_DUMMY_ISOTOPE)
        core, ctx = core.GetMol(), ctx.GetMol()
        # cheap pre-filter: one of the two labellings of the core must be a ring-arc core of the
        # slice; deterministic per class, so it may run before the class key
        spell = _canon(Chem.MolToSmiles(core))
        sw = Chem.RWMol(core)
        for a in sw.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetAtomMapNum(3 - a.GetAtomMapNum())
        spell_sw = _canon(Chem.MolToSmiles(sw.GetMol()))
        if self.ccore_id(spell) is None and self.ccore_id(spell_sw) is None:
            return None

        # class key: M with each cut bond u(core)-v(context) replaced by u-[3*]-[4*]-v; invariant
        # exactly under automorphisms of M acting on the oriented bond pair
        rw = Chem.RWMol(mol)
        for b in (b1, b2):
            bd = mol.GetBondWithIdx(b)
            u, w = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
            if u not in core_idx:
                u, w = w, u
            rw.RemoveBond(u, w)
            x3 = rw.AddAtom(Chem.Atom(0))
            x4 = rw.AddAtom(Chem.Atom(0))
            rw.GetAtomWithIdx(x3).SetIsotope(3)
            rw.GetAtomWithIdx(x4).SetIsotope(4)
            rw.AddBond(u, x3, Chem.BondType.SINGLE)
            rw.AddBond(x3, x4, Chem.BondType.SINGLE)
            rw.AddBond(x4, w, Chem.BondType.SINGLE)
        cls = Chem.MolToSmiles(rw.GetMol())

        # env and the env-canonical labellings, as cremdb_create stores them
        # (cremdb_create writes both pieces to SMILES first and canonicalises from the strings)
        res = get_std_context_core_permutations(Chem.MolToSmiles(ctx), Chem.MolToSmiles(core), self.radius, False)
        if res is None or not res[0]:
            return cls, None, 0
        env, cores = res
        eid = self.cenv_id(env)
        if eid is None:
            return cls, None, 0
        rows = {s for s in {_canon(c) for c in cores} if s is not None and
                (cid := self.ccore_id(s)) is not None and self.has_cpair(eid, cid)}
        if not rows:
            return cls, None, 0
        # geometry: M with the core atoms marked as inserted, as __frag_replace sees its product
        p = Chem.Mol(mol)
        for i in core_idx:
            p.GetAtomWithIdx(i).SetBoolProp(CREM_MARKER_PROP, True)
        if not geometry_ok(p):
            return cls, None, 0
        # parent, with the two anchors mapped for the forward-visibility check
        pm = Chem.RWMol(ctx)
        for a in pm.GetAtoms():
            if a.GetAtomicNum() == 0:
                nb = a.GetNeighbors()[0]
                nb.SetAtomMapNum(a.GetAtomMapNum())
                a.SetAtomicNum(1)
                a.SetAtomMapNum(0)
                a.SetIsotope(0)
        pm = pm.GetMol()
        Chem.SanitizeMol(pm)
        pm = Chem.RemoveHs(pm)
        parent_mapped = Chem.MolToSmiles(pm)
        for a in pm.GetAtoms():
            a.SetAtomMapNum(0)
        parent = _canon(Chem.MolToSmiles(pm))
        if parent is None:
            return cls, None, 0
        if self._forward_env(parent_mapped) != env:
            self.stats["cycle_reverse_only"] += 1
            return cls, None, 0
        # multiplicity: 2 if the context has an automorphism swapping its attachment points and
        # both labellings of the core are DB rows (two forward edges rebuild this cut), else 1
        mult = 1
        ctxm = Chem.MolFromSmiles(Chem.MolToSmiles(ctx))
        for a in ctxm.GetAtoms():
            a.SetAtomMapNum(0)                   # the map numbers would break the symmetry
        ranks = list(Chem.CanonicalRankAtoms(ctxm, breakTies=False))
        aps = [a.GetIdx() for a in ctxm.GetAtoms() if a.GetAtomicNum() == 0]
        if len(aps) == 2 and ranks[aps[0]] == ranks[aps[1]] and spell != spell_sw:
            both = [s for s in (spell, spell_sw)
                    if (cid := self.ccore_id(s)) is not None and self.has_cpair(eid, cid)]
            if len(both) == 2:
                mult = 2
                self.stats["cycle_cut_mult2"] += 1
        return cls, parent, mult

    def cut_parents(self, smi):
        """parents over all DB-supported cut classes (grow and ring-arc), for reachability"""
        return super().cut_parents(smi) + [p for p, _ in self.cycle_cuts(smi)]

    def parents(self, smi):
        """parents in S, one entry per in-edge (grow cut classes, ring-arc cut classes with their
        multiplicity, ROOT for a seed)"""
        if smi == ROOT:
            return []
        ps = super().parents(smi)                    # grow in-edges (+ ROOT)
        for p, m in self.cycle_cuts(smi):
            if self.reachable(p):
                ps.extend([p] * m)
        return ps
