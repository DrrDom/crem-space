"""The CReM grow graph on a prepared slice.

The grow graph is treated as a multigraph whose edges are (parent P, H-site symmetry class s of P, core c):
P -> P+c@s. Every edge into a product M corresponds to exactly one DB-supported *cut class* of M (a symmetry class
of acyclic single bonds, with an orientation telling which side is the removed core), so

    D+(P) = sum over H-site classes s of #cores(env(P, s), 1..min(MAX_ATOMS, HAC_MAX - HAC(P)))
    D-(M) = #DB-supported cut classes of M whose parent is reachable  (+1 if M is a seed)

D+ is an index count -- no product is built. The Metropolis ratio of a grow/prune walk uses only D+ and D-, and
edge multiplicities cancel, so nothing needs product-level deduplication.

The space S = molecules reachable by grow from a seed (an H-capped core of the chosen set, see prepare.py) within
the HAC cap. A virtual root (HAC 0) has one edge to every seed.
"""

import random
from array import array

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA

from crem.mol_context import get_canon_context_core

from cremspace.util import LRU, connect_ro, qi

RDLogger.DisableLog("rdApp.*")

ROOT = ""                                   # the virtual root node
_H_CORE = None


def _h_core():
    global _H_CORE
    if _H_CORE is None:
        p = Chem.SmilesParserParams()
        p.removeHs = False
        _H_CORE = Chem.MolFromSmiles("[H][*:1]", p)
    return _H_CORE


class GrowSpace:
    def __init__(self, db, set_name, max_atoms=None, hac_max=None, reach_budget=3000,
                 cache_size=200_000, rng=None, reach_unknown=True):
        # one long-lived read-only connection per process; immutable: a slice is never written while it is read
        self.con = connect_ro(db)
        self.con.execute("PRAGMA query_only = ON")
        self.con.execute("PRAGMA mmap_size = 68719476736")
        self.con.execute("PRAGMA cache_size = -262144")
        names = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "slice_meta" not in names:
            raise ValueError(f"{db} is not a cremspace slice (run `cremspace prepare` first)")
        meta = dict(self.con.execute("SELECT key, value FROM slice_meta"))
        self.meta = meta
        self.radius = int(meta["radius"])
        sets = meta["sets"].split(",")
        if set_name not in sets:
            raise ValueError(f"set {set_name!r} not in the slice; its sets: {sets}")
        self.set_index = si = sets.index(set_name)
        self.set_name, self.bit = set_name, 1 << si
        self.thr = int(meta.get("min_freq", 1))
        self.max_atoms = int(meta["max_atoms"]) if max_atoms is None else max_atoms
        self.hac_max = int(meta["hac_max"]) if hac_max is None else hac_max
        if self.max_atoms > int(meta["max_atoms"]) or self.hac_max > int(meta["hac_max"]):
            raise ValueError(f"max_atoms {self.max_atoms} / hac_max {self.hac_max} exceed the slice's "
                             f"{meta['max_atoms']} / {meta['hac_max']}")
        self.K = int(meta["max_atoms"])                     # columns of the packed tables
        self.pq, self.pe = f"pq_{si}", f"pe_{si}"
        self.rt = f"radius{self.radius}"
        self.qs = qi(set_name)
        self.rng = rng or random.Random()
        self.reach_budget = reach_budget
        self.reach_unknown = reach_unknown      # how an exhausted reachability search resolves
        self.seed_ids = array("q", (r[0] for r in self.con.execute(
            "SELECT seed_id FROM seeds WHERE mask & ? AND hac <= ?", (self.bit, self.hac_max))))
        if not self.seed_ids:
            raise ValueError(f"set {set_name!r} has no seeds with HAC <= {self.hac_max}")
        self._env_id = LRU(cache_size)
        self._env_cores = LRU(cache_size)
        self._core_smi = LRU(cache_size)
        self._core_id = LRU(cache_size)
        self._sites = LRU(cache_size)
        self._cuts = LRU(cache_size)
        self._reach = LRU(4 * cache_size)
        # states a walker has visited are reachable by construction (it got there from the root). Kept outside
        # the LRU: re-deriving them after eviction could exhaust the search budget and, with
        # reach_unknown=False, turn a visited state's only parent "unreachable" -- an absorbing state.
        self.proven = set()
        self.stats = dict(reach_calls=0, reach_budget_hit=0, reach_false=0, unstable_smiles=0)

    # ------------------------------------------------------------------ DB lookups
    def env_id(self, env):
        v = self._env_id.get(env, -1)
        if v == -1:
            r = self.con.execute("SELECT env_id FROM envs WHERE env = ?", (env,)).fetchone()
            v = r[0] if r else None
            self._env_id.put(env, v)
        return v

    def _packed(self, cache, table, eid):
        v = cache.get(eid)
        if v is None:
            r = self.con.execute(f"SELECT * FROM {table} WHERE env_id = ?", (eid,)).fetchone()
            v = (r[1], [0] + list(r[2:2 + self.max_atoms])) if r else (0, [0] * (self.max_atoms + 1))
            cache.put(eid, v)
        return v

    def env_cores(self, eid):
        """(start row in the packed table, cum) for the grow cores of this set at env eid;
        cum[k] = #cores with <= k atoms"""
        return self._packed(self._env_cores, self.pe, eid)

    def core_at(self, eid, r):
        """id of the r-th core (size-ordered) usable at env eid"""
        h, _ = self.env_cores(eid)
        return self.con.execute(f"SELECT core_smi_id FROM {self.pq} WHERE rid = ?", (h + r,)).fetchone()[0]

    def core_range(self, eid, c):
        """ids of the first c cores (size-ordered) usable at env eid"""
        h, _ = self.env_cores(eid)
        return [r[0] for r in self.con.execute(
            f"SELECT core_smi_id FROM {self.pq} WHERE rid >= ? AND rid < ? ORDER BY rid", (h, h + c))]

    def core_smi(self, cid):
        v = self._core_smi.get(cid)
        if v is None:
            v = self.con.execute("SELECT core_smi FROM frags WHERE core_smi_id = ?", (cid,)).fetchone()[0]
            self._core_smi.put(cid, v)
        return v

    def core_id(self, smi):
        v = self._core_id.get(smi, -1)
        if v == -1:
            r = self.con.execute("SELECT core_smi_id FROM frags WHERE core_smi = ?", (smi,)).fetchone()
            v = r[0] if r else None
            self._core_id.put(smi, v)
        return v

    def has_pair(self, eid, cid):
        return self.con.execute(
            f"SELECT 1 FROM {self.rt} WHERE env_id = ? AND core_smi_id = ? AND {self.qs} >= ?",
            (eid, cid, self.thr)).fetchone() is not None

    def is_seed(self, smi):
        return self.con.execute("SELECT 1 FROM seeds WHERE smi = ? AND mask & ? AND hac <= ?",
                                (smi, self.bit, self.hac_max)).fetchone() is not None

    def random_seed(self):
        sid = self.rng.choice(self.seed_ids)
        return self.con.execute("SELECT smi FROM seeds WHERE seed_id = ?", (sid,)).fetchone()[0]

    def seeds(self):
        """[(smi, hac)] of all seeds of the set"""
        return self.con.execute("SELECT smi, hac FROM seeds WHERE mask & ? AND hac <= ?",
                                (self.bit, self.hac_max)).fetchall()

    # ------------------------------------------------------------------ forward (grow)
    def sites(self, smi):
        """[(atom_idx, env_id or None)] one per symmetry class of H-bearing heavy atoms, and HAC"""
        v = self._sites.get(smi)
        if v is None:
            mol = Chem.MolFromSmiles(smi)
            hac = mol.GetNumHeavyAtoms()
            ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
            molh = Chem.AddHs(mol)
            seen, out = set(), []
            for a in mol.GetAtoms():
                if a.GetTotalNumHs() == 0 or ranks[a.GetIdx()] in seen:
                    continue
                seen.add(ranks[a.GetIdx()])
                h = next(n for n in molh.GetAtomWithIdx(a.GetIdx()).GetNeighbors() if n.GetAtomicNum() == 1)
                ctx = Chem.RWMol(molh)
                d = ctx.GetAtomWithIdx(h.GetIdx())
                d.SetAtomicNum(0)
                d.SetAtomMapNum(1)
                env, _ = get_canon_context_core(ctx, _h_core(), self.radius)
                out.append((a.GetIdx(), self.env_id(env) if env else None))
            v = (out, hac)
            self._sites.put(smi, v)
        return v

    def site_counts(self, smi):
        out, hac = self.sites(smi)
        k = min(self.max_atoms, self.hac_max - hac)
        if k < 1:
            return [], hac, k
        return [(ai, eid, self.env_cores(eid)[1][k] if eid is not None else 0) for ai, eid in out], hac, k

    def d_plus(self, smi):
        if smi == ROOT:
            return len(self.seed_ids)
        sc, _, _ = self.site_counts(smi)
        return sum(c for _, _, c in sc)

    def attach(self, smi, atom_idx, core_smi):
        """product of replacing one H on atom_idx of smi by core_smi ([*:1] attachment).

        Returned as the SMILES that survives a re-parse: for a few absurd ring systems in a DB (e.g. a benzene
        bridged meta by one sp3 carbon) RDKit writes an aromatic SMILES that it re-reads as Kekule, so the first
        string is not a stable identity. Such products are counted in stats['unstable_smiles']; their reverse cut
        usually finds no DB pair, and the walker then rejects the move. None if the product does not re-parse.
        """
        mol = Chem.MolFromSmiles(smi)
        core = Chem.MolFromSmiles(core_smi)
        n = mol.GetNumAtoms()
        rw = Chem.RWMol(Chem.CombineMols(mol, core))
        dummy = next(a for a in rw.GetAtoms() if a.GetIdx() >= n and a.GetAtomicNum() == 0)
        nb = dummy.GetNeighbors()[0]
        btype = rw.GetBondBetweenAtoms(dummy.GetIdx(), nb.GetIdx()).GetBondType()
        a = rw.GetAtomWithIdx(atom_idx)
        if a.GetNumExplicitHs() > 0:
            a.SetNumExplicitHs(a.GetNumExplicitHs() - 1)
        rw.AddBond(atom_idx, nb.GetIdx(), btype)
        rw.RemoveAtom(dummy.GetIdx())
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        s1 = Chem.MolToSmiles(m)
        m2 = Chem.MolFromSmiles(s1)
        if m2 is None:
            self.stats["unstable_smiles"] += 1
            return None
        s2 = Chem.MolToSmiles(m2)
        if s2 != s1:
            self.stats["unstable_smiles"] += 1
        return s2

    def sample_child(self, smi):
        """uniform over grow edges of smi; returns product SMILES or None if D+ = 0"""
        if smi == ROOT:
            return self.random_seed()
        sc, hac, k = self.site_counts(smi)
        tot = sum(c for _, _, c in sc)
        if tot == 0:
            return None
        r = self.rng.randrange(tot)
        for ai, eid, c in sc:
            if r < c:
                cid = self.core_at(eid, r)          # cores sorted by size: the first c have <= k atoms
                return self.attach(smi, ai, self.core_smi(cid))
            r -= c

    def children(self, smi):
        """all grow edges as a list of product SMILES (with multiplicity) -- exact enumeration and tests"""
        if smi == ROOT:
            return [s for s, _ in self.seeds()]
        sc, hac, k = self.site_counts(smi)
        out = [self.attach(smi, ai, self.core_smi(cid))
               for ai, eid, c in sc if c for cid in self.core_range(eid, c)]
        return [p for p in out if p is not None]

    # ------------------------------------------------------------------ backward (prune)
    def cut_edges(self, smi):
        """[(parent, env_id of the cut)] over DB-supported cut classes of smi (one entry per cut class), not yet
        filtered by reachability or forward visibility"""
        v = self._cuts.get(smi)
        if v is None:
            mol = Chem.MolFromSmiles(smi)
            ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
            for a in mol.GetAtoms():
                a.SetIntProp("orig_idx", a.GetIdx())
            seen, v = set(), []
            for _, ctx in rdMMPA.FragmentMol(mol, maxCuts=1, pattern="[!#1]!@!=!#[!#1]",
                                             resultsAsMols=True, maxCutBonds=500):
                comps = Chem.GetMolFrags(ctx, asMols=True)
                for c, x in ((comps[0], comps[1]), (comps[1], comps[0])):   # c = removed core
                    if not (1 <= c.GetNumHeavyAtoms() <= self.max_atoms):
                        continue
                    key = (ranks[_cut_atom(c)], ranks[_cut_atom(x)])        # oriented bond class
                    if key in seen:
                        continue
                    seen.add(key)
                    # cheap pre-filter: for a single-attachment piece the plain canonical SMILES equals crem's
                    # canonical core string, so a piece unknown to frags cannot be DB-supported
                    if self.core_id(Chem.MolToSmiles(c)) is None:
                        continue
                    env, core = get_canon_context_core(x, c, self.radius)
                    if env is None:
                        continue
                    eid, cid = self.env_id(env), self.core_id(core)
                    if eid is None or cid is None or not self.has_pair(eid, cid):
                        continue
                    p = Chem.RWMol(x)
                    for at in p.GetAtoms():
                        if at.GetAtomicNum() == 0:
                            at.SetAtomicNum(1)
                            at.SetAtomMapNum(0)
                            at.SetIsotope(0)
                    v.append((Chem.MolToSmiles(Chem.RemoveHs(p)), eid))
            self._cuts.put(smi, v)
        return v

    def cut_parents(self, smi):
        """parents over DB-supported cut classes of smi (one list entry per cut class)"""
        return [p for p, _ in self.cut_edges(smi)]

    def site_envs(self, smi):
        """set of env ids of the H-site classes of smi"""
        return {e for _, e in self.sites(smi)[0] if e is not None}

    def reachable(self, smi, _budget=None):
        """is smi in S (grow-reachable from a seed)? DFS with memo; an exhausted budget resolves to
        self.reach_unknown (counted in stats['reach_budget_hit'])"""
        if smi == ROOT or smi in self.proven:
            return True
        v = self._reach.get(smi)
        if v is not None:
            return v
        self.stats["reach_calls"] += 1
        budget = [self.reach_budget] if _budget is None else _budget
        res = self._reach_dfs(smi, budget, set())
        if res is None:
            self.stats["reach_budget_hit"] += 1
            res = self.reach_unknown
        elif not res:
            self.stats["reach_false"] += 1
        self._reach.put(smi, res)
        return res

    def _reach_dfs(self, smi, budget, stack):
        if smi in self.proven:
            return True
        v = self._reach.get(smi)
        if v is not None:
            return v
        if self.is_seed(smi):
            self._reach.put(smi, True)
            return True
        budget[0] -= 1
        if budget[0] < 0:
            return None
        stack.add(smi)
        unknown = False
        # big steps down first: smaller parents are more likely to be seeds
        ps = sorted(set(self.cut_parents(smi)), key=lambda s: len(s))
        # peek before recursing: one cached-reachable or seed parent settles this node; without it the DFS dives
        # into the first parent's whole prune tree before looking at the second one
        for p in ps:
            v = self._reach.get(p)
            if v or (v is None and p not in stack and self.is_seed(p)):
                self._reach.put(p, True)
                stack.discard(smi)
                self._reach.put(smi, True)
                return True
        for p in ps:
            if p in stack or self._reach.get(p) is False:
                continue
            r = self._reach_dfs(p, budget, stack)
            if r:
                stack.discard(smi)
                self._reach.put(smi, True)
                return True
            if r is None:
                unknown = True
                if budget[0] < 0:
                    break
        stack.discard(smi)
        if unknown:
            return None
        self._reach.put(smi, False)
        return False

    def parents(self, smi):
        """parents in S, one entry per cut class (with multiplicity); ROOT if smi is a seed"""
        if smi == ROOT:
            return []
        # keep an in-edge only if its parent is in S and the parent's own H-sites include the cut's env, i.e. the
        # grow edge p -> smi also exists forward. A few absurd DB ring systems break the SMILES round trip and
        # leave reverse-only edges, which otherwise give D+(p) = 0 on a prune move.
        ps = []
        for p, eid in self.cut_edges(smi):
            if eid in self.site_envs(p):
                if self.reachable(p):
                    ps.append(p)
            else:
                self.stats["reverse_only"] = self.stats.get("reverse_only", 0) + 1
        if self.is_seed(smi):
            ps.append(ROOT)
        return ps

    def d_minus(self, smi):
        return len(self.parents(smi))

    # ------------------------------------------------------------------ edges by destination bin
    def seed_hac_counts(self):
        """{hac: #seeds}: the root's out-edges by destination bin"""
        if not hasattr(self, "_seed_hac"):
            self._seed_hac = dict(self.con.execute(
                "SELECT hac, count(*) FROM seeds WHERE mask & ? AND hac <= ? GROUP BY hac",
                (self.bit, self.hac_max)))
        return self._seed_hac

    def d_plus_by_bin(self, smi):
        """{destination HAC: #grow edges}; sums to d_plus(smi)"""
        if smi == ROOT:
            return dict(self.seed_hac_counts())
        sc, hac, k = self.site_counts(smi)
        out = {}
        for _, eid, c in sc:
            if not c:
                continue
            cum = self.env_cores(eid)[1]
            for n in range(1, k + 1):
                m = cum[n] - cum[n - 1]
                if m:
                    out[hac + n] = out.get(hac + n, 0) + m
        return out

    def d_minus_by_bin(self, smi):
        """{source HAC: #in-edges from that bin} (root = bin 0); sums to d_minus(smi)"""
        out = {}
        for p in self.parents(smi):
            h = self.hac(p)
            out[h] = out.get(h, 0) + 1
        return out

    @staticmethod
    def hac(smi):
        return 0 if smi == ROOT else Chem.MolFromSmiles(smi).GetNumHeavyAtoms()


def _cut_atom(frag):
    """original-molecule index of the heavy atom bonded to the dummy of a single-cut fragment"""
    for a in frag.GetAtoms():
        if a.GetAtomicNum() == 0:
            return a.GetNeighbors()[0].GetIntProp("orig_idx")
    raise ValueError("no dummy")
