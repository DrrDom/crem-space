"""Mixing diagnostics on production samples: are within-bin samples consistent with one uniform sample?

With frozen weights pi(x) ~ 1/g(h(x)), samples at a given HAC are uniform over that bin. Poor mixing (walkers stuck
in chemical families) shows up as differences that a flat HAC histogram cannot reveal. Per HAC group and descriptor:
  walkers    Kruskal-Wallis across walkers on tour means, and the largest walker-mean shift (in sd of tour means)
  drift      KS test of the first vs second half of each walker's tours (pooled)
  tours      intraclass correlation of the descriptor by tour -> effective number of samples
  scaffolds  share of generic Murcko scaffolds seen by >= 2 walkers vs a label permutation null (much lower than
             the null: walkers explore different regions)
Per HAC bin (informative only): collisions -- identical molecules sampled by different walkers;
N(h) ~ sum_{a<b} n_a n_b / c when c >= 3.
Only the current production epoch is used (samples_prod.tsv).
"""
import math
import os
from collections import Counter, defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors as rd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

from cremspace.tm import walker_dirs

RDLogger.DisableLog("rdApp.*")
DESC = ["rings", "arom_rings", "ring_atoms_frac", "fsp3", "N", "O", "S", "halogen", "rotb"]


def describe(smi):
    m = Chem.MolFromSmiles(smi)
    ha = m.GetNumHeavyAtoms()
    cnt = Counter(a.GetSymbol() for a in m.GetAtoms())
    try:
        scaf = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(m)))
    except Exception:
        scaf = "?"
    ring_atoms = sum(1 for a in m.GetAtoms() if a.IsInRing())
    return dict(rings=rd.CalcNumRings(m), arom_rings=rd.CalcNumAromaticRings(m),
                ring_atoms_frac=ring_atoms / ha, fsp3=rd.CalcFractionCSP3(m),
                N=cnt["N"], O=cnt["O"], S=cnt["S"], halogen=cnt["F"] + cnt["Cl"] + cnt["Br"] + cnt["I"],
                rotb=rd.CalcNumRotatableBonds(m), scaffold=scaf)


def load_samples(set_dir):
    rows = []                                     # (walker, step, tour, h, smi)
    for d in walker_dirs(set_dir):
        w = os.path.basename(d)
        f = os.path.join(d, "samples_prod.tsv")
        if not os.path.exists(f):
            continue
        seen = {}
        with open(f) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4 or not line.endswith("\n"):    # truncated by a kill
                    continue
                s, t, h, smi = parts
                seen[int(s)] = (w, int(s), int(t), int(h), smi)
        rows.extend(seen[k] for k in sorted(seen))
    return rows


def icc(values, groups):
    """one-way ANOVA intraclass correlation and mean group size"""
    by = defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(v)
    by = [np.array(v) for v in by.values() if len(v) > 1]
    if len(by) < 3:
        return float("nan"), float("nan")
    k = len(by)
    n = sum(len(v) for v in by)
    m0 = (n - sum(len(v) ** 2 for v in by) / n) / (k - 1)
    grand = np.concatenate(by).mean()
    msb = sum(len(v) * (v.mean() - grand) ** 2 for v in by) / (k - 1)
    msw = sum(((v - v.mean()) ** 2).sum() for v in by) / (n - k)
    if msb + (m0 - 1) * msw == 0:
        return float("nan"), m0
    return (msb - msw) / (msb + (m0 - 1) * msw), m0


def shared_scaffold_share(scafs, walkers):
    by = defaultdict(set)
    for s, w in zip(scafs, walkers):
        by[s].add(w)
    return sum(1 for s in scafs if len(by[s]) >= 2) / len(scafs)


def default_groups(hac_max):
    """three HAC groups of (nearly) equal width, e.g. 1-12, 13-24, 25-36"""
    b = [round(hac_max * k / 3) for k in range(4)]
    return [(b[k] + 1, b[k + 1]) for k in range(3) if b[k + 1] >= b[k] + 1]


def diagnose_chemistry(set_dir, hac_max, groups=None, perm=200, min_samples=50):
    rows = load_samples(set_dir)
    out = dict(n=len(rows), groups={}, collisions={})
    if not rows:
        return out
    cache = {}
    D = [cache.setdefault(r[4], describe(r[4])) for r in rows]
    walkers = sorted({r[0] for r in rows})
    out.update(walkers=len(walkers), distinct=len(cache))
    rng = np.random.default_rng(0)
    for lo, hi in groups or default_groups(hac_max):
        idx = [i for i, r in enumerate(rows) if lo <= r[3] <= hi]
        if len(idx) < min_samples:
            continue
        wl = [rows[i][0] for i in idx]
        tours = [(rows[i][0], rows[i][2]) for i in idx]
        # tests on tour means: samples within a tour are strongly autocorrelated, while tours restart from a
        # uniformly drawn seed and are close to independent
        units = sorted(set(tours))
        pos = {u: k for k, u in enumerate(units)}
        u_walker = [u[0] for u in units]
        res = dict(samples=len(idx), tours=len(units))
        for dn in DESC:
            v = np.array([D[i][dn] for i in idx], dtype=float)
            um = np.zeros(len(units))
            uc = np.zeros(len(units))
            for val, u in zip(v, tours):
                um[pos[u]] += val
                uc[pos[u]] += 1
            um /= uc
            grp = [um[[k for k, w in enumerate(u_walker) if w == ww]] for ww in walkers]
            grp = [x for x in grp if len(x) >= 5]
            kw = stats.kruskal(*grp).pvalue if len(grp) >= 2 and np.ptp(um) > 0 else float("nan")
            shift = max(abs(x.mean() - um.mean()) for x in grp) / um.std() if um.std() > 0 and grp else 0.0
            first, second = [], []
            for ww in walkers:
                ks = [k for k, w in enumerate(u_walker) if w == ww]
                first += [um[k] for k in ks[:len(ks) // 2]]
                second += [um[k] for k in ks[len(ks) // 2:]]
            dr = stats.ks_2samp(first, second).pvalue if first and second and np.ptp(um) > 0 else float("nan")
            r_icc, m0 = icc(v, tours)
            neff = len(v) / (1 + (m0 - 1) * r_icc) if np.isfinite(r_icc) else float("nan")
            res[dn] = dict(mean=float(v.mean()), kw_p=float(kw), max_shift_sd=float(shift), drift_p=float(dr),
                           tour_icc=float(r_icc), n_eff=float(neff))
        sc = [D[i]["scaffold"] for i in idx]
        obs = shared_scaffold_share(sc, wl)
        null = []
        for _ in range(perm):                   # permute walker labels over whole tours
            pm = dict(zip(units, rng.permutation(u_walker)))
            null.append(shared_scaffold_share(sc, [pm[u] for u in tours]))
        pz = (np.sum(np.array(null) <= obs) + 1) / (len(null) + 1)
        res["scaffolds"] = dict(distinct=len(set(sc)), shared=obs, null_mean=float(np.mean(null)), p_low=float(pz))
        out["groups"][f"{lo}-{hi}"] = res

    by_h = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        by_h[r[3]][r[4]][r[0]] += 1
    for h in sorted(by_h):
        per_w = Counter()
        c = 0
        for smi, cw in by_h[h].items():
            ks = list(cw.values())
            c += (sum(ks) ** 2 - sum(k * k for k in ks)) // 2
            per_w.update(cw)
        nw = list(per_w.values())
        pairs = (sum(nw) ** 2 - sum(k * k for k in nw)) / 2
        out["collisions"][str(h)] = dict(n=sum(nw), c=c, log10N=math.log10(pairs / c) if c >= 3 else float("nan"))
    return out
