"""`cremspace exact`: exact enumeration of a space up to a (small) HAC cap -- N(h) for every heavy-atom count.

Ground truth for validating the sampler on small spaces (e.g. the tiny example, or a real slice with a low
--hac-max). Molecules are expanded in HAC order (every edge increases HAC), so each level is complete when it is
expanded. Cost grows with the size of the space; only practical up to about 10^6-10^7 molecules.
"""
import argparse
import json
import time
from collections import defaultdict
from multiprocessing import Pool

from cremspace.space import open_space

G = None


def _init(db, set_name, mode, max_atoms, hac_max):
    global G
    G = open_space(db, set_name, mode, max_atoms=max_atoms, hac_max=hac_max)


def _expand(smi):
    return set(G.children(smi))


def enumerate_space(slice_path, set_name, mode="grow", hac_max=12, max_atoms=None, ncpu=4, save_smiles=None):
    _init(slice_path, set_name, mode, max_atoms, hac_max)
    by_h = defaultdict(set)
    for smi, h in G.seeds():
        by_h[h].add(smi)
    n_seed = {h: len(v) for h, v in by_h.items()}
    t = time.time()
    counts = {}
    sf = open(save_smiles, "w") if save_smiles else None
    with Pool(ncpu, initializer=_init, initargs=(slice_path, set_name, mode, max_atoms, hac_max)) as pool:
        for h in range(1, hac_max + 1):
            level = sorted(by_h.pop(h, ()))
            counts[h] = len(level)
            if h < hac_max:
                for ch in pool.imap_unordered(_expand, level, chunksize=20):
                    for c in ch:
                        by_h[G.hac(c)].add(c)
            if sf:
                sf.writelines(f"{s}\t{h}\n" for s in level)
            print(f"h={h} N={counts[h]:,} (seeds {n_seed.get(h, 0):,}) t={time.time() - t:.0f}s", flush=True)
    if sf:
        sf.close()
    return dict(set=set_name, mode=mode, hac_max=hac_max, max_atoms=G.max_atoms, radius=G.radius,
                counts=counts, seeds=n_seed, slice=slice_path)


def add_parser(sub):
    p = sub.add_parser("exact", help="exact enumeration of a small space (validation)", description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("slice")
    p.add_argument("--set", required=True)
    p.add_argument("--mode", choices=["grow", "cycle"], default="grow")
    p.add_argument("--hac-max", type=int, required=True)
    p.add_argument("--max-atoms", type=int, help="default: the slice's")
    p.add_argument("-c", "--ncpu", type=int, default=4)
    p.add_argument("-o", "--output", required=True, help="JSON with N(h)")
    p.add_argument("--save-smiles", help="also write every molecule (SMILES<TAB>HAC)")

    def run(a):
        r = enumerate_space(a.slice, a.set, a.mode, a.hac_max, a.max_atoms, a.ncpu, a.save_smiles)
        with open(a.output, "w") as f:
            json.dump(r, f, indent=1)
        import math
        print(f"log10 N(1..{a.hac_max}) = {math.log10(sum(r['counts'].values())):.3f}")
        return 0
    p.set_defaults(func=run)
