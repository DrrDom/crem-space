"""Automatic refreeze, run by the sampler for each set: production with frozen weights -> tm estimate -> new
frozen weights.

A Wang-Landau freeze can leave weights that trap production (at radius 1 walkers spent most steps at HAC 1, or never
entered some bins). Wang-Landau re-adaptation does not fix bins it rarely enters; the tm estimate does, because it
only needs some samples there. Every few hours the sampler pools the production counts of the set's walkers (all
epochs), solves tm, fills bins without an estimate by linear interpolation of log10 N, and judges the current epoch's
pooled HAC occupancy (share of production steps per bin / flat share). If the epoch has enough steps and the
occupancy is outside [occ_min, occ_max], it writes a new tag to REFREEZE.json; walkers pick it up within about a
minute, archive their epoch and continue with the new weights.
"""
import math
import time

import numpy as np

from cremspace.tm import load_states, pooled_counts, tm_solve
from cremspace.util import dump_json, load_json


def fill(v):
    v = np.array(v, float)
    ok = np.where(np.isfinite(v))[0]
    bad = [h for h in range(1, len(v)) if not np.isfinite(v[h])]
    for h in bad:
        lo, hi = ok[ok < h], ok[ok > h]
        if len(lo) and len(hi):
            a, b = lo[-1], hi[0]
            v[h] = v[a] + (v[b] - v[a]) * (h - a) / (b - a)
        elif len(lo) >= 2:
            a, b = lo[-2], lo[-1]
            v[h] = v[b] + (v[b] - v[a]) * (h - b) / (b - a)
    return v, bad


def check(set_dir, out, min_steps=10_000_000, occ_min=0.3, occ_max=3.0, max_iter=6):
    """one refreeze decision for a set; returns (status, message), status in
    None (wait), 'flat', 'refreeze', 'limit'"""
    W = load_states(set_dir)
    prod = [s for s in W if s.get("phase") == "prod"]
    if not prod:
        return None, "no walker in production"
    old = load_json(out)
    cur = old["tag"] if old else None
    k = int(cur[2:]) if cur else 0
    epoch = [s for s in prod if s.get("refreeze_tag") == cur]
    H = np.sum([s["H_prod"] for s in epoch], 0)[1:] if epoch else np.zeros(1)
    steps = int(H.sum())
    occ = H / max(steps, 1) * len(H)
    info = dict(walkers=len(prod), on_current=len(epoch), epoch_steps=steps,
                occ_min=round(float(occ.min()), 3), occ_max=round(float(occ.max()), 3),
                occ_argmin=int(occ.argmin()) + 1, occ_argmax=int(occ.argmax()) + 1)
    if len(epoch) < len(prod) or steps < min_steps:
        return None, f"waiting: {info}"
    if occ_min <= occ.min() and occ.max() <= occ_max:
        return "flat", f"flat: {info}"
    if k >= max_iter:
        return "limit", f"refreeze limit {max_iter} reached: {info}"
    F = R = n = 0
    for s in prod:
        f_, r_, n_, _ = pooled_counts(s)
        F, R, n = F + f_, R + r_, n + n_
    tm, bad = fill(tm_solve(F, R, n))
    if not np.isfinite(tm[1:]).all():
        return None, f"tm estimate incomplete, waiting: {info}"
    tag = f"rf{k + 1}"
    dump_json(dict(tag=tag, tm=tm.tolist(), filled=bad, prev=info, time=time.strftime("%Y-%m-%d %H:%M:%S")), out)
    top = tm[-1] if math.isfinite(tm[-1]) else float("nan")
    return "refreeze", f"{tag}: {info}, filled bins {bad}, log10 N(top)={top:.3f}"
