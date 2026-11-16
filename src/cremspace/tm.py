"""Transition-matrix estimate of log10 N(h) from the production counts of a set's walkers.

The exact identity  N_i <F_ij>_i = N_j <R_ji>_j  (F: forward edges from bin i into bin j, R: in-edges of bin j from
bin i, both averaged over the states of a bin, which production samples uniformly) gives one equation per ordered
bin pair,  ln N_j - ln N_i = ln(F_ij / n_i) - ln(R_ji / n_j);  with ln N_0 = 0 (the root) the system is solved by
weighted least squares. The counts of all production epochs are pooled (the identity holds for any frozen weights).
The uncertainty is a bootstrap over walkers.
"""
import glob
import json
import math
import os

import numpy as np

LN10 = math.log(10)


def tm_solve(F, R, n, ref=0):
    """least squares for u = ln N with u[ref] = 0; returns log10 N (nan for bins not connected by any equation)"""
    nb = len(n)
    rows, rhs, w = [], [], []
    for i in range(nb):
        for j in range(nb):
            if i != j and n[i] > 0 and n[j] > 0 and F[i, j] > 0 and R[j, i] > 0:
                rows.append((i, j))
                rhs.append(math.log(F[i, j] / n[i]) - math.log(R[j, i] / n[j]))
                w.append(1.0 / (1.0 / F[i, j] + 1.0 / R[j, i]))
    if not rows:
        return np.full(nb, np.nan)
    col = {b: c for c, b in enumerate(b for b in range(nb) if b != ref)}
    A = np.zeros((len(rows), nb - 1))
    for k, (i, j) in enumerate(rows):
        if i != ref:
            A[k, col[i]] = -1
        if j != ref:
            A[k, col[j]] = 1
    sw = np.sqrt(np.array(w))
    u = np.linalg.lstsq(A * sw[:, None], np.array(rhs) * sw, rcond=None)[0]
    connected = set()
    for i, j in rows:
        connected |= {i, j}
    out = np.insert(u, ref, 0.0) / LN10
    out[[h for h in range(nb) if h not in connected]] = np.nan
    return out


def log10_total(v, h_min=1):
    """log10 of sum_h N(h) over h >= h_min (finite bins only)"""
    v = np.asarray(v, float)[h_min:]
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan")
    m = v.max()
    return float(m + math.log10((10 ** (v - m)).sum()))


def walker_dirs(set_dir):
    return sorted(d for d in glob.glob(os.path.join(set_dir, "w*")) if os.path.isdir(d))


def load_states(set_dir):
    out = []
    for d in walker_dirs(set_dir):
        sp = os.path.join(d, "state.json")
        if not os.path.exists(sp):
            continue
        for _ in range(5):
            try:
                with open(sp) as fh:
                    st = json.load(fh)
                break
            except ValueError:
                import time
                time.sleep(1)
        else:
            continue
        st["_name"] = os.path.basename(d)
        st["_dir"] = d
        out.append(st)
    return out


def pooled_counts(st):
    """(F, R, n, H) of a walker: current production epoch plus all archived epochs"""
    nb = len(st["lng"])
    F, R = np.zeros((nb, nb)), np.zeros((nb, nb))
    n, H = np.zeros(nb), np.zeros(nb)
    epochs = list(st.get("prod_epochs", []))
    if st.get("phase") == "prod":
        epochs.append(st)
    for e in epochs:
        if e.get("tm_F") is None:
            continue
        F += np.array(e["tm_F"], float)
        R += np.array(e["tm_R"], float)
        n += np.array(e["tm_n"], float)
        H += np.array(e["H_prod"], float)
    return F, R, n, H


def estimate(states, boot=500, seed=0):
    """tm estimate from walkers in production: dict with per-bin log10 N and bootstrap sd, total and its sd and
    95% interval, plus the frozen-WL and reweighting estimates of the current epoch as a cross-check"""
    prod = [s for s in states if s.get("phase") == "prod" or s.get("prod_epochs")]
    if not prod:
        return None
    nb = len(prod[0]["lng"])
    C = [pooled_counts(s) for s in prod]
    Fs = np.array([c[0] for c in C])
    Rs = np.array([c[1] for c in C])
    ns = np.array([c[2] for c in C])
    tm = tm_solve(Fs.sum(0), Rs.sum(0), ns.sum(0))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(boot):
        k = rng.integers(0, len(prod), len(prod))
        boots.append(tm_solve(Fs[k].sum(0), Rs[k].sum(0), ns[k].sum(0)))
    boots = np.array(boots)
    tot_boot = np.array([log10_total(b) for b in boots])
    tot_boot = tot_boot[np.isfinite(tot_boot)]
    with np.errstate(invalid="ignore"):
        sd_bin = np.nanstd(boots, 0)

    cur = [s for s in prod if s.get("phase") == "prod"]
    wl = [(np.array(s["lng_frozen"]) - s["lng_frozen"][0]) / LN10 for s in cur]
    rew = []
    for s in cur:
        g, Hp = np.array(s["lng_frozen"]), np.array(s["H_prod"], float)
        if Hp[0] > 0:
            with np.errstate(divide="ignore"):
                v = g + np.log(Hp)
            v[~np.isfinite(v)] = np.nan
            rew.append((v - v[0]) / LN10)
    total = log10_total(tm)
    return dict(
        h=list(range(nb)), log10N=tm.tolist(), log10N_sd=sd_bin.tolist(),
        total=total, total_sd=float(np.std(tot_boot)) if len(tot_boot) else float("nan"),
        total_ci95=[float(np.percentile(tot_boot, 2.5)), float(np.percentile(tot_boot, 97.5))]
        if len(tot_boot) else [float("nan")] * 2,
        n_tm=ns.sum(0).tolist(), walkers=len(prod),
        total_wl=float(np.mean([log10_total(v) for v in wl])) if wl else float("nan"),
        total_rew=float(np.mean([log10_total(v) for v in rew])) if rew else float("nan"),
        epochs=max(len(s.get("prod_epochs", [])) + (s.get("phase") == "prod") for s in prod),
    )
