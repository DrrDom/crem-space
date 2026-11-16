"""`cremspace diagnose`: estimate, convergence checks and verdict for every set of one or more RUN directories.

For each set (RUN/<set>/) it writes estimate.json and report.txt, and for each RUN a summary.tsv. A set is DONE when
all checks pass:
  production  every walker is in production
  precision   bootstrap sd (over walkers) of log10 N_total < --sd                                   (0.1)
  stability   the estimate from the first 50% of every walker's production differs from the full one by < --drift,
              and the last three points of the trace (80, 90, 100%) span < --drift                 (0.1)
  coverage    every walker has >= --min-rt production round trips (root -> HAC max -> root)         (20)
  mixing      descriptor tests across walkers on tour means: every Kruskal-Wallis and scaffold-sharing p-value
              >= --alpha / number of tests (Bonferroni) and every walker-mean shift <= --shift-max sd (0.05, 0.5)
Otherwise the verdict is CHECK with the failed checks and a suggested extension of production (rerun
`cremspace sample` with the same RUN directory).
"""
import argparse
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from cremspace.tm import estimate, load_states, log10_total, tm_solve
from cremspace.util import dump_json, load_json


# ---------------------------------------------------------------------------------------------- convergence trace
def _counts(d):
    return (np.array(d["F"] if "F" in d else d["tm_F"], float), np.array(d["R"] if "R" in d else d["tm_R"], float),
            np.array(d["n"] if "n" in d else d["tm_n"], float))


def _walker_trace(st, fractions):
    """cumulative (F, R, n) of one walker after the first f of its production (all epochs), for each f"""
    epochs = [(e, os.path.join(st["_dir"], f"tm_hist.e{k}.jsonl"), e["end_step"] - e["freeze_step"])
              for k, e in enumerate(st.get("prod_epochs", []), 1) if e.get("tm_F") is not None]
    if st.get("phase") == "prod":
        epochs.append((st, os.path.join(st["_dir"], "tm_hist.jsonl"), st["step"] - st["freeze_step"]))
    total = sum(n for _, _, n in epochs)
    if total <= 0:
        return None, 0
    targets = [f * total for f in fractions]
    best = [None] * len(fractions)        # (steps, counts)
    nb = len(st["lng"])
    zero = (np.zeros((nb, nb)), np.zeros((nb, nb)), np.zeros(nb))
    off_steps, off = 0, zero

    def offer(steps, cnt):
        for i, t in enumerate(targets):
            if steps <= t + 1e-9 and (best[i] is None or steps >= best[i][0]):
                best[i] = (steps, cnt)

    offer(0, zero)
    for e, path, n in epochs:
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    try:
                        s = json.loads(line)
                    except ValueError:                     # truncated by a kill
                        continue
                    if s["prod_steps"] > n:                # written after the last checkpoint, then re-run
                        continue
                    c = _counts(s)
                    offer(off_steps + s["prod_steps"], tuple(a + b for a, b in zip(off, c)))
        final = tuple(a + b for a, b in zip(off, _counts(e)))
        offer(off_steps + n, final)
        off_steps, off = off_steps + n, final
    return best, total


def convergence(states, fractions=tuple(0.1 * k for k in range(1, 11))):
    prod = [s for s in states if s.get("phase") == "prod" or s.get("prod_epochs")]
    tr = [_walker_trace(s, fractions) for s in prod]
    tr = [t for t, n in tr if t]
    if not tr:
        return None
    trace = []
    for i in range(len(fractions)):
        F = sum(t[i][1][0] for t in tr)
        R = sum(t[i][1][1] for t in tr)
        n = sum(t[i][1][2] for t in tr)
        trace.append(log10_total(tm_solve(F, R, n)))
    return dict(fractions=[round(f, 2) for f in fractions], trace=trace)


# ---------------------------------------------------------------------------------------------- one set
def diagnose_set(set_dir, sd_max=0.1, drift_max=0.1, min_rt=20, alpha=0.05, shift_max=0.5, boot=500,
                 exact=None, run_info=None):
    name = os.path.basename(set_dir.rstrip("/"))
    run_info = run_info or {}
    hac_max = run_info.get("hac_max")
    states = load_states(set_dir)
    res = dict(set=name, set_dir=set_dir, mode=run_info.get("mode"), radius=run_info.get("radius"),
               hac_max=hac_max, walkers=len(states))
    phases = [s.get("phase", "adapt") for s in states]
    res["walkers_in_production"] = sum(p == "prod" for p in phases)
    shared = load_json(os.path.join(set_dir, "shared.json"), default={})
    res["shared_lnf"] = shared.get("lnf")
    if not states:
        res.update(verdict="NO WALKERS", checks={})
        return res
    if hac_max is None:
        hac_max = len(states[0]["lng"]) - 1
        res["hac_max"] = hac_max
    est = estimate(states, boot=boot)
    if est is None:
        lnf = f"{shared['lnf']:.2g}" if shared.get("lnf") else "?"
        res.update(verdict=f"ADAPTING (shared ln f {lnf}, freezes at the run's --freeze-lnf)", checks={})
        return res
    res["estimate"] = est
    conv = convergence(states)
    res["convergence"] = conv
    rts = [s.get("n_roundtrips", {}).get("prod", 0) for s in states]
    prod_steps, prod_hours = [], []
    for s in states:
        n = sum(e["end_step"] - e["freeze_step"] for e in s.get("prod_epochs", []))
        if s.get("phase") == "prod":
            n += s["step"] - s["freeze_step"]
        prod_steps.append(n)
        prod_hours.append(s.get("elapsed", 0) / 3600 * n / max(s["step"], 1))
    res.update(round_trips=rts, prod_steps=prod_steps, prod_hours_median=float(np.median(prod_hours)))
    from cremspace.chemistry import diagnose_chemistry
    chem = diagnose_chemistry(set_dir, hac_max)
    res["chemistry"] = chem

    tests = [t for g in chem["groups"].values() for k, t in g.items() if isinstance(t, dict) and "kw_p" in t]
    scaf = [g["scaffolds"]["p_low"] for g in chem["groups"].values() if "scaffolds" in g]
    p_min = alpha / max(len(tests) + len(scaf), 1)
    kw = min((t["kw_p"] for t in tests if math.isfinite(t["kw_p"])), default=float("nan"))
    shift = max((t["max_shift_sd"] for t in tests), default=float("nan"))
    sc = min(scaf, default=float("nan"))
    drift_half = abs(conv["trace"][4] - conv["trace"][-1]) if conv else float("nan")
    span3 = (max(conv["trace"][-3:]) - min(conv["trace"][-3:])) if conv else float("nan")
    checks = dict(
        production=res["walkers_in_production"] == len(states),
        precision=bool(est["total_sd"] < sd_max),
        stability=bool(drift_half < drift_max and span3 < drift_max),
        coverage=bool(min(rts) >= min_rt),
        mixing=bool(tests) and not (kw < p_min or sc < p_min or shift > shift_max),
    )
    res.update(checks=checks, min_round_trips=min(rts), drift_half=drift_half, span_last3=span3,
               min_kw_p=kw, max_shift_sd=shift, min_scaffold_p=sc, p_threshold=p_min,
               thresholds=dict(sd=sd_max, drift=drift_max, min_rt=min_rt, alpha=alpha, shift_max=shift_max))
    failed = [k for k, v in checks.items() if not v]
    if not failed:
        res["verdict"] = "DONE"
    else:
        factor = 1.0
        if not checks["coverage"]:
            factor = max(factor, min_rt / max(min(rts), 1))
        if not checks["precision"]:
            factor = max(factor, (est["total_sd"] / sd_max) ** 2)
        if not (checks["stability"] and checks["mixing"]):
            factor = max(factor, 2.0)
        if not checks["production"]:
            res["verdict"] = f"CHECK: {', '.join(failed)}; continue sampling (some walkers are still adapting)"
        else:
            extra = max(1.0, (factor - 1) * res["prod_hours_median"])
            res["suggested_extra_hours"] = round(extra, 1)
            res["verdict"] = f"CHECK: {', '.join(failed)}; extend production by ~{extra:.0f} h"
    if exact:
        counts = {int(k): v for k, v in exact["counts"].items()}
        ex = [math.log10(counts[h]) if counts.get(h) else float("nan") for h in range(hac_max + 1)]
        tot = math.log10(sum(v for h, v in counts.items() if 1 <= h <= hac_max))
        res["exact"] = dict(log10N=ex, total=tot, z=(est["total"] - tot) / est["total_sd"]
                            if est["total_sd"] > 0 else float("nan"))
    return res


def report(res):
    L = [f"set {res['set']}  mode {res.get('mode')}  radius {res.get('radius')}  HAC 1..{res.get('hac_max')}",
         f"walkers {res['walkers']}, in production {res['walkers_in_production']}"]
    est = res.get("estimate")
    if est:
        L.append(f"production epochs {est['epochs']}; production steps per walker min {min(res['prod_steps']):,} "
                 f"median {int(np.median(res['prod_steps'])):,}; round trips min {min(res['round_trips'])} "
                 f"median {int(np.median(res['round_trips']))}")
        L.append("")
        ex = res.get("exact")
        L.append(f"{'HAC':>3} {'log10 N':>8} {'sd':>6}" + (f" {'exact':>8}" if ex else "") + f" {'tm samples':>11}")
        for h in range(1, len(est["log10N"])):
            line = f"{h:>3} {est['log10N'][h]:8.3f} {est['log10N_sd'][h]:6.3f}"
            if ex:
                line += f" {ex['log10N'][h]:8.3f}"
            L.append(line + f" {int(est['n_tm'][h]):>11,}")
        L.append("")
        L.append(f"log10 N(1..{res['hac_max']}) = {est['total']:.3f} +- {est['total_sd']:.3f}  "
                 f"(95% CI {est['total_ci95'][0]:.3f}..{est['total_ci95'][1]:.3f}); cross-checks of the current "
                 f"epoch: frozen WL {est['total_wl']:.3f}, reweighted {est['total_rew']:.3f}")
        if ex:
            L.append(f"exact log10 N = {ex['total']:.3f}; (estimate - exact) / sd = {ex['z']:+.2f}")
        conv = res.get("convergence")
        if conv:
            L.append("trace (first f of every walker's production): " +
                     "  ".join(f"{int(100 * f)}%:{t:.3f}" for f, t in zip(conv["fractions"], conv["trace"])))
        th = res["thresholds"]
        ck = res["checks"]
        mark = lambda b: "pass" if b else "FAIL"                # noqa: E731
        L.append("")
        L.append(f"production  all walkers in production: {mark(ck['production'])}")
        L.append(f"precision   sd {est['total_sd']:.3f} < {th['sd']}: {mark(ck['precision'])}")
        L.append(f"stability   |50%-100%| {res['drift_half']:.3f}, last-3 span {res['span_last3']:.3f} < "
                 f"{th['drift']}: {mark(ck['stability'])}")
        L.append(f"coverage    min round trips {res['min_round_trips']} >= {th['min_rt']}: {mark(ck['coverage'])}")
        L.append(f"mixing      min KW p {res['min_kw_p']:.3g}, min scaffold p {res['min_scaffold_p']:.3g} "
                 f">= {res['p_threshold']:.2g}; max walker shift {res['max_shift_sd']:.2f} <= {th['shift_max']} sd: "
                 f"{mark(ck['mixing'])}")
        chem = res.get("chemistry", {})
        for g, r in chem.get("groups", {}).items():
            L.append(f"  HAC {g}: {r['samples']:,} samples in {r['tours']:,} tours; scaffolds shared "
                     f"{r['scaffolds']['shared']:.3f} vs null {r['scaffolds']['null_mean']:.3f}")
            for dn, t in r.items():
                if isinstance(t, dict) and "kw_p" in t:
                    L.append(f"    {dn:<16} mean {t['mean']:7.3f}  KW p {t['kw_p']:8.2e}  shift {t['max_shift_sd']:5.2f}"
                             f"  drift p {t['drift_p']:8.2e}  tour ICC {t['tour_icc']:5.2f}  n_eff {t['n_eff']:8.0f}")
    L.append("")
    L.append(f"verdict: {res['verdict']}")
    return "\n".join(L) + "\n"


def _job(args):
    set_dir, kw = args
    res = diagnose_set(set_dir, **kw)
    if res.get("estimate") or res["walkers"]:
        dump_json(res, os.path.join(set_dir, "estimate.json"), indent=1, default=float)
        with open(os.path.join(set_dir, "report.txt"), "w") as f:
            f.write(report(res))
    return res


SUMMARY_COLS = ["set", "mode", "radius", "hac_max", "log10N", "sd", "ci95_low", "ci95_high", "min_round_trips",
                "median_prod_steps", "min_KW_p", "max_shift_sd", "min_scaffold_p", "verdict"]


def summary_row(res):
    est = res.get("estimate") or {}
    f = lambda v, d=3: "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"  # noqa
    g = lambda v: "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.3g}"         # noqa
    ci = est.get("total_ci95", [None, None])
    return [res["set"], str(res.get("mode")), str(res.get("radius")), str(res.get("hac_max")), f(est.get("total")),
            f(est.get("total_sd")), f(ci[0]), f(ci[1]), str(res.get("min_round_trips", "")),
            str(int(np.median(res["prod_steps"]))) if res.get("prod_steps") else "", g(res.get("min_kw_p")),
            f(res.get("max_shift_sd"), 2), g(res.get("min_scaffold_p")), res["verdict"]]


def diagnose(runs, ncpu=4, exact=None, **kw):
    # an enumeration with a higher cap also gives the counts at a lower one (edges only add atoms)
    exact_by = {}
    for f in exact or []:
        e = load_json(f)
        exact_by.setdefault((e["set"], e["mode"]), []).append(e)

    def find_exact(s, mode, hac_max):
        ok = [e for e in exact_by.get((s, mode), []) if int(e["hac_max"]) >= hac_max]
        return min(ok, key=lambda e: int(e["hac_max"])) if ok else None
    rc = 0
    for run_dir in runs:
        run = load_json(os.path.join(run_dir, "run.json"))
        if run is None:
            raise SystemExit(f"{run_dir}: no run.json (not a `cremspace sample` output directory)")
        info = dict(mode=run["mode"], hac_max=run["hac_max"], radius=int(run["slice_meta"]["radius"]))
        jobs = []
        for s in run["sets"]:
            sd = os.path.join(run_dir, s)
            if os.path.isdir(sd):
                jobs.append((sd, dict(kw, run_info=info, exact=find_exact(s, run["mode"], run["hac_max"]))))
        with Pool(max(1, min(ncpu, len(jobs)))) as pool:
            results = pool.map(_job, jobs)
        lines = ["\t".join(SUMMARY_COLS)] + ["\t".join(summary_row(r)) for r in results]
        with open(os.path.join(run_dir, "summary.tsv"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"== {run_dir}  (mode {info['mode']}, radius {info['radius']}, HAC <= {info['hac_max']})")
        for r in results:
            est = r.get("estimate")
            val = f"log10 N = {est['total']:.3f} +- {est['total_sd']:.3f}" if est else "no estimate yet"
            print(f"  {r['set']:<30} {val:<30} {r['verdict']}")
            if r["verdict"] != "DONE":
                rc = 1
        print(f"  details: {run_dir}/<set>/report.txt; table: {run_dir}/summary.tsv")
    return rc


def add_parser(sub):
    p = sub.add_parser("diagnose", help="estimate, convergence checks and verdict per set", description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="RUN directories of `cremspace sample`")
    p.add_argument("-c", "--ncpu", type=int, default=4, help="sets diagnosed in parallel")
    p.add_argument("--sd", type=float, default=0.1, help="precision threshold (log10 units)")
    p.add_argument("--drift", type=float, default=0.1, help="stability threshold (log10 units)")
    p.add_argument("--min-rt", type=int, default=20, help="production round trips required per walker")
    p.add_argument("--alpha", type=float, default=0.05, help="family-wise level of the mixing tests")
    p.add_argument("--shift-max", type=float, default=0.5, help="largest walker-mean shift (sd of tour means)")
    p.add_argument("--boot", type=int, default=500, help="bootstrap replicates")
    p.add_argument("--exact", nargs="*", help="output(s) of `cremspace exact` to compare with")
    p.set_defaults(func=lambda a: diagnose(a.runs, ncpu=a.ncpu, exact=a.exact, sd_max=a.sd, drift_max=a.drift,
                                           min_rt=a.min_rt, alpha=a.alpha, shift_max=a.shift_max, boot=a.boot))

