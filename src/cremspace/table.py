"""`cremspace table`: one table of the final estimates from the estimate.json files that `cremspace diagnose`
wrote in one or more RUN directories (e.g. grow and grow + make_cycle runs, or several radii)."""
import argparse
import math
import os
import sys

from cremspace.util import load_json


def table(runs, per_hac=False, out=sys.stdout):
    rows, sources = [], set()
    hmax = 0
    for run_dir in runs:
        run = load_json(os.path.join(run_dir, "run.json"))
        if run is None:
            raise SystemExit(f"{run_dir}: no run.json")
        sm = run["slice_meta"]
        sources.add((sm.get("source"), sm.get("source_size"), sm.get("source_mtime")))
        for s in run["sets"]:
            res = load_json(os.path.join(run_dir, s, "estimate.json"))
            if res is None:
                print(f"warning: {run_dir}/{s}: no estimate.json (run `cremspace diagnose {run_dir}`)",
                      file=sys.stderr)
                continue
            est = res.get("estimate")
            if not est:
                print(f"warning: {run_dir}/{s}: no estimate yet ({res['verdict']})", file=sys.stderr)
                continue
            if res["verdict"] != "DONE":
                print(f"warning: {run_dir}/{s}: {res['verdict']}", file=sys.stderr)
            hmax = max(hmax, run["hac_max"])
            rows.append((run_dir, run, s, res, est))
    if len(sources) > 1:
        print("warning: the runs were sampled from slices of different source databases", file=sys.stderr)

    def f(v, d=3):
        return "" if v is None or not math.isfinite(v) else f"{v:.{d}f}"

    cols = ["run", "set", "mode", "radius", "min_freq", "max_atoms", "hac_max", "log10N", "sd", "ci95_low",
            "ci95_high", "verdict"]
    if per_hac:
        cols += [f"log10N_h{h}" for h in range(1, hmax + 1)]
    print("\t".join(cols), file=out)
    for run_dir, run, s, res, est in rows:
        sm = run["slice_meta"]
        line = [os.path.basename(os.path.normpath(run_dir)), s, run["mode"], str(sm.get("radius")),
                str(sm.get("min_freq")), str(run["max_atoms"]), str(run["hac_max"]), f(est["total"]),
                f(est["total_sd"]), f(est["total_ci95"][0]), f(est["total_ci95"][1]), res["verdict"]]
        if per_hac:
            v = est["log10N"]
            line += [f(v[h]) if h < len(v) else "" for h in range(1, hmax + 1)]
        print("\t".join(line), file=out)
    return 0


def add_parser(sub):
    p = sub.add_parser("table", help="table of the final estimates over RUN directories", description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="RUN directories (diagnosed)")
    p.add_argument("--per-hac", action="store_true", help="add log10 N for every heavy-atom count")
    p.add_argument("-o", "--output", help="TSV file (default: stdout)")

    def run(a):
        if a.output:
            with open(a.output, "w") as fh:
                return table(a.runs, a.per_hac, fh)
        return table(a.runs, a.per_hac)
    p.set_defaults(func=run)
