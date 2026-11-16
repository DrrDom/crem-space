"""`cremspace status`: one line per set of a RUN directory while (or after) it is sampled: the shared Wang-Landau
state, walker phases, production length and round trips, crashed or silent walkers and refreezes."""
import argparse
import collections
import os
import statistics
import time

from cremspace.tm import load_states
from cremspace.util import load_json


def status(runs):
    now = time.time()
    for run_dir in runs:
        run = load_json(os.path.join(run_dir, "run.json"))
        if run is None:
            print(f"{run_dir}: no run.json")
            continue
        inv = run["invocations"][-1] if run.get("invocations") else {}
        print(f"== {run_dir}: mode {run['mode']}, radius {run['slice_meta']['radius']}, HAC <= {run['hac_max']}; "
              f"last started {inv.get('start')} on {inv.get('host')} for {inv.get('hours')} h")
        for s in run["sets"]:
            sd = os.path.join(run_dir, s)
            sh = load_json(os.path.join(sd, "shared.json"), default={})
            W = load_states(sd)
            ph = collections.Counter(w.get("phase", "adapt") for w in W)
            prod = [w["step"] - w["freeze_step"] for w in W if w.get("phase") == "prod"]
            rts = [w.get("n_roundtrips", {}).get("prod", 0) for w in W]
            tb = stale = 0
            for w in W:
                lf = os.path.join(w["_dir"], "walker.log")
                if os.path.exists(lf):
                    with open(lf) as fh:
                        tail = fh.read().split("resumed at step")[-1]
                    tb += "Traceback" in tail
                    stale += now - os.path.getmtime(lf) > 1800 and "stopped at step" not in tail
            rf = load_json(os.path.join(sd, "REFREEZE.json"), default={})
            lnf = sh.get("lnf", float("nan"))
            print(f"  {s:<28} lnf {lnf:.2e} frozen {str(sh.get('frozen', '-')):<5} | {dict(ph)} | "
                  f"prod steps median {statistics.median(prod) if prod else 0:>11,.0f} | "
                  f"round trips min {min(rts) if rts else 0} | crashed {tb} silent>30min {stale}"
                  + (f" | refreeze {rf.get('tag')}" if rf else ""))
    return 0


def add_parser(sub):
    p = sub.add_parser("status", help="progress of RUN directories", description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.set_defaults(func=lambda a: status(a.runs))
