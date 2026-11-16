"""One Wang-Landau / production walker on a slice (started by `cremspace sample`; not a user command).

N(h) = number of molecules with h heavy atoms in the space S (grow, or grow + make_cycle).
Walk on S + {root} with two moves, each proposed with probability 1/2:
  grow : x -> y, uniform over the D+(x) forward edges of x
  prune: x -> y, uniform over the D-(x) in-edges of x (parents in S, the root for seeds)
Metropolis-Hastings acceptance for the target pi(x) ~ 1/g(h(x)):
  grow : min(1, g(h_x)/g(h_y) * D+(x)/D-(y))
  prune: min(1, g(h_x)/g(h_y) * D-(x)/D+(y))
Edge multiplicities cancel, so this is exact on the multigraph (see space/grow.py).

Phase "adapt": Wang-Landau with the 1/t schedule (Belardinelli & Pereyra 2007), shared by all walkers of a set:
every --sync-every steps a walker adds its local increments to one shared ln g and histogram under a file lock;
the flatness check (pooled, and every recently active walker must itself have visited every bin in the stage),
the ln f halving, the 1/t switch (t = total steps of all walkers) and the freeze decision are taken on the shared
state. The shared state lives in --shared (node-local) and is persisted to --shared-persist.
Phase "prod": starts when lnf <= --freeze-lnf (or when a file FREEZE exists in the walker dir or its parent).
ln g is frozen, so pi(x) is exactly uniform within each bin, and the walker records
  H_prod[h]           visit histogram
  F[i][j], R[i][j]    every --tm-every steps, the current x in bin i adds its forward edges into bin j (F) and its
                      in-edges from bin j (R) -> transition-matrix estimate from the exact identity
                      N_i <F_ij>_i = N_j <R_ji>_j  (tm.py)
  samples_prod.tsv    step, tour, h, smiles  every --sample-every steps
  tm_hist.jsonl       cumulative counts every --tm-snapshot steps (convergence trace)
Tours (root-to-root excursions; one reaching --hac-max is a round trip) go to tours.tsv.

Refreeze (--refreeze-file F): in production a walker checks F about once a minute. When it holds a new tag
({"tag": ..., "tm": per-bin log10 N}, written by the sampler from the pooled production counts), the walker archives
its production epoch (with its counts) and restarts production with ln g frozen at that estimate. The tm identity
holds for any frozen weights, so the estimate pools all epochs.

Checkpoints to OUT/state.json every --checkpoint steps and on SIGTERM, and resumes from it. Lines written to the
tsv files after the last checkpoint are written again on resume; the analysis dedupes by step.
"""
import argparse
import fcntl
import glob
import json
import math
import os
import random
import signal
import statistics
import time

from cremspace.space import open_space
from cremspace.space.grow import ROOT
from cremspace.util import dump_json


class Shared:
    """shared Wang-Landau state of one set, kept in a JSON file under an flock"""

    def __init__(self, path, persist, nb, seed_glob, cover_frac=1.0):
        self.path, self.persist, self.nb, self.seed_glob = path, persist, nb, seed_glob
        self.cover_frac = cover_frac
        self.bins = list(range(nb))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.n_sync = 0

    def _init(self):
        for f in (self.path, self.persist):
            if f and os.path.exists(f):
                with open(f) as fh:
                    return json.load(fh)
        # seed from existing walker checkpoints of the set (adapt phase), normalised to bin 0
        L, lnfs, t = [], [], 0
        for f in glob.glob(self.seed_glob):
            with open(f) as fh:
                w = json.load(fh)
            t += w["step"]
            if w.get("phase", "adapt") == "adapt" and len(w["lng"]) == self.nb:
                L.append([v - w["lng"][0] for v in w["lng"]])
                lnfs.append(w["lnf"])
        lng = [sum(c) / len(L) for c in zip(*L)] if L else [0.0] * self.nb
        return dict(lng=lng, H=[0] * self.nb, lnf=statistics.median(lnfs) if lnfs else 1.0,
                    t=t, one_over_t=False, frozen=False, stage_log=[["seeded", len(L), t]])

    def sync(self, dlng, dH, dt, flat, freeze_lnf, force_freeze=False, wid=None):
        with open(self.path + ".lock", "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if os.path.exists(self.path):
                with open(self.path) as fh:
                    S = json.load(fh)
            else:
                S = self._init()
            nb = self.nb
            S.setdefault("cover", {})
            S.setdefault("seen", {})
            now = time.time()
            if wid is not None:
                S["seen"][wid] = now
                cv = S["cover"].get(wid, [False] * nb)
                S["cover"][wid] = [c or d > 0 for c, d in zip(cv, dH)]
            if not S["frozen"]:
                S["lng"] = [a + b for a, b in zip(S["lng"], dlng)]
                S["H"] = [a + b for a, b in zip(S["H"], dH)]
                S["t"] += dt
                if not S["one_over_t"]:
                    hv = [S["H"][i] for i in self.bins]
                    # pooled flatness is not enough: walkers confined to different HAC ranges add up to a flat
                    # pooled histogram. Every walker active in the last hour must itself have visited every bin
                    # in this stage (or a cover_frac share of them).
                    active = [w for w, t_ in S["seen"].items() if now - t_ < 3600]
                    n_cov = sum(all(S["cover"].get(w, [False] * nb)[i] for i in self.bins) for w in active)
                    covered = n_cov >= math.ceil(self.cover_frac * len(active) - 1e-9)
                    if min(hv) > 0 and min(hv) >= flat * (sum(hv) / len(hv)) and covered:
                        S["lnf"] /= 2
                        S["H"] = [0] * nb
                        S["cover"] = {}
                        S["stage_log"].append([S["t"], S["lnf"]])
                    if S["t"] > 0 and S["lnf"] < len(self.bins) / S["t"]:
                        S["one_over_t"] = True
                        S["stage_log"].append([S["t"], "1/t"])
                if S["one_over_t"] and S["t"] > 0:
                    S["lnf"] = len(self.bins) / S["t"]
                if S["lnf"] <= freeze_lnf or force_freeze:
                    S["frozen"] = True
                    S["lng_frozen"] = list(S["lng"])
                    S["stage_log"].append([S["t"], "freeze", S["lnf"]])
            dump_json(S, self.path)
            self.n_sync += 1
            if self.persist and (self.n_sync % 10 == 0 or S["frozen"]):
                dump_json(S, self.persist)
        return S


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slice", required=True)
    p.add_argument("--set", required=True)
    p.add_argument("--mode", choices=["grow", "cycle"], default="grow")
    p.add_argument("-o", "--out", required=True, help="walker output dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hac-max", type=int, default=36)
    p.add_argument("--max-atoms", type=int, default=10)
    p.add_argument("--steps", type=int, default=10 ** 12, help="total steps (including resumed ones)")
    p.add_argument("--hours", type=float, default=1e9, help="stop after this wall time")
    p.add_argument("--flat", type=float, default=0.8)
    p.add_argument("--freeze-lnf", type=float, default=1e-3,
                   help="switch to production once the 1/t stage has lnf <= this")
    p.add_argument("--tm-every", type=int, default=5, help="production: edge-count sampling interval")
    p.add_argument("--tm-snapshot", type=int, default=10000,
                   help="production: append cumulative counts to tm_hist.jsonl every this many steps")
    p.add_argument("--checkpoint", type=int, default=500)
    p.add_argument("--sample-every", type=int, default=50)
    p.add_argument("--shared", required=True, help="shared-WL state file (node-local)")
    p.add_argument("--shared-persist", help="persistent copy of the shared state")
    p.add_argument("--sync-every", type=int, default=200)
    p.add_argument("--refreeze-file", help="production: re-freeze ln g at the estimate in this JSON when its tag "
                                           "changes")
    p.add_argument("--max-reject-run", type=int, default=10000,
                   help="reset to the root after this many consecutive rejected moves if the state has no "
                        "reachable parent")
    p.add_argument("--reach-budget", type=int, default=3000)
    p.add_argument("--reach-unknown", choices=["reachable", "unreachable"], default="reachable",
                   help="how a reachability search that exhausts its budget is resolved")
    return p.parse_args(argv)


def run(a):
    os.makedirs(a.out, exist_ok=True)
    stop = {"flag": False}

    def on_term(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    rng = random.Random(a.seed)
    G = open_space(a.slice, a.set, a.mode, max_atoms=a.max_atoms, hac_max=a.hac_max,
                   reach_budget=a.reach_budget, rng=rng, reach_unknown=(a.reach_unknown == "reachable"))
    nb = a.hac_max + 1
    sp = os.path.join(a.out, "state.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            st = json.load(fh)
        rng.setstate(tuple(tuple(x) if isinstance(x, list) else x for x in st["rng"]))
        print(f"resumed at step {st['step']} phase={st['phase']}", flush=True)
    else:
        st = dict(step=0, x=ROOT, lng=[0.0] * nb, H=[0] * nb, visits=[0] * nb, lnf=1.0, one_over_t=False,
                  stage_log=[], acc=dict(grow=[0, 0], prune=[0, 0]), phase="adapt", tour=0, tour_start=0,
                  tour_max_h=0, tour_seed="", n_tours={"adapt": 0, "prod": 0}, n_roundtrips={"adapt": 0, "prod": 0})
    st["args"] = vars(a)
    lng, H, visits = st["lng"], st["H"], st["visits"]
    x = st["x"]
    hx = G.hac(x)
    if x != ROOT:
        G._reach.put(x, True)
        G.proven.add(x)
    t0, t_ck = time.time(), time.time()
    fs = open(os.path.join(a.out, "samples.tsv"), "a")
    ft = open(os.path.join(a.out, "tours.tsv"), "a")
    fp = open(os.path.join(a.out, "samples_prod.tsv"), "a") if st["phase"] == "prod" else None
    set_dir = os.path.dirname(os.path.abspath(a.out))
    freeze_files = [os.path.join(a.out, "FREEZE"), os.path.join(set_dir, "FREEZE")]
    SH = Shared(a.shared, a.shared_persist, nb, os.path.join(set_dir, "w*", "state.json"))
    dlng, dH, dt = [0.0] * nb, [0] * nb, 0

    def save():
        st.update(x=x, lng=lng, H=H, visits=visits, rng=rng.getstate(), stats=G.stats,
                  elapsed=st.get("elapsed", 0) + time.time() - t_ck)
        dump_json(st, sp)

    def freeze(frozen):
        nonlocal fp, lng
        lng = list(frozen)
        st["phase"] = "prod"
        st["freeze_step"] = st["step"]
        st["lng_frozen"] = list(lng)
        st["H_prod"] = [0] * nb
        st["tm_n"] = [0] * nb
        st["tm_F"] = [[0] * nb for _ in range(nb)]
        st["tm_R"] = [[0] * nb for _ in range(nb)]
        st["stage_log"].append([st["step"], "freeze"])
        fp = open(os.path.join(a.out, "samples_prod.tsv"), "a")
        print(f"step {st['step']}: frozen at lnf={st['lnf']:.3g}, production starts", flush=True)

    def archive_epoch(why):
        """move this production epoch aside (files *.e<k>.*, counts in st['prod_epochs'])"""
        nonlocal fp
        ep = len(st.get("prod_epochs", [])) + 1
        if fp:
            fp.close()
            fp = None
        for fn in ("samples_prod.tsv", "tm_hist.jsonl"):
            fpath = os.path.join(a.out, fn)
            if os.path.exists(fpath):
                os.replace(fpath, os.path.join(a.out, fn.replace(".", f".e{ep}.", 1)))
        st.setdefault("prod_epochs", []).append(dict(
            freeze_step=st["freeze_step"], end_step=st["step"], why=why, tag=st.get("refreeze_tag"),
            tm_n=st["tm_n"], H_prod=st["H_prod"], tm_F=st["tm_F"], tm_R=st["tm_R"], lng_frozen=st["lng_frozen"]))
        for k in ("lng_frozen", "H_prod", "tm_n", "tm_F", "tm_R", "freeze_step"):
            st.pop(k, None)
        return ep

    wid = os.path.basename(a.out.rstrip("/"))
    S = SH.sync(dlng, dH, 0, a.flat, a.freeze_lnf, wid=wid)
    if st["phase"] == "adapt":
        lng, st["lnf"] = list(S["lng"]), S["lnf"]
        if S["frozen"]:
            freeze(S["lng_frozen"])
    print(f"joined shared WL state: lnf={S['lnf']:.3g} t={S['t']:,} frozen={S['frozen']}", flush=True)

    def outside(state):
        """state has no reachable parent under the current definition: it is not in S"""
        return state != ROOT and not G.parents(state)

    def reset_to_root(why):
        nonlocal x, hx
        st["resets"] = st.get("resets", 0) + 1
        print(f"step {st['step']}: RESET to root ({why}); state was {x}", flush=True)
        st["tour_max_h"] = -1                     # the interrupted tour is not a round trip
        x, hx = ROOT, 0

    if outside(x):
        reset_to_root("no reachable parent on resume")
    reject_run = 0
    t_rf = 0.0                                    # last look at --refreeze-file

    def refreeze_check():
        nonlocal t_rf
        t_rf = time.time()
        try:
            with open(a.refreeze_file) as fh:
                R_ = json.load(fh)
        except (OSError, ValueError):
            return
        if R_.get("tag") in (None, st.get("refreeze_tag")):
            return
        old = st["lng_frozen"]
        new = [math.log(10) * v if v is not None and math.isfinite(v) else o for v, o in zip(R_["tm"], old)]
        ep = archive_epoch("refreeze")
        st["refreeze_tag"] = R_["tag"]
        freeze(new)
        print(f"step {st['step']}: refreeze {R_['tag']}: epoch {ep} archived, ln g frozen at its tm", flush=True)

    while st["step"] < a.steps and time.time() - t0 < a.hours * 3600 and not stop["flag"]:
        x_prev = x
        if rng.random() < 0.5:
            kind = "grow"
            dpx = G.d_plus(x)
            y = G.sample_child(x) if dpx else None
            if y is not None:
                G._reach.put(y, True)             # child of a reachable node
                ps_y = G.parents(y)
                if x in ps_y:
                    ratio = dpx / len(ps_y)
                else:
                    # one-way edge: the reverse cut cannot see it (unstable SMILES round trip of an absurd DB
                    # ring system). Its reverse proposal probability is 0, so MH rejects it.
                    st["one_way"] = st.get("one_way", 0) + 1
                    y = None
        else:
            kind = "prune"
            ps = G.parents(x)
            y = rng.choice(ps) if ps else None
            if y is not None:
                dpy = G.d_plus(y)
                if dpy:
                    ratio = len(ps) / dpy
                else:                             # cannot happen with forward-visible parents; reject
                    st["one_way_rev"] = st.get("one_way_rev", 0) + 1
                    y = None
        st["acc"][kind][1] += 1
        if y is not None:
            hy = G.hac(y)
            la = lng[hx] - lng[hy] + math.log(ratio)
            if la >= 0 or rng.random() < math.exp(la):
                x, hx = y, hy
                if x != ROOT:
                    G.proven.add(x)
                reject_run = -1
                st["acc"][kind][0] += 1
        reject_run += 1
        if reject_run >= a.max_reject_run:
            # a long rejection run alone is not proof of an absorbing state (a tiny molecule can reject 10k times
            # in a row under poor weights); reset only if the state has no reachable parent, i.e. is not in S
            if outside(x):
                reset_to_root(f"{reject_run} consecutive rejections, no reachable parent")
            reject_run = 0
        st["step"] += 1
        step = st["step"]
        prod = st["phase"] == "prod"

        # tours: root-to-root excursions
        if x_prev == ROOT and x != ROOT:
            st["tour"] += 1
            st["tour_start"], st["tour_max_h"], st["tour_seed"] = step, hx, x
        elif x != ROOT:
            st["tour_max_h"] = max(st["tour_max_h"], hx)
        elif x_prev != ROOT:                      # returned to the root: tour complete
            ph = st["phase"]
            st["n_tours"][ph] += 1
            if st["tour_max_h"] >= a.hac_max:
                st["n_roundtrips"][ph] += 1
            ft.write(f"{ph}\t{st['tour']}\t{st['tour_start']}\t{step}\t{st['tour_max_h']}\t{st['tour_seed']}\n")

        visits[hx] += 1
        if not prod:
            lng[hx] += st["lnf"]
            H[hx] += 1
            if step % a.sample_every == 0 and x != ROOT:
                fs.write(f"{step}\t{hx}\t{x}\n")
            dlng[hx] += st["lnf"]
            dH[hx] += 1
            dt += 1
            if step % a.sync_every == 0:
                force = step % a.checkpoint == 0 and any(os.path.exists(f) for f in freeze_files)
                S = SH.sync(dlng, dH, dt, a.flat, a.freeze_lnf, force, wid=wid)
                dlng, dH, dt = [0.0] * nb, [0] * nb, 0
                lng, st["lnf"] = list(S["lng"]), S["lnf"]
                st["one_over_t"] = S["one_over_t"]
                if S["frozen"]:
                    freeze(S["lng_frozen"])
        else:
            st["H_prod"][hx] += 1
            if step % a.tm_every == 0:
                st["tm_n"][hx] += 1
                for j, c in G.d_plus_by_bin(x).items():
                    st["tm_F"][hx][j] += c
                for j, c in G.d_minus_by_bin(x).items():
                    st["tm_R"][hx][j] += c
            if step % a.sample_every == 0 and x != ROOT:
                fp.write(f"{step}\t{st['tour']}\t{hx}\t{x}\n")
            if (step - st["freeze_step"]) % a.tm_snapshot == 0:
                with open(os.path.join(a.out, "tm_hist.jsonl"), "a") as fh:
                    fh.write(json.dumps(dict(step=step, prod_steps=step - st["freeze_step"],
                                             n=st["tm_n"], F=st["tm_F"], R=st["tm_R"], H=st["H_prod"],
                                             tours=st["n_tours"]["prod"],
                                             roundtrips=st["n_roundtrips"]["prod"])) + "\n")

        if step % a.checkpoint == 0:
            if prod and a.refreeze_file and time.time() - t_rf >= 60:
                refreeze_check()
            save()
            t_ck = time.time()
            for f in (fs, ft, fp):
                if f:
                    f.flush()
            lnN = [(v - lng[0]) / math.log(10) for v in lng]
            print(f"step {step} t={time.time() - t0:.0f}s {st['phase']} lnf={st['lnf']:.2e} h={hx} "
                  f"acc g/p={st['acc']['grow'][0]}/{st['acc']['grow'][1]} "
                  f"{st['acc']['prune'][0]}/{st['acc']['prune'][1]} "
                  f"unvisited={sum(v == 0 for v in visits)} log10N[top]={lnN[-1]:.2f} "
                  f"tours={st['n_tours']} rt={st['n_roundtrips']} one_way={st.get('one_way', 0)} "
                  f"stats={G.stats}", flush=True)
    if st["phase"] == "adapt" and dt:
        S = SH.sync(dlng, dH, dt, a.flat, a.freeze_lnf, wid=wid)
        if SH.persist:
            dump_json(S, SH.persist)
    save()
    for f in (fs, ft, fp):
        if f:
            f.close()
    print(f"stopped at step {st['step']} ({'signal' if stop['flag'] else 'limit'})", flush=True)


if __name__ == "__main__":
    run(parse_args())
