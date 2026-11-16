"""`cremspace sample`: run Wang-Landau / production walkers for one or more sets of a prepared slice.

For every set, --walkers processes walk the space (grow, or grow + make_cycle). They adapt one shared Wang-Landau
density of states, freeze it at ln f <= --freeze-lnf and continue in production, which collects the
transition-matrix counts and the samples used by `cremspace diagnose`. Everything is resumable: run the same command
with the same RUN directory again to continue (and so to extend production).

Speed: walkers do many small random reads of the slice. Before starting, the slice is copied to RAM (/dev/shm) when
it fits in the memory this job may use (cgroup limit, MemAvailable and /dev/shm free space, leaving --walker-mem-gb
per walker), else to node-local disk ($TMPDIR) when that is a different file system with room, else it is read in
place. Concurrent jobs on one node share one copy; the last one to finish removes it.

While walkers run, every --refreeze-every hours the sampler checks each set's production: if the frozen weights
trap the walkers (production HAC occupancy far from flat), it re-freezes them at the current transition-matrix
estimate (see refreeze.py). Walkers stop after --hours; SIGTERM/SIGINT make them checkpoint and stop.
"""
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

from cremspace.refreeze import check as refreeze_check
from cremspace.util import code_version, dump_json, load_json, log, read_meta

# slice_meta keys that define the space; a resumed run must use a slice that agrees on them
SPACE_KEYS = ("format_version", "source", "source_size", "source_mtime", "radius", "max_atoms", "hac_max",
              "min_freq", "modes")
RUN_KEYS = ("mode", "hac_max", "max_atoms", "freeze_lnf", "reach_budget", "reach_unknown")


# ---------------------------------------------------------------------------------------------- memory and placement
def _meminfo_available():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _read_int(path):
    try:
        with open(path) as f:
            v = f.read().strip()
        return None if v in ("max", "") else int(v)
    except (OSError, ValueError):
        return None


def _cgroup_available():
    """bytes this process may still allocate under its cgroup memory limits (None if unlimited/unknown).
    Files in /dev/shm are charged to the cgroup of the process that writes them."""
    try:
        with open("/proc/self/cgroup") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    best = None
    for line in lines:
        hier, ctrl, path = line.split(":", 2)
        if hier == "0" and ctrl == "":                       # cgroup v2
            d = os.path.join("/sys/fs/cgroup", path.lstrip("/"))
            while True:
                lim, cur = _read_int(os.path.join(d, "memory.max")), _read_int(os.path.join(d, "memory.current"))
                if lim is not None and cur is not None:
                    best = lim - cur if best is None else min(best, lim - cur)
                if os.path.normpath(d) == "/sys/fs/cgroup":
                    break
                d = os.path.dirname(d)
        elif "memory" in ctrl.split(","):                     # cgroup v1
            d = os.path.join("/sys/fs/cgroup/memory", path.lstrip("/"))
            lim = _read_int(os.path.join(d, "memory.limit_in_bytes"))
            cur = _read_int(os.path.join(d, "memory.usage_in_bytes"))
            if lim is not None and cur is not None and lim < 1 << 60:
                best = lim - cur if best is None else min(best, lim - cur)
    return best


def _fs_free(path):
    try:
        s = os.statvfs(path)
        return s.f_bavail * s.f_frsize
    except OSError:
        return 0


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def choose_placement(slice_path, n_walkers, copy="auto", walker_mem_gb=2.0):
    """(kind, base_dir, reason); kind in shm, local, none"""
    size = os.path.getsize(slice_path)
    need = int(size * 1.02) + (64 << 20)
    gib = 2 ** 30
    if copy == "none":
        return "none", None, "copy disabled"
    shm_ok = os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)
    mem = [v for v in (_meminfo_available(), _cgroup_available()) if v is not None]
    free_mem = min(mem) if mem else None
    reserve = int(n_walkers * walker_mem_gb * gib)
    if copy in ("auto", "shm") and shm_ok:
        fits = free_mem is not None and need + reserve < free_mem and need < _fs_free("/dev/shm")
        if fits or copy == "shm":
            return "shm", "/dev/shm", (f"slice {size / gib:.1f} GiB + {n_walkers} walkers x {walker_mem_gb:g} GiB "
                                       f"fit in {free_mem / gib if free_mem else float('nan'):.1f} GiB available"
                                       if fits else "forced by --copy shm")
    tmp = os.environ.get("TMPDIR")
    if copy in ("auto", "local") and tmp and os.path.isdir(tmp):
        try:
            same = os.stat(tmp).st_dev == os.stat(os.path.dirname(os.path.abspath(slice_path))).st_dev
        except OSError:
            same = True
        if (not same and need < _fs_free(tmp)) or copy == "local":
            return "local", tmp, f"node-local disk $TMPDIR={tmp}"
    why = (f"slice {size / gib:.1f} GiB + {n_walkers} walkers x {walker_mem_gb:g} GiB do not fit in "
           f"{free_mem / gib:.1f} GiB available" if free_mem is not None else "available memory unknown")
    return "none", None, f"reading in place ({why}; no separate node-local disk)"


class SliceCopy:
    """a copy of the slice shared by the cremspace processes of one node: base/cremspace-<key>/slice.db with one
    file per user process in users/; the last user removes it"""

    def __init__(self, slice_path, base):
        st = os.stat(slice_path)
        key = hashlib.sha1(f"{os.path.abspath(slice_path)}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
        self.src = slice_path
        self.dir = os.path.join(base, f"cremspace-{key}")
        self.path = os.path.join(self.dir, "slice.db")
        self.lock = os.path.join(base, f"cremspace-{key}.lock")
        self.me = os.path.join(self.dir, "users", str(os.getpid()))

    def _prune(self):
        ud = os.path.join(self.dir, "users")
        live = []
        for f in os.listdir(ud) if os.path.isdir(ud) else []:
            if f.isdigit() and _alive(int(f)):
                live.append(f)
            else:
                try:
                    os.remove(os.path.join(ud, f))
                except OSError:
                    pass
        return live

    def acquire(self):
        with open(self.lock, "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            os.makedirs(os.path.join(self.dir, "users"), exist_ok=True)
            self._prune()
            if not os.path.exists(self.path):
                t = time.time()
                tmp = self.path + ".partial"
                shutil.copyfile(self.src, tmp)
                os.replace(tmp, self.path)
                log(f"slice copied to {self.path}", t)
            else:
                log(f"using the existing copy {self.path}")
            open(self.me, "w").close()
        return self.path

    def release(self):
        try:
            with open(self.lock, "a+") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                try:
                    os.remove(self.me)
                except OSError:
                    pass
                if not self._prune():
                    shutil.rmtree(self.dir, ignore_errors=True)
                    log(f"removed {self.dir}")
        except OSError as e:
            log(f"could not clean up {self.dir}: {e}")


# ---------------------------------------------------------------------------------------------- run directory
def _check_run(run_dir, meta, params, sets, walkers):
    rp = os.path.join(run_dir, "run.json")
    run = load_json(rp)
    if run is None:
        run = dict(params, slice_meta={k: meta.get(k) for k in SPACE_KEYS}, sets=[], walkers={}, invocations=[])
    else:
        bad = [f"{k}: run {run.get(k)!r} vs now {params[k]!r}" for k in RUN_KEYS if run.get(k) != params[k]]
        bad += [f"slice {k}: run {run['slice_meta'].get(k)!r} vs now {meta.get(k)!r}"
                for k in SPACE_KEYS if str(run["slice_meta"].get(k)) != str(meta.get(k))]
        if bad:
            raise SystemExit(f"{run_dir} was sampled with other settings; use a new RUN directory or the same "
                             "settings:\n  " + "\n  ".join(bad))
    for s in sets:
        if s not in run["sets"]:
            run["sets"].append(s)
        run["walkers"][s] = max(walkers, run["walkers"].get(s, 0))
    run["slice"] = params["slice"]
    return run


def sample(slice_path, sets, mode, run_dir, walkers=16, hours=72.0, hac_max=None, max_atoms=None,
           freeze_lnf=1e-3, copy="auto", walker_mem_gb=2.0, refreeze=True, refreeze_every=3.0,
           refreeze_min_steps=10_000_000, reach_budget=3000, reach_unknown="reachable", max_steps=None):
    slice_path = os.path.abspath(slice_path)
    run_dir = os.path.abspath(run_dir)
    meta = read_meta(slice_path)
    slice_sets = meta["sets"].split(",")
    sets = sets or slice_sets
    missing = [s for s in sets if s not in slice_sets]
    if missing:
        raise SystemExit(f"sets {missing} are not in the slice; its sets: {slice_sets}")
    if mode == "cycle" and "cycle" not in meta.get("modes", "").split(","):
        raise SystemExit("the slice was prepared with --no-cycle; prepare it again for --mode cycle")
    hac_max = int(meta["hac_max"]) if hac_max is None else hac_max
    max_atoms = int(meta["max_atoms"]) if max_atoms is None else max_atoms
    if hac_max > int(meta["hac_max"]) or max_atoms > int(meta["max_atoms"]):
        raise SystemExit(f"--hac-max/--max-atoms exceed the slice's {meta['hac_max']}/{meta['max_atoms']}")
    params = dict(slice=slice_path, mode=mode, hac_max=hac_max, max_atoms=max_atoms, freeze_lnf=freeze_lnf,
                  reach_budget=reach_budget, reach_unknown=reach_unknown)
    os.makedirs(run_dir, exist_ok=True)

    # one sampler per RUN directory at a time
    lockf = open(os.path.join(run_dir, ".lock"), "a+")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another `cremspace sample` is running on {run_dir}")
    run = _check_run(run_dir, meta, params, sets, walkers)

    n_total = walkers * len(sets)
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 1
    if n_total > ncpu:
        log(f"WARNING: {n_total} walkers on {ncpu} CPUs; walkers will share CPUs")

    kind, base, why = choose_placement(slice_path, n_total, copy, walker_mem_gb)
    log(f"slice placement: {kind} ({why})")
    sc = SliceCopy(slice_path, base) if kind != "none" else None
    use_path = sc.acquire() if sc else slice_path
    live_base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else run_dir
    live = os.path.join(live_base, f"cremspace-run-{hashlib.sha1(run_dir.encode()).hexdigest()[:12]}-{os.getpid()}")
    os.makedirs(live, exist_ok=True)

    run["invocations"].append(dict(start=time.strftime("%Y-%m-%d %H:%M:%S"), host=socket.gethostname(),
                                   hours=hours, walkers=walkers, sets=sets, placement=kind, version=code_version()))
    dump_json(run, os.path.join(run_dir, "run.json"), indent=1)

    procs = []
    for s in sets:
        sd = os.path.join(run_dir, s)
        os.makedirs(sd, exist_ok=True)
        for i in range(1, walkers + 1):
            out = os.path.join(sd, f"w{i:02d}")
            os.makedirs(out, exist_ok=True)
            cmd = [sys.executable, "-m", "cremspace.walker", "--slice", use_path, "--set", s, "--mode", mode,
                   "-o", out, "--seed", str(i), "--hac-max", str(hac_max), "--max-atoms", str(max_atoms),
                   "--hours", str(hours), "--freeze-lnf", str(freeze_lnf),
                   "--shared", os.path.join(live, f"{s}.json"), "--shared-persist", os.path.join(sd, "shared.json"),
                   "--refreeze-file", os.path.join(sd, "REFREEZE.json"),
                   "--reach-budget", str(reach_budget), "--reach-unknown", reach_unknown]
            if max_steps:
                cmd += ["--steps", str(max_steps)]
            lf = open(os.path.join(out, "walker.log"), "a")
            procs.append((s, i, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), lf))
    log(f"{len(procs)} walkers started: {len(sets)} set(s) x {walkers}, mode {mode}, HAC <= {hac_max}, "
        f"{hours:g} h; logs in {run_dir}/<set>/w*/walker.log")

    stop = {"sig": None}

    def on_sig(signum, frame):
        stop["sig"] = signum
    old = {sig: signal.signal(sig, on_sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    rf_next = {s: time.time() + refreeze_every * 3600 for s in sets} if refreeze else {}
    try:
        while any(p.poll() is None for _, _, p, _ in procs):
            if stop["sig"] is not None:
                log(f"signal {stop['sig']}: stopping walkers (they checkpoint first)")
                for _, _, p, _ in procs:
                    if p.poll() is None:
                        p.send_signal(signal.SIGTERM)
                t_end = time.time() + 120
                while time.time() < t_end and any(p.poll() is None for _, _, p, _ in procs):
                    time.sleep(1)
                for _, _, p, _ in procs:
                    if p.poll() is None:
                        p.kill()
                break
            now = time.time()
            for s in list(rf_next):
                if now >= rf_next[s]:
                    sd = os.path.join(run_dir, s)
                    status, msg = refreeze_check(sd, os.path.join(sd, "REFREEZE.json"), min_steps=refreeze_min_steps)
                    with open(os.path.join(sd, "refreeze.log"), "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
                    if status in ("refreeze",):
                        log(f"{s}: {msg}")
                    if status in ("flat", "limit"):
                        del rf_next[s]
                    else:
                        rf_next[s] = now + refreeze_every * 3600
            time.sleep(5)
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
        for s in sets:
            src = os.path.join(live, f"{s}.json")
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(run_dir, s, "shared.json.tmp"))
                os.replace(os.path.join(run_dir, s, "shared.json.tmp"), os.path.join(run_dir, s, "shared.json"))
        shutil.rmtree(live, ignore_errors=True)
        if sc:
            sc.release()
        for _, _, _, lf in procs:
            lf.close()
    failed = [(s, i, p.returncode) for s, i, p, _ in procs if p.returncode not in (0, None, -signal.SIGTERM)]
    for s, i, rc in failed:
        log(f"walker {s}/w{i:02d} exited with code {rc}; see its walker.log")
    log(f"done; next: cremspace diagnose {run_dir}")
    return 1 if failed else 0


def add_parser(sub):
    import argparse
    p = sub.add_parser("sample", help="run the walkers (resumable: rerun to continue)", description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("slice", help="slice from `cremspace prepare`")
    p.add_argument("--sets", help="comma-separated sets (default: all sets of the slice)")
    p.add_argument("--mode", choices=["grow", "cycle"], default="grow",
                   help="grow: grow_mol only; cycle: grow_mol + make_cycle(ring_closures=True)")
    p.add_argument("-o", "--out", required=True, help="RUN directory (created; reused to resume)")
    p.add_argument("--walkers", type=int, default=16, help="walkers per set")
    p.add_argument("--hours", type=float, default=72, help="wall time of this invocation")
    p.add_argument("--hac-max", type=int, help="largest molecule (heavy atoms); default: the slice's")
    p.add_argument("--max-atoms", type=int, help="largest core added in one step; default: the slice's")
    p.add_argument("--freeze-lnf", type=float, default=1e-3, help="Wang-Landau ln f at which production starts")
    p.add_argument("--copy", choices=["auto", "shm", "local", "none"], default="auto",
                   help="where to copy the slice before sampling (auto: RAM if it fits, else node-local disk)")
    p.add_argument("--walker-mem-gb", type=float, default=2.0, help="memory to leave per walker when deciding "
                                                                    "whether the slice fits in RAM")
    p.add_argument("--no-refreeze", action="store_true", help="disable the automatic refreeze")
    p.add_argument("--refreeze-every", type=float, default=3.0, help="hours between refreeze checks")
    p.add_argument("--refreeze-min-steps", type=int, default=10_000_000,
                   help="pooled production steps of an epoch before it is judged")
    p.add_argument("--reach-budget", type=int, default=3000,
                   help="node budget of the reachability search of a parent")
    p.add_argument("--reach-unknown", choices=["reachable", "unreachable"], default="reachable",
                   help="how a search that exhausts its budget is resolved")
    p.add_argument("--max-steps", type=int, help="stop each walker after this many steps in total (testing)")
    p.set_defaults(func=lambda a: sample(
        a.slice, a.sets.split(",") if a.sets else None, a.mode, a.out, walkers=a.walkers, hours=a.hours,
        hac_max=a.hac_max, max_atoms=a.max_atoms, freeze_lnf=a.freeze_lnf, copy=a.copy,
        walker_mem_gb=a.walker_mem_gb, refreeze=not a.no_refreeze, refreeze_every=a.refreeze_every,
        refreeze_min_steps=a.refreeze_min_steps, reach_budget=a.reach_budget, reach_unknown=a.reach_unknown,
        max_steps=a.max_steps))
