"""Small helpers shared by the commands."""
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import OrderedDict
from urllib.request import pathname2url


def log(msg, t0=None, file=sys.stderr):
    el = f"  [{time.time() - t0:.0f}s]" if t0 is not None else ""
    print(f"{time.strftime('%H:%M:%S')} {msg}{el}", file=file, flush=True)


def qi(name):
    """quote an SQL identifier"""
    return '"' + name.replace('"', '""') + '"'


def sqlite_uri(path, **params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    return f"file:{pathname2url(os.path.abspath(path))}" + (f"?{q}" if q else "")


def connect_ro(path, immutable=True):
    """read-only connection. immutable=1 skips file locking: with many processes on one file, mode=ro alone
    takes and drops a shared lock per statement (~1 ms per lookup instead of ~3 us). Only for files that
    nothing writes while they are open (a prepared slice)."""
    params = dict(mode="ro")
    if immutable:
        params["immutable"] = 1
    return sqlite3.connect(sqlite_uri(path, **params), uri=True, check_same_thread=False)


def dump_json(obj, path, **kw):
    """write path atomically; a failed write raises (an unchecked close can leave an empty file)"""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


def load_json(path, retries=10, default=None):
    """json.load that tolerates a file being replaced concurrently"""
    for _ in range(retries):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except (OSError, ValueError):
            time.sleep(1)
    return default


def read_meta(slice_path):
    con = connect_ro(slice_path)
    try:
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "slice_meta" not in names:
            raise ValueError(f"{slice_path} is not a cremspace slice (run `cremspace prepare` first)")
        return dict(con.execute("SELECT key, value FROM slice_meta"))
    finally:
        con.close()


def code_version():
    """package version, plus the git commit when running from a checkout"""
    from cremspace import __version__
    v = dict(cremspace=__version__)
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run(["git", "-C", here, "describe", "--always", "--dirty", "--tags"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            v["git"] = r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import crem
        v["crem"] = getattr(crem, "__version__", "?")
    except ImportError:
        pass
    try:
        import rdkit
        v["rdkit"] = rdkit.__version__
    except ImportError:
        pass
    return v


class LRU(OrderedDict):
    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize

    def get(self, k, default=None):
        if k in self:
            self.move_to_end(k)
            return self[k]
        return default

    def put(self, k, v):
        self[k] = v
        self.move_to_end(k)
        if len(self) > self.maxsize:
            self.popitem(last=False)
