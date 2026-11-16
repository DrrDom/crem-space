"""`cremspace prepare`: extract the part of a CReM v2 database that the sampler needs into one SQLite file.

grow_mol replaces one H by a single-attachment core; make_cycle(ring_closures=True) joins two H-bearing atoms by a
ring-arc core (two attachment points carrying the v2 ring-cut label). For one radius R the slice keeps

  radius{R}       the rows of both kinds with 1 <= core_num_atoms <= MAX_ATOMS (grow rows also dist2 = 0) and
                  count >= max(1, MIN_FREQ) in at least one selected set; the selected set columns only
  envs, frags,    the environments and cores those rows reference (ids are the source's), and their H-capped
  frags_h         cores
  seeds           the walk's starting molecules: the H-capped cores (frags_h) referenced by *any* radius-R row
                  of a set (any size, any number of attachment points), canonical, 1 <= HAC <= HAC_MAX, with a
                  bitmask of the sets that reference them
  pe_i, pq_i      packed grow cores of set i: per env the first row and cumulative counts by core size, so the
                  r-th core of an env is one primary-key lookup (radius-1 envs can have millions of cores)
  ce_i, cq_i      the same for the ring-arc cores
  cinv            composition key of every ring-arc env -> set mask (pre-filter for anchor pairs, cycle.py)
  slice_meta      parameters and provenance

The slice is itself a valid CReM v2 database: crem.grow_mol and crem.make_cycle run on it unchanged and give the
same products as on the source (within the size limits), which the tests use. The source is opened read-only and
scanned sequentially twice (rows, then seed masks); it is never modified. The slice is built in --tmp-dir (or next
to the output) and renamed into place when complete.
"""
import os
import shutil
import sqlite3
import time
from multiprocessing import Pool

import numpy as np

from cremspace import __version__
from cremspace.util import log, qi, sqlite_uri

RESERVED = {"env_id", "core_smi_id", "core_num_atoms", "dist2", "is_ring_closure"}


def _canon(item):
    h_id, smi = item
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return h_id, None, 0
    return h_id, Chem.MolToSmiles(m), m.GetNumHeavyAtoms()


_RADIUS = None


def _key_init(radius):
    global _RADIUS
    _RADIUS = radius
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _keys(rows):
    from cremspace.space.cycle import env_key
    return [(env_key(env, _RADIUS), mask) for env, mask in rows]


def _pack(cur, src_table, kind_sql, set_col, thr, K, pe, pq):
    """packed per-env layout of one set: pq(rid, core_smi_id) ordered by (env, size, core), pe(env_id, start, c1..cK)"""
    cur.execute("DROP TABLE IF EXISTS temp.tmp_p")
    cur.execute("CREATE TEMP TABLE tmp_p(rid INTEGER PRIMARY KEY, env_id INTEGER, n INTEGER, core_smi_id INTEGER)")
    cur.execute(f"""INSERT INTO tmp_p(env_id, n, core_smi_id)
                    SELECT r.env_id, r.core_num_atoms, r.core_smi_id FROM {src_table} r
                    JOIN frags f ON f.core_smi_id = r.core_smi_id
                    WHERE {kind_sql} AND r.{qi(set_col)} >= {thr}
                    ORDER BY r.env_id, r.core_num_atoms, r.core_smi_id""")
    cums = ", ".join(f"SUM(n <= {k})" for k in range(1, K + 1))
    cdl = ", ".join(f"c{k} INTEGER" for k in range(1, K + 1))
    cur.execute(f"CREATE TABLE {pe}(env_id INTEGER PRIMARY KEY, start INTEGER, {cdl})")
    cur.execute(f"INSERT INTO {pe} SELECT env_id, MIN(rid), {cums} FROM tmp_p GROUP BY env_id")
    cur.execute(f"CREATE TABLE {pq}(rid INTEGER PRIMARY KEY, core_smi_id INTEGER NOT NULL)")
    cur.execute(f"INSERT INTO {pq} SELECT rid, core_smi_id FROM tmp_p")
    cur.execute("DROP TABLE temp.tmp_p")
    n = cur.execute(f"SELECT count(*) FROM {pq}").fetchone()[0]
    ne = cur.execute(f"SELECT count(*) FROM {pe}").fetchone()[0]
    return n, ne


def prepare(source, radius, output, sets=None, cycle=True, max_atoms=10, hac_max=36, min_freq=1,
            tmp_dir=None, ncpu=4, cache_mb=4096, overwrite=False):
    from crem.db import get_db_info

    source = os.path.abspath(source)
    output = os.path.abspath(output)
    if os.path.exists(output) and not overwrite:
        raise FileExistsError(f"{output} exists (use --overwrite to replace it)")
    info = get_db_info(source)
    if info["version"] != 2:
        raise ValueError(f"{source}: CReM database version {info['version']}; only v2 databases are supported")
    rt = f"radius{radius}"
    if rt not in info["radius_tables"]:
        raise ValueError(f"{source} has no table {rt}; radius tables: {sorted(info['radius_tables'])}")
    available = info["radius_tables"][rt]
    if not available:
        raise ValueError(f"{source}: {rt} has no set (count) columns")
    if sets:
        missing = [s for s in sets if s not in available]
        if missing:
            raise ValueError(f"sets {missing} not in {rt}; available: {available}")
    else:
        sets = list(available)
    if len(sets) > 62:
        raise ValueError("at most 62 sets per slice")
    thr = max(1, int(min_freq))
    K = max_atoms

    work_dir = tmp_dir or os.path.dirname(output)
    os.makedirs(work_dir, exist_ok=True)
    work = os.path.join(work_dir, os.path.basename(output) + ".partial")
    for stale in (work, work + "-journal"):
        if os.path.exists(stale):
            os.remove(stale)
    t_all = time.time()
    log(f"source {source}, {rt}, sets {sets}, cores 1..{K} atoms, seeds HAC <= {hac_max}, "
        f"count >= {thr}, cycle={cycle}")
    con = sqlite3.connect(sqlite_uri(work), uri=True)
    cur = con.cursor()
    for p in ("journal_mode = OFF", "synchronous = OFF", "page_size = 8192", "temp_store = FILE",
              f"cache_size = -{cache_mb * 1024 // 2}"):
        cur.execute(f"PRAGMA {p}")
    if tmp_dir:
        cur.execute(f"PRAGMA temp_store_directory = '{tmp_dir}'")
    cur.execute(f"ATTACH DATABASE '{sqlite_uri(source, mode='ro')}' AS src")
    cur.execute(f"PRAGMA src.cache_size = -{cache_mb * 1024 // 2}")
    cur.execute("PRAGMA src.mmap_size = 0")

    any_set = " OR ".join(f"{qi(s)} >= {thr}" for s in sets)

    # ---- cores by kind (cheap: frags is far smaller than the radius tables) ----
    t = time.time()
    # every attachment point carries a map number: a single-attachment core has no ':2]'
    cur.execute("CREATE TEMP TABLE core1(core_smi_id INTEGER PRIMARY KEY)")
    cur.execute("INSERT INTO core1 SELECT core_smi_id FROM src.frags WHERE instr(core_smi, ':2]') = 0")
    if cycle:
        cur.execute("CREATE TEMP TABLE core2(core_smi_id INTEGER PRIMARY KEY)")
        cur.execute("""INSERT INTO core2 SELECT core_smi_id FROM src.frags
                       WHERE instr(core_smi, '[1*:1]') > 0 AND instr(core_smi, '[1*:2]') > 0
                         AND instr(core_smi, ':3]') = 0""")
    n1 = cur.execute("SELECT count(*) FROM core1").fetchone()[0]
    n2 = cur.execute("SELECT count(*) FROM core2").fetchone()[0] if cycle else 0
    log(f"cores: {n1:,} single-attachment, {n2:,} ring-arc", t)

    # ---- rows: one sequential pass over src.radiusR ----
    t = time.time()
    set_ddl = ", ".join(f"{qi(s)} INTEGER NOT NULL DEFAULT 0" for s in sets)
    cur.execute(f"""CREATE TABLE {rt}(env_id INTEGER NOT NULL, core_smi_id INTEGER NOT NULL,
                    core_num_atoms INTEGER NOT NULL, dist2 INTEGER NOT NULL, {set_ddl})""")
    cols = ", ".join(["env_id", "core_smi_id", "core_num_atoms", "dist2"] + [qi(s) for s in sets])
    kind = "(dist2 = 0 AND core_smi_id IN (SELECT core_smi_id FROM core1))"
    if cycle:
        kind = f"({kind} OR core_smi_id IN (SELECT core_smi_id FROM core2))"
    # NOT INDEXED: the planner can otherwise pick a skip-scan of the (env_id, ...) index, i.e. random reads over
    # the whole file (radius 1); a sequential scan is always right here
    cur.execute(f"""INSERT INTO {rt} SELECT {cols} FROM src.{rt} NOT INDEXED
                    WHERE core_num_atoms BETWEEN 1 AND {K} AND {kind} AND ({any_set})""")
    con.commit()
    n_rows = cur.execute(f"SELECT count(*) FROM {rt}").fetchone()[0]
    log(f"{rt}: {n_rows:,} rows kept", t)
    cur.execute("DROP TABLE temp.core1")
    if cycle:
        cur.execute("DROP TABLE temp.core2")

    # ---- referenced envs, cores, H-capped cores (primary-key lookups into the source) ----
    t = time.time()
    cur.execute(f"CREATE UNIQUE INDEX idx_{rt}_pair ON {rt}(env_id, core_smi_id)")
    cur.execute(f"CREATE INDEX idx_{rt}_lookup ON {rt}(env_id, core_num_atoms, dist2)")
    cur.execute(f"CREATE INDEX tmp_{rt}_core ON {rt}(core_smi_id)")
    cur.execute("CREATE TABLE envs(env_id INTEGER PRIMARY KEY, env TEXT NOT NULL)")
    cur.execute(f"""INSERT INTO envs SELECT e.env_id, e.env FROM (SELECT DISTINCT env_id FROM {rt}) d
                    JOIN src.envs e ON e.env_id = d.env_id""")
    cur.execute("CREATE TABLE frags(core_smi_id INTEGER PRIMARY KEY, core_smi TEXT NOT NULL, "
                "core_smi_h_id INTEGER NOT NULL)")
    cur.execute(f"""INSERT INTO frags SELECT f.core_smi_id, f.core_smi, f.core_smi_h_id
                    FROM (SELECT DISTINCT core_smi_id FROM {rt}) d JOIN src.frags f ON f.core_smi_id = d.core_smi_id""")
    cur.execute("CREATE TABLE frags_h(core_smi_h_id INTEGER PRIMARY KEY, smi TEXT NOT NULL)")
    cur.execute("""INSERT INTO frags_h SELECT h.core_smi_h_id, h.smi
                   FROM (SELECT DISTINCT core_smi_h_id FROM frags) d JOIN src.frags_h h ON h.core_smi_h_id = d.core_smi_h_id""")
    cur.execute(f"DROP INDEX tmp_{rt}_core")
    cur.execute("CREATE UNIQUE INDEX idx_envs_env ON envs(env)")
    cur.execute("CREATE UNIQUE INDEX idx_frags_core ON frags(core_smi)")
    cur.execute("CREATE UNIQUE INDEX idx_frags_h_smi ON frags_h(smi)")
    con.commit()
    n_env = cur.execute("SELECT count(*) FROM envs").fetchone()[0]
    n_frag = cur.execute("SELECT count(*) FROM frags").fetchone()[0]
    log(f"envs {n_env:,}, frags {n_frag:,}", t)

    # ---- packed layouts ----
    per_set = {s: {} for s in sets}
    single, ring_arc = "instr(f.core_smi, ':2]') = 0 AND r.dist2 = 0", "instr(f.core_smi, ':2]') > 0"
    for i, s in enumerate(sets):
        t = time.time()
        n, ne = _pack(cur, rt, single, s, thr, K, f"pe_{i}", f"pq_{i}")
        per_set[s].update(grow_rows=n, grow_envs=ne)
        if cycle:
            n, ne = _pack(cur, rt, ring_arc, s, thr, K, f"ce_{i}", f"cq_{i}")
            per_set[s].update(cycle_rows=n, cycle_envs=ne)
        con.commit()
        log(f"packed set {s}: {per_set[s]}", t)

    # ---- seeds: frags_h referenced by any radius-R row of a set (second pass over the source) ----
    t = time.time()
    dt = np.uint64
    max_core = cur.execute("SELECT max(core_smi_id) FROM src.frags").fetchone()[0] or 0
    cmask = np.zeros(max_core + 1, dtype=dt)
    mask_expr = " | ".join(f"(({qi(s)} >= {thr}) << {i})" for i, s in enumerate(sets))
    c2 = con.cursor()
    c2.execute(f"SELECT core_smi_id, {mask_expr} FROM src.{rt} NOT INDEXED WHERE {any_set}")
    n = 0
    while True:
        rows = c2.fetchmany(2_000_000)
        if not rows:
            break
        a = np.asarray(rows, dtype=np.int64)
        np.bitwise_or.at(cmask, a[:, 0], a[:, 1].astype(dt))
        n += len(rows)
        if n % 100_000_000 < 2_000_000:
            log(f"  seed masks: {n:,} rows", t)
    max_h = cur.execute("SELECT max(core_smi_h_id) FROM src.frags_h").fetchone()[0] or 0
    hmask = np.zeros(max_h + 1, dtype=dt)
    c2.execute("SELECT core_smi_id, core_smi_h_id FROM src.frags")
    while True:
        rows = c2.fetchmany(5_000_000)
        if not rows:
            break
        a = np.asarray(rows, dtype=np.int64)
        np.bitwise_or.at(hmask, a[:, 1], cmask[a[:, 0]])
    del cmask
    log(f"  {n:,} rows of src.{rt} read; {int((hmask > 0).sum()):,} H-capped cores referenced", t)

    t = time.time()
    cur.execute("CREATE TABLE seeds(seed_id INTEGER PRIMARY KEY, smi TEXT NOT NULL UNIQUE, "
                "hac INTEGER NOT NULL, mask INTEGER NOT NULL)")
    sql = ("INSERT INTO seeds(smi, hac, mask) VALUES (?,?,?) "
           "ON CONFLICT(smi) DO UPDATE SET mask = mask | excluded.mask")
    c3 = con.cursor()
    c3.execute("SELECT core_smi_h_id, smi FROM src.frags_h")
    # sqlite objects are thread-bound and the pool feeds tasks from a helper thread: read in the main thread,
    # hand the pool plain lists
    with Pool(ncpu) as pool:
        while True:
            rows = c3.fetchmany(1_000_000)
            if not rows:
                break
            todo = [(h, smi) for h, smi in rows if hmask[h]]
            batch = [(smi, hac, int(hmask[h_id])) for h_id, smi, hac in pool.map(_canon, todo, chunksize=2000)
                     if smi is not None and 1 <= hac <= hac_max]
            cur.executemany(sql, batch)
    cur.execute("CREATE INDEX idx_seeds_hac ON seeds(hac)")
    con.commit()
    for i, s in enumerate(sets):
        per_set[s]["seeds"] = cur.execute(f"SELECT count(*) FROM seeds WHERE mask & {1 << i}").fetchone()[0]
    log(f"seeds: {cur.execute('SELECT count(*) FROM seeds').fetchone()[0]:,}", t)

    # ---- composition keys of ring-arc envs ----
    if cycle:
        t = time.time()
        mexpr = " | ".join(f"(MAX(r.{qi(s)} >= {thr}) << {i})" for i, s in enumerate(sets))
        cur.execute("CREATE TEMP TABLE tmp_k(key TEXT NOT NULL, mask INTEGER NOT NULL)")
        rd = con.cursor()
        rd.execute(f"""SELECT e.env, m.mask FROM
                         (SELECT r.env_id, {mexpr} AS mask FROM {rt} r JOIN frags f ON f.core_smi_id = r.core_smi_id
                          WHERE {ring_arc} GROUP BY r.env_id) m
                       JOIN envs e ON e.env_id = m.env_id""")
        n_env = n_bad = 0
        with Pool(ncpu, initializer=_key_init, initargs=(radius,)) as pool:
            while True:
                block = rd.fetchmany(400_000)
                if not block:
                    break
                for res in pool.imap_unordered(_keys, [block[i:i + 5000] for i in range(0, len(block), 5000)]):
                    good = [r for r in res if r[0] is not None]
                    n_bad += len(res) - len(good)
                    n_env += len(res)
                    cur.executemany("INSERT INTO tmp_k VALUES (?,?)", good)
        bit_or = " | ".join(f"MAX(mask & {1 << i})" for i in range(len(sets)))
        cur.execute("CREATE TABLE cinv(key TEXT PRIMARY KEY, mask INTEGER NOT NULL) WITHOUT ROWID")
        cur.execute(f"INSERT INTO cinv SELECT key, {bit_or} FROM tmp_k GROUP BY key")
        cur.execute("DROP TABLE temp.tmp_k")
        con.commit()
        n_keys = cur.execute("SELECT count(*) FROM cinv").fetchone()[0]
        log(f"ring-arc envs {n_env:,} -> {n_keys:,} composition keys ({n_bad} without two attachment points)", t)

    st = os.stat(source)
    try:
        import crem
        crem_v = getattr(crem, "__version__", "?")
    except ImportError:
        crem_v = "?"
    meta = dict(format="cremspace-slice", format_version=1, source=source, source_size=st.st_size,
                source_mtime=int(st.st_mtime), radius=radius, sets=",".join(sets), max_atoms=K, hac_max=hac_max,
                min_freq=thr, modes="grow,cycle" if cycle else "grow", built=time.strftime("%Y-%m-%d %H:%M:%S"),
                cremspace=__version__, crem=crem_v)
    cur.execute("CREATE TABLE slice_meta(key TEXT PRIMARY KEY, value TEXT)")
    cur.executemany("INSERT INTO slice_meta VALUES (?,?)", [(k, str(v)) for k, v in meta.items()])
    cur.execute("PRAGMA user_version = 2")
    con.commit()
    cur.execute("DETACH DATABASE src")
    con.close()
    shutil.move(work, output)
    log(f"{output}: {os.path.getsize(output) / 2 ** 20:,.1f} MiB", t_all)
    for s in sets:
        log(f"  {s}: {per_set[s]}")
    return per_set


def add_parser(sub):
    p = sub.add_parser("prepare", help="extract the part of a CReM v2 database the sampler needs (one file)",
                       description=__doc__, formatter_class=_fmt())
    p.add_argument("-d", "--database", required=True, help="CReM v2 fragment database")
    p.add_argument("-r", "--radius", type=int, required=True, help="context radius")
    p.add_argument("-o", "--output", required=True, help="output slice (SQLite file)")
    p.add_argument("--sets", help="comma-separated set columns to keep (default: all)")
    p.add_argument("--no-cycle", action="store_true", help="grow only: skip the make_cycle (ring-arc) rows")
    p.add_argument("--max-atoms", type=int, default=10, help="largest core (heavy atoms) used by grow/make_cycle")
    p.add_argument("--hac-max", type=int, default=36, help="largest seed (heavy atoms); the sampling cap must "
                                                           "not exceed it")
    p.add_argument("--min-freq", type=int, default=1,
                   help="use rows with count >= this in the set (as crem's min_freq; at least 1)")
    p.add_argument("--tmp-dir", default=os.environ.get("TMPDIR") or None,
                   help="build here (node-local disk) and move the finished file to --output")
    p.add_argument("-c", "--ncpu", type=int, default=4)
    p.add_argument("--cache-mb", type=int, default=4096, help="SQLite page cache")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=lambda a: prepare(a.database, a.radius, a.output,
                                          sets=a.sets.split(",") if a.sets else None, cycle=not a.no_cycle,
                                          max_atoms=a.max_atoms, hac_max=a.hac_max, min_freq=a.min_freq,
                                          tmp_dir=a.tmp_dir, ncpu=a.ncpu, cache_mb=a.cache_mb,
                                          overwrite=a.overwrite))


def _fmt():
    import argparse
    return argparse.RawDescriptionHelpFormatter
