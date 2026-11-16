# CReM-space

How many molecules can a CReM fragment database generate? `cremspace` estimates the size of the chemical space
reachable with [CReM](https://github.com/DrrDom/crem) from the fragments of one database, without enumerating it.

The space is defined by CReM's own operations:

- **grow**: `grow_mol` replaces one hydrogen by a fragment (core) of the database whose context matches;
- **grow + make_cycle**: additionally `make_cycle(ring_closures=True)` joins two hydrogen-bearing atoms by a
  ring-arc fragment and closes a ring.

Starting molecules (seeds) are the hydrogen-capped fragments of the database. The space contains every molecule
reachable from a seed by these operations up to a heavy-atom cap (default 36), with cores of at most 10 heavy atoms
per step. The result is the number of molecules N(h) for every heavy-atom count h = 1..cap and their total.

The estimate comes from a Monte Carlo walk over the space (Wang-Landau sampling, then a transition-matrix estimate
from an exact balance identity). Errors are typically 0.01-0.05 in log10 N, i.e. a few percent, for spaces of
10^10-10^25 molecules.

## Installation

```
conda create -n cremspace -c conda-forge python=3.12 rdkit numpy scipy
conda activate cremspace
pip install crem            # >= 0.3.2
pip install .               # from this repository
cremspace --help
```

`cremspace` needs a **CReM v2 fragment database** (created by `cremdb_create` or `crem.db.create_db` of CReM 0.3
or later). Databases of older formats have to be rebuilt.

## Workflow

```
cremspace prepare   -d crem_v2.db -r 5 -o slice_r5.db                   # once per database and radius
cremspace sample    slice_r5.db --sets A,B --mode grow -o runs/r5_grow   # the long part; rerun to extend
cremspace diagnose  runs/r5_grow                                        # estimate + verdict per set
cremspace table     runs/r5_grow runs/r5_cycle > estimates.tsv          # final table
```

`cremspace status RUN` shows the progress of a running job; `cremspace exact` enumerates a small space exactly
(validation).

### 1. `prepare`: extract what the sampler needs

```
cremspace prepare -d crem_v2.db -r 5 -o slice_r5.db [--sets A,B] [--no-cycle] [--max-atoms 10] [--hac-max 36]
                  [--min-freq 1] [-c 8] [--tmp-dir /scratch/$USER]
```

The sampler makes millions of small random lookups, which are slow on a database of hundreds of GB on a network
file system. `prepare` reads the database once, sequentially and read-only, and writes a single SQLite file
(the *slice*) with only what the two operations can use at one radius:

| option | meaning |
|---|---|
| `-r` | context radius (one slice per radius) |
| `--sets` | set columns of the database to keep (default: all). A slice with several sets serves all of them |
| `--max-atoms` | largest core added in one step (heavy atoms) |
| `--hac-max` | largest seed; the sampling cap cannot exceed it |
| `--min-freq` | use only fragments seen at least this many times in the set, as CReM's `min_freq` |
| `--no-cycle` | grow rows only (smaller, faster; no `--mode cycle` later) |
| `--tmp-dir` | build on node-local disk, then move the finished file to `-o` |

A slice is itself a valid CReM v2 database (`grow_mol` and `make_cycle` run on it and give the same products as on
the full database within these limits). A slice also contains some additional tables with pre-computed values to 
speed-up further calculations.

### 2. `sample`: run the walkers

```
cremspace sample slice_r5.db --sets A,B --mode grow -o runs/r5_grow [--walkers 16] [--hours 72]
                 [--hac-max 36] [--freeze-lnf 1e-3] [--copy auto]
```

For every set, `--walkers` processes walk the space. They first adapt a shared Wang-Landau density of states until
its modification factor ln f reaches `--freeze-lnf`, then freeze it and continue in *production*, which collects
the counts the estimate is computed from. Walkers stop after `--hours` (or on SIGTERM, after a checkpoint).

- **Resume and extend.** Run the same command with the same `-o` again; every walker continues from its checkpoint.
  Settings that define the space (mode, cap, slice, ...) must stay the same, `--hours` and `--walkers` may change.
  More sets can be added to an existing RUN directory.
- **CPUs and memory.** One CPU per walker: `--walkers 16` with two sets needs 32 CPUs. Each walker uses 1-3 GB.
- **RAM copy.** The slice is copied to `/dev/shm` before sampling when it fits into the memory the job may use
  (cgroup limit and free memory, leaving `--walker-mem-gb` per walker), else to node-local `$TMPDIR` if that is a
  separate disk, else it is read in place. Jobs on the same node share one copy; the last one removes it.
  `--copy none|shm|local` overrides the choice. A job killed with SIGKILL can leave `/dev/shm/cremspace-*` behind;
  the next `cremspace sample` on that node cleans it up, or remove it by hand.
- **Refreeze.** If the frozen weights trap the walkers in part of the space (production far from flat over heavy-
  atom counts), the sampler re-freezes them at the current transition-matrix estimate (checked every
  `--refreeze-every` hours; logged in `RUN/<set>/refreeze.log`). Estimates pool all production epochs.
- **How long?** At radius 5 with 16 walkers per set, sets of 10^13-10^25 molecules needed 3-7 days to pass all
  checks (`diagnose`); start with `--hours 72` and extend as `diagnose` suggests.

The RUN directory:

```
runs/r5_grow/
  run.json                    slice, its parameters, mode, cap, and every invocation (host, hours, code versions)
  summary.tsv                 written by diagnose
  A/
    shared.json               shared Wang-Landau state
    REFREEZE.json, refreeze.log   automatic refreeze (only if it happened)
    w01/ ... w16/             walkers: state.json (checkpoint and counts), walker.log, tours.tsv,
                              samples_prod.tsv (production samples), tm_hist.jsonl (convergence trace)
    estimate.json, report.txt written by diagnose
```

### 3. `diagnose`: estimate and verdict

```
cremspace diagnose runs/r5_grow [more RUN dirs] [-c 8]
```

For every set, `diagnose` computes the estimate (transition matrix on the pooled production counts of all walkers,
uncertainty by bootstrap over walkers) and runs the convergence checks. The verdict is **DONE** when all pass:

| check | passes when |
|---|---|
| production | every walker is in production |
| precision | bootstrap sd of log10 N_total < 0.1 (`--sd`) |
| stability | the estimate from the first half of each walker's production differs from the full one by < 0.1, and the last three points of the trace span < 0.1 (`--drift`) |
| coverage | every walker made >= 20 production round trips, from the root (seeds) to the cap and back (`--min-rt`) |
| mixing | walkers sample the same chemistry: per heavy-atom group and descriptor (rings, heteroatoms, Fsp3, ...), Kruskal-Wallis tests across walkers on tour means, and scaffold sharing between walkers vs a permutation null, all at a Bonferroni-corrected level 0.05 (`--alpha`); and no walker's mean is shifted by more than 0.5 sd (`--shift-max`) |

Otherwise the verdict is `CHECK: <failed checks>; extend production by ~X h`. Rerun `cremspace sample` with the same
RUN directory for about that long and diagnose again. `RUN/<set>/report.txt` shows log10 N(h) per heavy-atom count,
the convergence trace and every test; `RUN/summary.tsv` has one line per set.

### 4. `table`: final estimates

```
cremspace table runs/r5_grow runs/r5_cycle [--per-hac] -o estimates.tsv
```

One row per RUN directory and set: mode, radius, min_freq, max_atoms, cap, log10 N (total over heavy-atom counts
1..cap), its sd and 95% interval, and the verdict (`--per-hac` adds log10 N for every heavy-atom count). Sets that
are not DONE are reported with a warning.

## Example: the tiny database (minutes on a laptop)

`examples/tiny/tiny.db` is a CReM v2 database built from 1000 ChEMBL molecules (radius 5, one set `example`;
`make_tiny_db.py` shows how). Its space is small enough to enumerate exactly, so the estimate can be checked
against the truth. With 4 walkers per mode (8 CPUs) the whole example takes about 10 minutes:

```
cd examples/tiny
cremspace prepare -d tiny.db -r 5 -o tiny_r5.db --hac-max 18
cremspace sample tiny_r5.db --mode grow  -o runs/grow  --walkers 4 --hours 0.15 --hac-max 16 &
cremspace sample tiny_r5.db --mode cycle -o runs/cycle --walkers 4 --hours 0.15 --hac-max 16
wait
cremspace diagnose runs/grow runs/cycle --exact exact_grow_h18.json exact_cycle_h18.json
cremspace table runs/grow runs/cycle
```

`exact_*_h18.json` hold the exact counts up to 18 heavy atoms (`cremspace exact tiny_r5.db --set example --mode grow
--hac-max 18 -o exact_grow_h18.json`); `diagnose --exact` adds them to the report with (estimate - exact) / sd.
A test run gave:

| mode | log10 N (1-16 heavy atoms) | exact |
|---|---|---|
| grow | 4.135 +- 0.003 | 4.136 |
| grow + make_cycle | 4.180 +- 0.005 | 4.173 |

Both sets reach DONE: the walkers adapt within a minute or two and then make thousands of round trips. In a
real space they make a few per hour, which is why real runs take days.

## Running on a cluster

`examples/pbs/` and `examples/slurm/` contain job scripts for `prepare` and `sample`. Recommendations:

- run `prepare` on a node with fast local disk (`--tmp-dir` on it) and enough memory for `--cache-mb`;
- run `sample` with all sets of one radius and mode in one job on a large-memory node: the slice is then copied to
  RAM once and shared by all walkers;
- request one CPU per walker and about 2 GB per walker plus the slice size;
- `sample` is resumable: chain jobs (`qsub -W depend=afterany:...`, `sbatch --dependency=afterany:...`) with the
  same command until `diagnose` reports DONE.

## Using your own fragment database

1. Build a CReM v2 database from your molecules (`cremdb_create`, see the CReM documentation); fragment sets
   (e.g. a filtered subset) become set columns of the database.
2. Choose the radius (CReM's generation default is 3; larger radii give smaller, more conservative spaces), the cap
   and `--min-freq` as you would use them for generation.
3. `prepare`, `sample` (`--mode grow` and/or `--mode cycle`), `diagnose` until DONE, `table`.

Cost grows with the size of the space and steeply at small radii (more cores per context, longer walks between
the root and the cap). At radius 1 the SureChEMBL + ChEMBL spaces (10^28-10^49 molecules) could only be sampled
up to 25 heavy atoms. For such spaces sample with a lower cap (`--hac-max`) and report the space up to that cap.

## How it works

The space is a directed graph: an edge P -> M for every way CReM makes M from P (one per H-site class and core for
grow, per anchor-pair class and core for make_cycle). Every edge adds heavy atoms, and every edge into M corresponds
to exactly one way of cutting M back into a parent and a database fragment. A virtual root (0 heavy atoms) has an
edge to every seed. With D+(x) the number of edges out of x and D-(x) the number into x:

1. **Walk.** Propose a forward move (uniform over the D+(x) edges) or a backward move (uniform over the D-(x)
   in-edges from parents in the space) with probability 1/2 each, and accept with the Metropolis-Hastings ratio
   for the target pi(x) ~ 1/g(h(x)), with h the heavy-atom count:
   min(1, g(h_x)/g(h_y) * D+(x)/D-(y)) forward, min(1, g(h_x)/g(h_y) * D-(x)/D+(y)) backward. Only degrees are
   needed, never the full list of products.
2. **Adapt.** Wang-Landau (1/t schedule) adapts g(h) so that all heavy-atom counts are visited equally; all walkers
   of a set share one g.
3. **Produce.** With g frozen, the walk samples each heavy-atom bin uniformly. For the current molecule in bin i the
   walker records F_ij, its edges into bin j, and R_ij, its edges from bin j.
4. **Estimate.** Counting the edges between two bins from both ends gives the exact identity
   N_i <F_ij>_i = N_j <R_ji>_j, so ln N_j - ln N_i = ln <F_ij>_i - ln <R_ji>_j. With N_0 = 1 (the root) the system
   over all bin pairs is solved by weighted least squares. The estimate does not depend on how well g was adapted;
   the bootstrap over walkers gives its uncertainty.

Exact enumeration of small spaces (`cremspace exact`) and the tests (`pytest`) check that the edges are CReM's
(products equal to `grow_mol` / `make_cycle`) and that every forward edge has exactly one reverse cut.

## Tests

```
pip install pytest
pytest                 # fast tests (about a minute)
pytest -m slow         # also a short sampling run on the tiny database (several minutes)
```

## License

MIT (code). The example database is derived from ChEMBL (CC BY-SA 3.0), see `examples/tiny/LICENSE`.
