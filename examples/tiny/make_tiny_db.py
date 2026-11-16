#!/usr/bin/env python3
"""How examples/tiny/tiny.db was made (kept for provenance; you do not need to run it).

molecules.smi: 1000 ChEMBL 36 molecules (12-30 heavy atoms) drawn at random (seed 1) from the subset without
structural alerts and with SA score <= 2 used in the publication. tiny.db is a standard CReM v2 fragment database
built from them with crem's own API (radius 5, cores up to 10 heavy atoms, one fragment set named "example").

usage: python make_tiny_db.py [molecules.smi] [tiny.db]
"""
import sys

from crem.db import create_db

smi = sys.argv[1] if len(sys.argv) > 1 else "molecules.smi"
out = sys.argv[2] if len(sys.argv) > 2 else "tiny.db"
create_db(smi, out, "example", radii=(5,), ncpu=4, max_heavy_atoms=10, verbose=False)
