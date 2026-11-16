"""cremspace command line: prepare -> sample -> diagnose -> table (status, exact as helpers)."""
import argparse
import sys

from cremspace import __version__


def main(argv=None):
    from cremspace import diagnose, exact, prepare, sampler, status, table
    p = argparse.ArgumentParser(
        prog="cremspace",
        description="Estimate the number of molecules a CReM fragment database can generate with grow_mol "
                    "(and make_cycle). Workflow: prepare -> sample -> diagnose -> table.")
    p.add_argument("--version", action="version", version=f"cremspace {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True
    prepare.add_parser(sub)
    sampler.add_parser(sub)
    diagnose.add_parser(sub)
    table.add_parser(sub)
    status.add_parser(sub)
    exact.add_parser(sub)
    a = p.parse_args(argv)
    rc = a.func(a)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
