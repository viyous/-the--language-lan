"""Lan compiler command-line entry point.

The implementation lives in the `lan_compiler` package; this module keeps the
historical `python lanc.py ...` command and `from lanc import ...` imports working.
"""
import sys

from lan_compiler import *  # noqa: F401,F403  (re-export the former lanc namespace)
from lan_compiler import __all__, main  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
