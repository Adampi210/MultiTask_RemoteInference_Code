"""
_bootstrap.py - add src/main/ to sys.path so execute scripts can
``import valuefunction1`` (etc.) without the caller setting PYTHONPATH.

Each script in src/execute/ does:

    from _bootstrap import paths

before importing anything from src/main/. The side-effect of the import is
that src/main is prepended to sys.path; `paths` is then re-exported so callers
have a single import for the path constants.
"""
import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.normpath(os.path.join(_HERE, "..", "main"))

if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)


# Re-export the paths module so execute scripts only need this one import.
import paths  # noqa: E402,F401
paths.ensure_dirs()
