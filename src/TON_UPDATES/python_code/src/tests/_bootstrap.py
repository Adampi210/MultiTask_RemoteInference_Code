"""Bootstrap sys.path for tests so ``import MGF1`` (etc.) resolves to src/main/."""
import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN_DIR = os.path.normpath(os.path.join(_HERE, "..", "main"))

if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

import paths  # noqa: E402,F401
paths.ensure_dirs()
