"""Load the plugin (the repository root is the package) as ``hermes_supermemory``.

Hermes Agent must be importable (``agent``, ``hermes_cli``, ``utils``), e.g. installed in the same venv.
"""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

if "hermes_supermemory" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "hermes_supermemory", _ROOT / "__init__.py", submodule_search_locations=[str(_ROOT)])
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["hermes_supermemory"] = _module
    _spec.loader.exec_module(_module)
