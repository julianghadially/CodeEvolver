"""GEPA optimization integration for CodeEvolver.

Note: The gepa package is only installed on the GEPA worker image (gepa_image),
not on the FastAPI web image. Imports that depend on gepa are guarded so that
modules like gepa_state.py can be imported by the web server for progress
tracking without triggering ImportError.
"""

from .gepa_state import GEPAStateRecord

try:
    from .adapter import CodeEvolverDSPyAdapter
    from .callback import CallbackJobUpdater, CallbackProgressTracker
    from .component_selector import CodeFrequencyComponentSelector
    from .gepa_sandbox import GEPASandbox
    from .optimizer import run_gepa_optimization
except ImportError:
    # Running on web image without gepa — only GEPAStateRecord is available
    pass

__all__ = [
    "CodeEvolverDSPyAdapter",
    "CallbackJobUpdater",
    "CallbackProgressTracker",
    "CodeFrequencyComponentSelector",
    "GEPASandbox",
    "GEPAStateRecord",
    "run_gepa_optimization",
]
