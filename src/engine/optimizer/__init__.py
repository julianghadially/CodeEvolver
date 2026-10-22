"""CodeEvolver optimizer package.

V3 re-exports the EE (exploit and explore) optimizer. Sibling
submodules can host additional optimizers in the future; this __init__ keeps
the surface area minimal.
"""

from .v3 import IterationContext
from src.engine.schemas.iteration_history import IterationHistory

__all__ = [
    "IterationContext",
    "IterationHistory",
]
