"""Pure-PyTorch 4-direction XAI / MI core (requires torch; zero ``comfy.*`` imports).

Modules: ``session`` (engine + hook policy), ``heatmap`` (Dir. 01), ``layer_selection``
(Dir. 02), ``patching`` (Dir. 03), ``verification`` (Dir. 04), ``profiling``.
"""

from .profiling import Profiler, measure
from .session import (
    AttentionCall,
    LogitEdit,
    ProbObserver,
    ProbZero,
    Region,
    RunningMap,
    TextAxisBlur,
    XAISession,
    chunked_attention,
)

__all__ = [
    "AttentionCall", "LogitEdit", "ProbObserver", "ProbZero", "Profiler", "Region", "RunningMap",
    "TextAxisBlur", "XAISession", "chunked_attention", "measure",
]
