"""ComfyUI custom-node entry point for XAI_DiT (thin wrapper over ``xai_dit.core``).

Install by linking this folder into ``ComfyUI/custom_nodes/`` (or ``pip install -e`` the repo).
When loaded by ComfyUI as a top-level module, the repo ``src/`` directory is put on
``sys.path`` so the absolute ``xai_dit`` imports resolve.
"""

import sys
from pathlib import Path

try:
    import xai_dit  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xai_dit.comfyui.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
