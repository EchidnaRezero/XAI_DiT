"""Shared CPU-only test fixtures for the M1 invariant matrix (``plan/m1.md`` §3).

* No model checkpoints, no GPU: every tensor is a tiny deterministic dummy
  (``B=2, Hq=8, Hkv=2 (GQA g=4), N=16 image tokens, L=8 text slots, d=16``).
* This module never imports torch at import time, so ``tests/test_manifests.py`` (pure
  Python) and ``tests/test_adapters_base.py`` (torch-free contract) still run on a box
  without PyTorch. Torch-backed fixtures call :func:`require_torch`, which skips the test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))  # src-layout package without an editable install

HAS_TORCH = importlib.util.find_spec("torch") is not None
TORCH_SKIP_REASON = "PyTorch is not installed (CPU-only box without torch); run on a torch host"

SIGMAS = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0)  # S = 8 Euler updates
DIMS = SimpleNamespace(B=2, Hq=8, Hkv=2, g=4, d=16, N=16, L=8, grid=(4, 4), sigmas=SIGMAS)


def require_torch():
    """Import torch or skip the calling test (skip helper for non-manifest tests)."""
    return pytest.importorskip("torch", reason=TORCH_SKIP_REASON)


@pytest.fixture(autouse=True)
def _deterministic_seed():
    if HAS_TORCH:
        import torch

        torch.manual_seed(0)
    yield


@pytest.fixture
def dims() -> SimpleNamespace:
    return DIMS


@pytest.fixture
def make_ctx():
    """``StepContext`` factory: ``make_ctx(order=(COND, UNCOND), chunk=1, sigma=1.0)``."""
    from xai_dit.adapters.base import StepContext

    def make(order=(0, 1), chunk: int = 1, sigma: float = 1.0, sigmas=SIGMAS) -> StepContext:
        order = tuple(order)
        return StepContext(tuple(float(s) for s in sigmas), float(sigma), order, chunk * len(order))

    return make


@pytest.fixture
def capabilities():
    """Dummy capabilities mirroring the three target architectures at toy size."""
    from xai_dit.adapters.base import AdapterCapability

    common = dict(num_blocks=4, num_q_heads=DIMS.Hq, num_kv_heads=DIMS.Hkv, head_dim=DIMS.d)
    return {
        "anima": AdapterCapability(name="anima_dummy", attention_kind="cross_attn", seq_order="cross",
                                   text_updated_per_block=False, supports_concept_attention=False,
                                   has_output_gate=False, text_pad_to=DIMS.L, **common),
        "moss": AdapterCapability(name="moss_dummy", attention_kind="joint_attn", seq_order="img_txt",
                                  text_updated_per_block=True, supports_concept_attention=True,
                                  has_output_gate=False, **common),
        "krea2": AdapterCapability(name="krea2_dummy", attention_kind="joint_attn", seq_order="txt_img",
                                   text_updated_per_block=True, supports_concept_attention=True,
                                   has_output_gate=True, **common),
    }


@pytest.fixture
def layouts():
    from xai_dit.adapters.base import SeqLayout

    return {order: SeqLayout(order, DIMS.N, DIMS.L) for order in ("cross", "img_txt", "txt_img")}


@pytest.fixture
def make_qkv():
    """``make_qkv(layout, batch=2, hq=8, hkv=2, d=16, dtype=float32, seed=0) -> (q, k, v)`` (pre GQA repeat)."""
    torch = require_torch()

    def make(layout, batch: int = DIMS.B, hq: int = DIMS.Hq, hkv: int = DIMS.Hkv, d: int = DIMS.d,
             dtype=None, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(batch, hq, layout.n_query, d, generator=g)
        k = torch.randn(batch, hkv, layout.n_key, d, generator=g)
        v = torch.randn(batch, hkv, layout.n_key, d, generator=g)
        if dtype is not None:
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        return q, k, v

    return make


@pytest.fixture
def dense_attention():
    """Float64 dense reference with explicit ``repeat_interleave`` GQA: ``(probs, out)``."""
    torch = require_torch()

    def ref(q, k, v=None, *, scale=None, logit_fn=None):
        g = q.shape[1] // k.shape[1]
        s = q.double() @ k.double().repeat_interleave(g, dim=1).transpose(-1, -2)
        s = s * (q.shape[-1] ** -0.5 if scale is None else scale)
        if logit_fn is not None:
            s = logit_fn(s)
        p = torch.softmax(s, dim=-1)
        out = None if v is None else p @ v.double().repeat_interleave(g, dim=1)
        return p, out

    return ref


@pytest.fixture
def native_sdpa():
    """Class of counting "native" kernels (fused SDPA after GQA repeat), as an adapter would pass."""
    torch = require_torch()
    import torch.nn.functional as F

    class Native:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, q, k, v):
            self.calls += 1
            g = q.shape[1] // k.shape[1]
            return F.scaled_dot_product_attention(q, k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1))

    return Native


@pytest.fixture
def prob_collector():
    """``ProbObserver`` that materialises every streamed ``(q, k)`` cell into a float64 buffer."""
    torch = require_torch()
    from xai_dit.core.session import ProbObserver

    class ProbCollector(ProbObserver):
        def __init__(self, shape) -> None:
            self.probs = torch.full(tuple(shape), float("nan"), dtype=torch.float64)
            self.finished = 0

        def __call__(self, probs, q_sl, k_sl) -> None:
            self.probs[:, :, q_sl, k_sl] = probs.detach().double()

        def finish(self) -> None:
            self.finished += 1

        @classmethod
        def factory(cls, store: list):
            def make(call):
                c = cls((call.q.shape[0], call.q.shape[1], call.q.shape[2], call.k.shape[2]))
                store.append(c)
                return c

            return make

    return ProbCollector
