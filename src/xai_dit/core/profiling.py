"""Stage-wise time / memory profiler (evolved from ``Anima_HEATMAP/anima_heatmap/profiling.py``).

Reports per stage: ``seconds``, ``calls``, ``sec_per_call``, peak GPU ``allocated`` /
``reserved`` bytes, peak CPU RSS bytes; plus run-level ``wall_clock_sec`` and
``sec_per_forward`` (``experiments/design/output_spec.md`` ``resources`` block).

Nested stages are supported: CUDA peak counters are folded into every open stage before
each reset, so an inner stage never hides an outer stage's peak.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from typing import Any, ContextManager, Iterator, Optional

import torch

_active: ContextVar[Optional["Profiler"]] = ContextVar("xai_dit_profiler", default=None)

FORWARD_STAGE = "forward"


def cpu_rss_bytes() -> Optional[int]:
    """Current process resident set size, or ``None`` if unavailable on this platform."""
    try:
        import psutil  # optional

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            return None
    return None


@dataclass
class StageStats:
    seconds: float = 0.0
    calls: int = 0
    gpu_allocated_peak: int = 0
    gpu_reserved_peak: int = 0
    cpu_rss_peak: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "calls": self.calls,
            "sec_per_call": self.seconds / self.calls if self.calls else None,
            "gpu_allocated_gb": self.gpu_allocated_peak / 2**30,
            "gpu_reserved_gb": self.gpu_reserved_peak / 2**30,
            "cpu_rss_gb": self.cpu_rss_peak / 2**30,
        }


class Profiler:
    """Opt-in profiler. Use ``with prof.activate(): ... with measure("dit"): ...``."""

    def __init__(self, device: Optional[torch.device | str] = None) -> None:
        dev = torch.device(device) if device is not None else None
        self.device = dev if dev is not None and dev.type == "cuda" and torch.cuda.is_available() else None
        self.stages: dict[str, StageStats] = {}
        self.counts: dict[str, int] = {}
        self._open: list[StageStats] = []
        self._started: Optional[float] = None
        self.wall_clock_sec = 0.0

    def _sync(self) -> None:
        if self.device is not None:
            torch.cuda.synchronize(self.device)

    def _fold_peaks(self, reset: bool) -> None:
        rss = cpu_rss_bytes() or 0
        alloc = reserved = 0
        if self.device is not None:
            alloc = torch.cuda.max_memory_allocated(self.device)
            reserved = torch.cuda.max_memory_reserved(self.device)
        for st in self._open:
            st.gpu_allocated_peak = max(st.gpu_allocated_peak, alloc)
            st.gpu_reserved_peak = max(st.gpu_reserved_peak, reserved)
            st.cpu_rss_peak = max(st.cpu_rss_peak, rss)
        if reset and self.device is not None:
            torch.cuda.reset_peak_memory_stats(self.device)

    @contextmanager
    def activate(self) -> Iterator["Profiler"]:
        token = _active.set(self)
        self._started = perf_counter()
        try:
            yield self
        finally:
            self.wall_clock_sec += perf_counter() - self._started
            _active.reset(token)

    @contextmanager
    def stage(self, name: str) -> Iterator[StageStats]:
        st = self.stages.setdefault(name, StageStats())
        self._sync()
        self._fold_peaks(reset=True)
        self._open.append(st)
        started = perf_counter()
        try:
            with torch.profiler.record_function(f"xai_dit::{name}"):
                yield st
        finally:
            self._sync()
            st.seconds += perf_counter() - started
            st.calls += 1
            self._fold_peaks(reset=True)
            self._open.remove(st)

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + n

    def summary(self) -> dict[str, Any]:
        fwd = self.stages.get(FORWARD_STAGE)
        return {
            "stages": {k: v.as_dict() for k, v in self.stages.items()},
            "counts": dict(self.counts),
            "sec_per_forward": fwd.seconds / fwd.calls if fwd and fwd.calls else None,
            "wall_clock_sec": self.wall_clock_sec,
            "device": str(self.device) if self.device is not None else "cpu",
        }


def active_profiler() -> Optional[Profiler]:
    return _active.get()


def measure(name: str) -> ContextManager[Any]:
    """Stage context on the active profiler, or a no-op when profiling is off."""
    prof = _active.get()
    return prof.stage(name) if prof is not None else nullcontext()


def record_count(name: str, n: int = 1) -> None:
    prof = _active.get()
    if prof is not None:
        prof.count(name, n)
