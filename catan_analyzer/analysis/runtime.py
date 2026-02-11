from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


class AnalysisCancelled(RuntimeError):
    """Raised when an analysis run is cancelled by the user."""


ProgressCallback = Callable[[str, float], None]


def _clamp_fraction(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class _SubrangeConfig:
    start: float
    end: float
    stage_prefix: str


class AnalysisRuntime:
    """Cancellation + progress helper passed through analysis layers."""

    def __init__(
        self,
        *,
        cancel_event: threading.Event | None = None,
        on_progress: ProgressCallback | None = None,
        min_progress_interval_s: float = 0.08,
        min_progress_delta: float = 0.005,
        _subrange: _SubrangeConfig | None = None,
        _parent: "AnalysisRuntime | None" = None,
    ) -> None:
        self._cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self._on_progress = on_progress
        self._min_progress_interval_s = max(0.0, float(min_progress_interval_s))
        self._min_progress_delta = max(0.0, float(min_progress_delta))
        self._subrange = _subrange
        self._parent = _parent
        self._last_progress_at = 0.0
        self._last_fraction = -1.0
        self._last_stage = ""

    def subrange(self, start: float, end: float, *, stage_prefix: str = "") -> "AnalysisRuntime":
        start_f = _clamp_fraction(start)
        end_f = _clamp_fraction(end)
        if end_f < start_f:
            start_f, end_f = end_f, start_f
        return AnalysisRuntime(
            cancel_event=self._cancel_event,
            on_progress=self._on_progress,
            min_progress_interval_s=self._min_progress_interval_s,
            min_progress_delta=self._min_progress_delta,
            _subrange=_SubrangeConfig(start=start_f, end=end_f, stage_prefix=stage_prefix),
            _parent=self,
        )

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise AnalysisCancelled("Analysis cancelled.")

    def report_progress(self, stage: str, fraction: float, *, force: bool = False) -> None:
        self.raise_if_cancelled()

        if self._subrange is not None:
            mapped = self._subrange.start + ((self._subrange.end - self._subrange.start) * _clamp_fraction(fraction))
            stage_text = f"{self._subrange.stage_prefix}{stage}" if self._subrange.stage_prefix else stage
            assert self._parent is not None
            self._parent.report_progress(stage_text, mapped, force=force)
            return

        if self._on_progress is None:
            return

        now = time.perf_counter()
        clamped = _clamp_fraction(fraction)
        stage_changed = stage != self._last_stage
        progressed_enough = abs(clamped - self._last_fraction) >= self._min_progress_delta
        waited_enough = (now - self._last_progress_at) >= self._min_progress_interval_s
        if not force and not stage_changed and not progressed_enough and not waited_enough:
            return

        self._on_progress(stage, clamped)
        self._last_stage = stage
        self._last_fraction = clamped
        self._last_progress_at = now
