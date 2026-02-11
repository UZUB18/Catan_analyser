import threading
import unittest

from catan_analyzer.analysis.runtime import AnalysisCancelled, AnalysisRuntime


class AnalysisRuntimeTests(unittest.TestCase):
    def test_subrange_maps_progress_and_prefix(self) -> None:
        events: list[tuple[str, float]] = []
        runtime = AnalysisRuntime(
            on_progress=lambda stage, fraction: events.append((stage, round(fraction, 3))),
            min_progress_interval_s=0.0,
            min_progress_delta=0.0,
        )
        child = runtime.subrange(0.20, 0.60, stage_prefix="Phase: ")
        child.report_progress("working", 0.50, force=True)

        self.assertEqual(events[-1], ("Phase: working", 0.4))

    def test_cancelled_runtime_raises(self) -> None:
        cancel_event = threading.Event()
        runtime = AnalysisRuntime(cancel_event=cancel_event)
        runtime.cancel()

        with self.assertRaises(AnalysisCancelled):
            runtime.raise_if_cancelled()


if __name__ == "__main__":
    unittest.main()
