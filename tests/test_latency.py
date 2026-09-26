"""CPU tests for exact interval accounting and scoped CUDA observer plumbing."""
import contextlib
import json
import unittest
from types import SimpleNamespace

from benchmarks.latency import LatencyTrace, intersect_intervals, interval_ms, summarize_intervals, union_intervals


class IntervalTests(unittest.TestCase):
    def test_union_handles_overlap_nesting_adjacency_and_zero_length(self):
        self.assertEqual(union_intervals([(4, 9), (1, 5), (2, 3), (9, 10), (12, 12)]), [(1.0, 10.0)])
        self.assertEqual(interval_ms([(1, 7), (4, 8)]), 7)
        self.assertEqual(intersect_intervals([(1, 7), (5, 9)], [(2, 3), (6, 11)]), [(2.0, 3.0), (6.0, 9.0)])

    def test_invalid_intervals_fail(self):
        for interval in [(2, 1), (0, float("nan")), (float("inf"), float("inf"))]:
            with self.assertRaises(ValueError):
                union_intervals([interval])

    def test_nested_phase_spans_and_waits_are_not_added(self):
        records = [
            {"kind": "draft_window", "start_ms": 0, "end_ms": 7},
            {"kind": "target_verify", "start_ms": 7, "end_ms": 20},
            {"kind": "h2d", "start_ms": 1, "end_ms": 10, "bytes": 100},
            {"kind": "h2d", "start_ms": 8, "end_ms": 15, "bytes": 200},
            {"kind": "prefetch_wait", "start_ms": 6, "end_ms": 10},
            {"kind": "demand_wait", "start_ms": 8, "end_ms": 12},
            {"kind": "demand_wait", "start_ms": 13, "end_ms": 16},
        ]
        summary = summarize_intervals(records)
        self.assertEqual(summary["h2d_union_ms"], 14)
        self.assertEqual(summary["span_sum_ms"]["h2d"], 16)
        self.assertEqual(summary["exposed_wait_union_ms"], 9)
        self.assertEqual(summary["h2d_overlapping_wait_ms"], 8)
        self.assertEqual(summary["h2d_outside_observed_wait_ms"], 6)
        self.assertEqual(summary["target_verify_nonwait_span_ms"], 5)
        self.assertEqual(summary["h2d_bytes"], 300)
        self.assertEqual(summarize_intervals([])["h2d_outside_observed_wait_ratio"], None)
        json.dumps(summary, allow_nan=False)


class FakeEvent:
    def __init__(self):
        self.time = None

    def record(self, stream):
        self.time = stream.time

    def elapsed_time(self, other):
        return other.time - self.time


class FakeStream:
    def __init__(self, name):
        self.name, self.time = name, 0.0

    def wait_event(self, event):
        self.time = max(self.time, event.time)


class FakeCuda:
    def __init__(self):
        self.compute = FakeStream("compute")
        self.transfer = FakeStream("transfer")
        self.prefetch = FakeStream("prefetch")
        self.synchronizations = 0

    def Event(self, **kwargs):
        return FakeEvent()

    def current_stream(self, device):
        return self.compute

    def synchronize(self, device):
        self.synchronizations += 1

    def stream(self, stream):
        return contextlib.nullcontext()


class FakeManager:
    specter_async_loading_enabled = True
    module_size = 64
    device = "fake"

    def __init__(self, cuda):
        self.cuda, self.stream_tx = cuda, cuda.transfer
        self.registered_experts = {
            (0, index): SimpleNamespace(uid=(0, index), offloaded=True, ready=None)
            for index in range(3)
        }

    def _async_copy_storage(self, destination, source, last_use_event, ready_event):
        if last_use_event is not None:
            self.stream_tx.wait_event(last_use_event)
        self.stream_tx.time += 5
        ready_event.record(self.stream_tx)

    def _submit_async_load(self, info, *args):
        info.ready = FakeEvent()
        self._async_copy_storage(None, None, None, info.ready)
        info.offloaded = False

    def _submit_async_staging_load(self, info, *args):
        return self._submit_async_load(info, *args)

    def _async_wait_event(self, stream, event):
        stream.wait_event(event)

    def load_experts_async(self, *uids, **kwargs):
        for uid in uids:
            info = self.registered_experts[uid]
            if info.offloaded:
                self._submit_async_load(info)
            self._async_wait_event(self.cuda.compute, info.ready)
            yield uid, "expert"

    def prefetch_experts_async(self, *uids, **kwargs):
        for uid in uids:
            info = self.registered_experts[uid]
            if info.offloaded:
                self._submit_async_load(info)

    def prefetch_experts_bounded_async(self, *uids, **kwargs):
        for uid in uids[:kwargs["max_new_loads"]]:
            info = self.registered_experts[uid]
            if info.offloaded:
                self._submit_async_load(info)


class FakeController:
    def __init__(self, cuda):
        self.cuda, self.stream = cuda, cuda.prefetch
        self.mutex = None

    def begin_window(self, length):
        pass

    def _wait_for_prefetch(self, stream, event):
        stream.wait_event(event)

    def before_verify(self):
        if self.mutex is not None:
            self._wait_for_prefetch(self.cuda.compute, self.mutex)

    def end_window(self):
        pass


class FakeModel:
    def __init__(self, cuda):
        self.cuda = cuda

    def forward(self, value):
        self.cuda.compute.time += 2
        return value + 1


class TraceScopeTests(unittest.TestCase):
    def make_trace(self):
        cuda = FakeCuda()
        manager, controller = FakeManager(cuda), FakeController(cuda)
        draft, target = FakeModel(cuda), FakeModel(cuda)
        return LatencyTrace(draft, target, controller, manager, _cuda=cuda)

    def test_separate_pass_events_requests_and_method_restoration(self):
        trace = self.make_trace()
        objects = [trace.draft, trace.target, trace.controller, trace.manager]
        before = [set(vars(instance)) for instance in objects]
        with trace:
            with self.assertRaises(RuntimeError):
                trace.result()
            trace.controller.begin_window(2)
            self.assertEqual(trace.draft.forward(3), 4)
            trace.manager.prefetch_experts_async((0, 0))
            trace.controller.mutex = trace.manager.registered_experts[(0, 0)].ready
            trace.controller.before_verify()
            self.assertEqual(list(trace.manager.load_experts_async((0, 0), (0, 1))), [
                ((0, 0), "expert"), ((0, 1), "expert")
            ])
            self.assertEqual(trace.target.forward(4), 5)
            trace.controller.end_window()
        self.assertEqual([set(vars(instance)) for instance in objects], before)
        result = trace.result()
        self.assertEqual(trace.cuda.synchronizations, 2)
        self.assertEqual(result["summary"]["h2d_union_ms"], 10)
        self.assertEqual(result["summary"]["exposed_wait_union_ms"], 8)
        self.assertEqual(result["summary"]["h2d_bytes"], 128)
        self.assertEqual(result["requests"][1]["residency_hits"], 1)
        self.assertEqual(result["requests"][1]["residency_misses"], 1)
        self.assertEqual([record["request_kind"] for record in result["records"] if record["kind"] == "h2d"], ["prefetch", "demand"])
        self.assertEqual(result["iterations"][0]["iteration"], 0)
        json.dumps(result, allow_nan=False)

    def test_copy_span_excludes_slot_reuse_dependency_wait(self):
        trace = self.make_trace()
        with trace:
            last_use, ready = FakeEvent(), FakeEvent()
            last_use.time = 20
            trace.manager._async_copy_storage(None, None, last_use, ready)
        copy = next(record for record in trace.result()["records"] if record["kind"] == "h2d")
        self.assertEqual((copy["start_ms"], copy["end_ms"]), (20, 25))
        self.assertEqual(trace.result()["summary"]["h2d_union_ms"], 5)

    def test_decode_exception_restores_every_method(self):
        trace = self.make_trace()
        original = trace.draft.forward
        trace.draft.forward = original
        with self.assertRaisesRegex(ValueError, "decode failure"):
            with trace:
                trace.controller.begin_window(2)
                trace.draft.forward(1)
                raise ValueError("decode failure")
        self.assertIs(trace.draft.forward, original)
        self.assertNotIn("forward", vars(trace.target))
        self.assertNotIn("_scoped_latency_trace", vars(trace.manager))
        self.assertTrue(trace.result()["failed"])
        self.assertTrue(next(record for record in trace.result()["records"] if record["kind"] == "draft_window")["incomplete"])

    def test_install_exception_restores_already_installed_methods(self):
        trace = self.make_trace()
        trace.controller = SimpleNamespace(stream=trace.cuda.prefetch)
        with self.assertRaises(AttributeError):
            with trace:
                pass
        self.assertNotIn("forward", vars(trace.draft))
        self.assertNotIn("forward", vars(trace.target))
        self.assertNotIn("_scoped_latency_trace", vars(trace.manager))


if __name__ == "__main__":
    unittest.main()
