"""Opt-in CUDA event tracing for a separate, instrumented diagnostic decode.

Intervals share one CUDA event origin and are resolved only after decode.
Forward spans include launch gaps and stream waits; they are not kernel-only
compute measurements. Transfer time is never inferred from wall-time residuals.
"""
from __future__ import annotations

import math
import time
from contextlib import AbstractContextManager, contextmanager

from benchmarks.variants import ScopedPatches


def union_intervals(intervals):
    """Return disjoint intervals without double-counting nested or overlap time."""
    ordered = []
    for begin, end in intervals:
        begin, end = float(begin), float(end)
        if not math.isfinite(begin) or not math.isfinite(end) or end < begin:
            raise ValueError("Intervals must have finite, ordered endpoints")
        if end > begin:
            ordered.append((begin, end))
    result = []
    for begin, end in sorted(ordered):
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((begin, end))
    return result


def interval_ms(intervals):
    return sum(end - begin for begin, end in union_intervals(intervals))


def intersect_intervals(left, right):
    left, right = union_intervals(left), union_intervals(right)
    intersections = []
    i = j = 0
    while i < len(left) and j < len(right):
        begin = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > begin:
            intersections.append((begin, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return intersections


def summarize_intervals(records):
    """Summarize event spans using interval unions and direct intersections."""
    kinds = {}
    for record in records:
        interval = (record["start_ms"], record["end_ms"])
        union_intervals([interval])
        kinds.setdefault(record["kind"], []).append(interval)
    waits = union_intervals(kinds.get("demand_wait", []) + kinds.get("prefetch_wait", []))
    loads = union_intervals(kinds.get("h2d", []))
    draft = union_intervals(kinds.get("draft_window", []))
    verify = union_intervals(kinds.get("target_verify", []))
    load_ms = interval_ms(loads)
    overlapping_wait = interval_ms(intersect_intervals(loads, waits))
    nonwait = max(0.0, load_ms - overlapping_wait)
    return {
        "span_union_ms": {kind: interval_ms(intervals) for kind, intervals in sorted(kinds.items())},
        "span_sum_ms": {kind: sum(end - begin for begin, end in intervals) for kind, intervals in sorted(kinds.items())},
        "span_count": {kind: len(intervals) for kind, intervals in sorted(kinds.items())},
        "h2d_union_ms": load_ms,
        "h2d_bytes": sum(int(record.get("bytes", 0)) for record in records if record["kind"] == "h2d"),
        "exposed_wait_union_ms": interval_ms(waits),
        "h2d_overlapping_wait_ms": overlapping_wait,
        "h2d_outside_observed_wait_ms": nonwait,
        "h2d_outside_observed_wait_ratio": nonwait / load_ms if load_ms else None,
        "h2d_overlapping_draft_span_ms": interval_ms(intersect_intervals(loads, draft)),
        "h2d_overlapping_verify_span_ms": interval_ms(intersect_intervals(loads, verify)),
        "target_verify_nonwait_span_ms": max(0.0, interval_ms(verify) - interval_ms(intersect_intervals(verify, waits))),
    }


def _json_uid(uid):
    if isinstance(uid, (tuple, list)):
        return [_json_uid(value) for value in uid]
    if uid is None or isinstance(uid, (str, int, float, bool)):
        return uid
    raise TypeError("Expert UIDs must contain JSON scalar values")


class LatencyTrace(AbstractContextManager):
    """Temporarily instrument supplied instances; never patch torch or classes.

    Use this context in a separate pass after clean TPOT measurement. Results
    expose raw events, residency request counts, and per-iteration unions.
    ``h2d_outside_observed_wait_ratio`` is a temporal observation, not a claim
    that every non-wait transfer overlapped useful compute.
    """

    def __init__(self, draft, target, controller, manager, *, _cuda=None):
        if _cuda is None:
            import torch
            _cuda = torch.cuda
        self.cuda = _cuda
        self.draft, self.target = draft, target
        self.controller, self.manager = controller, manager
        self.device = manager.device
        self._patches = None
        self._closed = False
        self._pending = []
        self.requests = []
        self._iteration = -1
        self._phase = "prefill"
        self._request_kind = "unknown"
        self._copy_uid = None
        self._draft_start = None
        self._failed = False

    def _event(self, stream):
        event = self.cuda.Event(enable_timing=True, blocking=False)
        event.record(stream)
        return event

    def _append(self, kind, start, end, **metadata):
        self._pending.append({
            "kind": kind, "iteration": self._iteration,
            "phase": self._phase, "start": start, "end": end, **metadata,
        })

    @contextmanager
    def span(self, kind, stream=None, **metadata):
        stream = stream if stream is not None else self.cuda.current_stream(self.device)
        iteration, phase = self._iteration, self._phase
        start = self._event(stream)
        try:
            yield
        finally:
            self._append(kind, start, self._event(stream), iteration=iteration, phase=phase, **metadata)

    @contextmanager
    def _request_context(self, kind):
        old = self._request_kind
        self._request_kind = kind
        try:
            yield
        finally:
            self._request_kind = old

    def _record_request(self, kind, uids):
        misses = sum(bool(self.manager.registered_experts[uid].offloaded) for uid in uids)
        self.requests.append({
            "kind": kind, "iteration": self._iteration, "phase": self._phase,
            "uids": [_json_uid(uid) for uid in uids],
            "requests": len(uids), "residency_hits": len(uids) - misses,
            "residency_misses": misses,
        })

    def _finish_draft(self, *, incomplete=False):
        if self._draft_start is not None:
            self._append("draft_window", self._draft_start, self._event(self._compute), incomplete=incomplete)
            self._draft_start = None

    def _install(self):
        patches, manager, controller = self._patches, self.manager, self.controller

        def model_forward(model, draft):
            original = model.forward

            def forward(*args, **kwargs):
                kind = "draft_forward" if draft else (
                    "target_verify" if self._phase == "verify" else
                    "target_prefill" if self._iteration < 0 else "target_tail"
                )
                with self.span(kind):
                    return original(*args, **kwargs)

            patches.set(model, "forward", forward)

        model_forward(self.draft, True)
        model_forward(self.target, False)
        original_begin = controller.begin_window
        original_before = controller.before_verify
        original_end = controller.end_window
        original_prefetch_wait = controller._wait_for_prefetch

        def begin_window(*args, **kwargs):
            if self._draft_start is not None:
                raise RuntimeError("A previous traced draft window is still active")
            self._iteration += 1
            self._phase = "draft"
            self._draft_start = self._event(self._compute)
            return original_begin(*args, **kwargs)

        def before_verify(*args, **kwargs):
            self._finish_draft()
            self._phase = "prefetch_barrier"
            try:
                return original_before(*args, **kwargs)
            finally:
                self._phase = "verify"

        def end_window(*args, **kwargs):
            try:
                return original_end(*args, **kwargs)
            finally:
                self._phase = "between_windows"

        def prefetch_wait(stream, event):
            with self.span("prefetch_wait", stream):
                return original_prefetch_wait(stream, event)

        patches.set(controller, "begin_window", begin_window)
        patches.set(controller, "before_verify", before_verify)
        patches.set(controller, "end_window", end_window)
        patches.set(controller, "_wait_for_prefetch", prefetch_wait)

        for method in ("_submit_async_load", "_submit_async_staging_load"):
            original_submit = getattr(manager, method)

            def submit(info, *args, _original=original_submit, **kwargs):
                previous = self._copy_uid
                self._copy_uid = info.uid
                try:
                    return _original(info, *args, **kwargs)
                finally:
                    self._copy_uid = previous

            patches.set(manager, method, submit)

        original_copy = manager._async_copy_storage

        def copy_storage(destination, source, last_use_event, ready_event):
            # Move the same dependency ahead of the timing event so slot-reuse
            # waits cannot be misreported as actual H2D transfer time.
            with self.cuda.stream(manager.stream_tx):
                if last_use_event is not None:
                    manager.stream_tx.wait_event(last_use_event)
                with self.span("h2d", manager.stream_tx, request_kind=self._request_kind,
                               uid=_json_uid(self._copy_uid), bytes=int(manager.module_size)):
                    return original_copy(destination, source, None, ready_event)

        patches.set(manager, "_async_copy_storage", copy_storage)
        original_wait = manager._async_wait_event

        def wait_event(stream, event):
            if stream == self._compute:
                with self.span("demand_wait", stream, request_kind=self._request_kind):
                    return original_wait(stream, event)
            return original_wait(stream, event)

        patches.set(manager, "_async_wait_event", wait_event)
        original_demand = manager.load_experts_async

        def load_experts(*uids, **kwargs):
            self._record_request("demand", uids)
            with self._request_context("demand"):
                iterator = original_demand(*uids, **kwargs)

            def iterate():
                try:
                    while True:
                        with self._request_context("demand"):
                            try:
                                item = next(iterator)
                            except StopIteration:
                                return
                        yield item
                finally:
                    with self._request_context("demand"):
                        close = getattr(iterator, "close", None)
                        if close is not None:
                            close()

            return iterate()

        patches.set(manager, "load_experts_async", load_experts)
        # Both public prefetch paths directly enter the shared private planner;
        # neither calls the other, so requests are counted once.
        for method in ("prefetch_experts_async", "prefetch_experts_bounded_async"):
            original_prefetch = getattr(manager, method)

            def prefetch(*uids, _original=original_prefetch, **kwargs):
                self._record_request("prefetch", uids)
                with self._request_context("prefetch"):
                    return _original(*uids, **kwargs)

            patches.set(manager, method, prefetch)

    def __enter__(self):
        if self._patches is not None or self._closed:
            raise RuntimeError("Each trace context can be used only once")
        if not getattr(self.manager, "specter_async_loading_enabled", False):
            raise ValueError("Latency tracing requires the asynchronous expert manager")
        if getattr(self.manager, "_scoped_latency_trace", False):
            raise RuntimeError("A latency trace is already installed on this manager")
        if self.draft is self.target:
            raise ValueError("Draft and target tracing requires distinct model instances")
        self.cuda.synchronize(self.device)
        self._compute = self.cuda.current_stream(self.device)
        self._origin = self._event(self._compute)
        self.manager.stream_tx.wait_event(self._origin)
        self.controller.stream.wait_event(self._origin)
        self._patches = ScopedPatches()
        try:
            self._patches.set(self.manager, "_scoped_latency_trace", True)
            self._install()
        except BaseException:
            self._patches.__exit__(None, None, None)
            self._patches = None
            raise
        self._wall_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._failed = exc_type is not None
        try:
            self._finish_draft(incomplete=True)
            self.cuda.synchronize(self.device)
            self._wall_ms = (time.perf_counter() - self._wall_start) * 1000.0
        finally:
            self._patches.__exit__(exc_type, exc, traceback)
            self._patches = None
            self._closed = True
        return False

    def result(self):
        if not self._closed:
            raise RuntimeError("Resolve CUDA trace events only after the diagnostic pass")
        records = []
        for pending in self._pending:
            start, end = pending["start"], pending["end"]
            record = {key: value for key, value in pending.items() if key not in {"start", "end"}}
            record.update(start_ms=float(self._origin.elapsed_time(start)),
                          end_ms=float(self._origin.elapsed_time(end)))
            records.append(record)
        records.sort(key=lambda record: (record["start_ms"], record["end_ms"], record["kind"]))
        iteration_ids = sorted({record["iteration"] for record in records if record["iteration"] >= 0})
        return {
            "schema_version": 1,
            "instrumented_diagnostic_pass": True,
            "failed": self._failed,
            "clock": "CUDA event milliseconds relative to one common origin",
            "instrumented_wall_ms": self._wall_ms,
            "notes": [
                "Report clean TPOT from a separate uninstrumented decode pass.",
                "Model-forward event spans include launch gaps and waits, not only active kernels.",
                "H2D outside observed waits is not necessarily useful compute overlap.",
                "Overlapping phase and transfer spans must not be added as sequential components.",
                "Residency hits may still have in-flight transfers; they are not readiness hits.",
                "The final non-speculative target tail belongs to the last iteration ID but is separately labeled.",
            ],
            "records": records,
            "requests": list(self.requests),
            "summary": summarize_intervals(records),
            "iterations": [{"iteration": index, **summarize_intervals([
                record for record in records if record["iteration"] == index
            ])} for index in iteration_ids],
        }
