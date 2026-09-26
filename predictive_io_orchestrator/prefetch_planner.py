"""PIO bandwidth-aware Cold Start / Prefetch Overlap / Final Cleanup."""
import math
import time
import torch
from predictive_io_orchestrator.pattern_analyzer import IncrementalRoutingPatternAnalyzer


class LayerAwarePrefetchPlanner(IncrementalRoutingPatternAnalyzer):
    def __init__(self, *args, initial_fraction=0.2, bandwidth_gbps=12.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_fraction = initial_fraction
        self.bandwidth_bytes_per_ms = bandwidth_gbps * 1e6
        self.draft_ms = 1.0
        self.phase = 'idle'
        self.boundaries = []
        self._timings = []
        # Verification misses enter resident slots, released only after use.
        self.expert_manager.set_specter_async_demand_policy('resident')

    def begin_window(self, draft_len):
        self.clear()
        self.draft_len = draft_len
        self.phase = 'cold_start'
        self.next_boundary = max(1, int(self.initial_fraction * draft_len))
        self._last_step = torch.cuda.Event(enable_timing=True)
        self._last_step.record()
        self._predicted = [set() for _ in range(self.layer_num)]

    def update_from_bound_routes(self):
        if self.phase != 'final_cleanup':
            super().update_from_bound_routes()

    def _poll_timings(self):
        pending = []
        for kind, start, end, amount in self._timings:
            if not end.query():
                pending.append((kind, start, end, amount))
                continue
            elapsed = max(start.elapsed_time(end), 0.001)
            if kind == 'draft':
                self.draft_ms = 0.75 * self.draft_ms + 0.25 * elapsed
            elif amount:
                self.bandwidth_bytes_per_ms = 0.75 * self.bandwidth_bytes_per_ms + 0.25 * amount / elapsed
        self._timings = pending

    def after_draft_step(self, step_idx):
        done = torch.cuda.Event(enable_timing=True)
        done.record()
        self._timings.append(('draft', self._last_step, done, 1))
        self._last_step = done
        self._poll_timings()
        completed = step_idx + 1
        if self.phase == 'final_cleanup' or completed < self.next_boundary:
            return
        if completed >= self.draft_len:
            self.phase = 'final_cleanup'
            self.clear()
            return
        manager = self.expert_manager
        before = int(manager.async_loading_stats.get('h2d_bytes', 0))
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self._lookahead_ready)
            plans = self.get_all_layer_topm_uids(self.max_cached_experts)
            start.record(manager.stream_tx)
            for layer, uids in enumerate(plans):
                # Protect all selected residents while only new misses transfer.
                for _, _, ready in manager.prefetch_experts_async(*uids, unordered=True):
                    if ready is not None:
                        self.stream.wait_event(ready)
                self._predicted[layer] = set(uids)
            end.record(manager.stream_tx)
            self.mutex = torch.cuda.Event()
            self.mutex.record(self.stream)
        transferred = int(manager.async_loading_stats.get('h2d_bytes', 0)) - before
        self._timings.append(('h2d', start, end, transferred))
        chunk = max(1, math.ceil(transferred / self.bandwidth_bytes_per_ms / max(self.draft_ms, 0.001)))
        filled = all(len(uids) >= self.max_cached_experts for uids in self._predicted)
        self.next_boundary = min(self.draft_len, completed + chunk)
        self.phase = 'final_cleanup' if filled or self.next_boundary == self.draft_len else 'prefetch_overlap'
        self.boundaries.append({'at': completed, 'next': self.next_boundary, 'phase': self.phase, 'h2d_bytes': transferred, 'draft_ms': self.draft_ms, 'estimated_bandwidth_bytes_per_ms': self.bandwidth_bytes_per_ms})
        if self.phase == 'final_cleanup':
            self.clear()

    def _wait_for_prefetch(self, stream, event):
        stream.wait_event(event)

    def before_verify(self):
        if self.mutex is not None:
            self._wait_for_prefetch(torch.cuda.current_stream(self.device), self.mutex)
        self.clear()

    def end_window(self):
        self.phase = 'idle'
        self._poll_timings()

    def abort_window(self):
        self.before_verify()
        self.phase = 'idle'

    def get_prefetch_policy_stats(self):
        return {'policy': 'paper_adaptive', 'boundaries': self.boundaries, 'draft_ms': self.draft_ms, 'estimated_bandwidth_bytes_per_ms': self.bandwidth_bytes_per_ms}
