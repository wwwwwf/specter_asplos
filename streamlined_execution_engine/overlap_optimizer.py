"""SEE stream/event ordering and pinned asynchronous expert transfers.

Used as the residency manager's execution mixin. Readiness and last-use
events are shared with the planner, so slot reuse cannot race expert GEMMs.
"""
import torch

class OverlapOrientedParallelismOptimizer:
    def _new_async_event(self):
        return torch.cuda.Event(enable_timing=False, blocking=False)

    def _async_current_stream(self):
        return torch.cuda.current_stream(device=self.device)

    def _async_record_event(self, event, stream):
        event.record(stream)

    def _async_wait_event(self, stream, event):
        stream.wait_event(event)

    def _async_copy_storage(
        self,
        destination,
        source,
        last_use_event,
        ready_event,
    ):
        if hasattr(source, "is_pinned") and not source.is_pinned():
            raise RuntimeError(
                "Specter async loading requires pinned host expert storage"
            )
        with torch.cuda.stream(self.stream_tx):
            if last_use_event is not None:
                self.stream_tx.wait_event(last_use_event)
            destination.copy_(source, non_blocking=True)
            ready_event.record(self.stream_tx)

