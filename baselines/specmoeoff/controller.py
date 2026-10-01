"""Route-frequency Top-M prefetching for the Transformers B=1 baseline."""
import torch


class LegacyPrefetchController:
    def __init__(self, case, manager, route_modules, fractions=(0.2, 0.6)):
        self.case = case
        self.manager = manager
        self.routes = tuple(route_modules)
        if len(self.routes) != case.layer_num:
            raise ValueError("Draft route layer count does not match model case")
        self.prefetch_thr = {int(case.gamma * fraction) for fraction in fractions}
        self.counts = torch.zeros((case.layer_num, case.num_experts), dtype=torch.int32, device=manager.device)
        self.stream = torch.cuda.Stream(device=manager.device)
        self.stats = {'draft_route_updates': 0, 'prefetch_plans': 0, 'prefetch_candidates': 0}
        self.done = None

    def after_draft_forward(self):
        for index, route in enumerate(self.routes):
            selected = getattr(route, '_selected_experts', None)
            if selected is None:
                raise RuntimeError("Adapted draft module does not expose routing observations")
            selected = selected.detach().long().flatten()
            self.counts[index].index_add_(0, selected, torch.ones_like(selected, dtype=torch.int32))
        self.stats['draft_route_updates'] += 1

    def after_draft_token(self, index):
        if index not in self.prefetch_thr:
            return
        ready = torch.cuda.Event(enable_timing=False)
        ready.record(torch.cuda.current_stream(self.manager.device))
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            values, experts = torch.topk(self.counts, self.case.num_experts - self.case.offload_per_layer,
                                        dim=1, sorted=False)
            # Preserve the old controller's host-side per-layer admission.
            candidates = experts.cpu().tolist()
            positive = (values > 0).cpu().tolist()
            for layer, (indices, enabled) in enumerate(zip(candidates, positive)):
                layer_id = layer + int(self.case.skip_first_layer)
                uids = [(layer_id, expert) for expert, keep in zip(indices, enabled) if keep]
                self.stats['prefetch_candidates'] += len(uids)
                for _, _, event in self.manager.prefetch_experts(*uids, unordered=True):
                    if event is not None:
                        self.stream.wait_event(event)
            self.done = torch.cuda.Event(enable_timing=False)
            self.done.record(self.stream)
        self.stats['prefetch_plans'] += 1

    def before_verify(self):
        if self.done is not None:
            torch.cuda.current_stream(self.manager.device).wait_event(self.done)

    def finish_round(self):
        # The compute stream has joined the predictive stream before verify.
        self.counts.zero_()
        self.done = None
