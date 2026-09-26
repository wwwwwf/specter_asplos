"""Scoped prefetch timing ablations using the existing expert cache."""
from benchmarks.routing import _Patches


POLICIES = ('adaptive', 'none', 'early', 'late')
DEFINITIONS = {
    'adaptive': 'Unmodified runtime adaptive planner',
    'none': 'No predictive route updates or prefetch; unchanged resident demand cache',
    'early': 'One route-frequency plan after the first draft step',
    'late': 'One cumulative route-frequency plan immediately before verification',
}


class PrefetchPolicy:
    """Change prediction timing without changing target routing or cache size.

    These are runtime ablations, not implementations of external baselines.
    Use a fresh controller for each benchmark case.
    """
    def __init__(self, controller, policy='adaptive'):
        if policy not in POLICIES:
            raise ValueError(f'Unsupported prefetch policy: {policy}')
        self.controller, self.policy = controller, policy
        self._patches = _Patches()
        self._active = self._window_open = False
        self._submitted = False
        self._steps = 0
        self._aborting = False

    def _submit(self):
        import torch
        controller = self.controller
        manager = controller.expert_manager
        before = int(manager.async_loading_stats.get('h2d_bytes', 0))
        with torch.cuda.stream(controller.stream):
            controller.stream.wait_event(controller._lookahead_ready)
            plans = controller.get_all_layer_topm_uids(controller.max_cached_experts)
            for layer, uids in enumerate(plans):
                for _, _, ready in manager.prefetch_experts_async(*uids, unordered=True):
                    if ready is not None:
                        controller.stream.wait_event(ready)
                controller._predicted[layer] = set(uids)
            controller.mutex = torch.cuda.Event()
            controller.mutex.record(controller.stream)
        transferred = int(manager.async_loading_stats.get('h2d_bytes', 0)) - before
        controller.phase = 'final_cleanup'
        controller.boundaries.append({'at': self._steps, 'next': controller.draft_len,
                                     'phase': controller.phase, 'h2d_bytes': transferred,
                                     'policy': self.policy})
        controller.clear()
        self._submitted = True

    def __enter__(self):
        controller = self.controller
        if self._active or getattr(controller, '_benchmark_prefetch_policy', False):
            raise RuntimeError('A prefetch policy already owns this controller')
        self._active = True
        try:
            self._patches.set(controller, '_benchmark_prefetch_policy', True)
            if self.policy == 'adaptive':
                return self
            begin = controller.begin_window
            before_verify = controller.before_verify
            end = controller.end_window
            abort = controller.abort_window
            stats = controller.get_prefetch_policy_stats
            self._abort = abort

            def start(draft_len):
                begin(draft_len)
                self._window_open = True
                self._submitted = False
                self._steps = 0

            def after(step_idx):
                self._steps = step_idx + 1
                if self.policy == 'early' and not self._submitted:
                    self._submit()

            def verify():
                if self.policy == 'late' and not self._aborting and self._steps and not self._submitted:
                    self._submit()
                # Preserve the runtime's event wait and optional latency hook.
                return before_verify()

            def finish():
                result = end()
                self._window_open = False
                return result

            def cancel():
                self._aborting = True
                try:
                    return abort()
                finally:
                    self._aborting = False
                    self._window_open = False

            self._abort = cancel

            def policy_stats():
                result = dict(stats())
                result.update({'policy': self.policy, 'definition': DEFINITIONS[self.policy],
                               'baseline_scope': 'Runtime timing ablation with unchanged demand cache and capacity'})
                return result

            for name, value in (('begin_window', start), ('after_draft_step', after),
                                ('before_verify', verify), ('end_window', finish),
                                ('abort_window', cancel), ('get_prefetch_policy_stats', policy_stats)):
                self._patches.set(controller, name, value)
            if self.policy == 'none':
                self._patches.set(controller, 'update_from_bound_routes', lambda: None)
        except BaseException:
            self._patches.restore()
            self._active = False
            raise
        return self

    def __exit__(self, *exc):
        try:
            if self._window_open:
                self._abort()
        finally:
            self._window_open = False
            self._patches.restore()
            self._active = False

    def metadata(self):
        return {'policy': self.policy, 'definition': DEFINITIONS[self.policy],
                'cache_capacity': self.controller.max_cached_experts,
                'demand_policy': getattr(self.controller.expert_manager, 'specter_async_demand_policy', None),
                'baseline_scope': 'Runtime timing ablation; no external baseline identity or memory-fairness claim'}
