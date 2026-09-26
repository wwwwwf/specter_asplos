"""PIO incremental route counts and Top-M expert selection."""
import torch


class IncrementalRoutingPatternAnalyzer:
    def __init__(self, layer_num, gamma, max_cached_experts, num_experts,
                 topk, expert_manager, skip_fisrt_layer=False,
                 device="cuda:0", sticky_tiebreak=False, prefetch_min_count=1,
                 resident_first_selection=False):
        self.layer_num = layer_num
        self.gamma = gamma
        self.max_cached_experts = max_cached_experts
        self.num_experts = num_experts
        self.topk = topk
        self.expert_manager = expert_manager
        self.skip_fisrt_layer = skip_fisrt_layer
        self.device = device
        self.sticky_tiebreak = sticky_tiebreak
        self.prefetch_min_count = prefetch_min_count
        self.resident_first_selection = resident_first_selection
        if prefetch_min_count <= 0:
            raise ValueError("prefetch_min_count must be positive")
        expert_manager.enable_specter_async_loading()
        self.stream = torch.cuda.Stream(device=device)
        self.activation_counters = torch.zeros((layer_num, num_experts), dtype=torch.int32, device=device)
        self._sticky_prefetch_mask = torch.zeros_like(self.activation_counters, dtype=torch.bool)
        self.mutex = None
        self._draft_routes = []
        self._lookahead_ready = torch.cuda.Event(enable_timing=False, blocking=False)
        self.route_version = 0

    def bind_draft_routes(self, route_modules):
        route_modules = list(route_modules)
        if len(route_modules) != self.layer_num:
            raise ValueError(
                "Expected "
                f"{self.layer_num} draft MoE routes, "
                f"got {len(route_modules)}"
            )
        self._draft_routes = route_modules

    def _record_lookahead_ready(self):
        current_stream = torch.cuda.current_stream(
            device=self.activation_counters.device
        )
        self._lookahead_ready.record(current_stream)

    def update(self, layer_idx, selected_experts):
        if self.skip_fisrt_layer:
            layer_idx = layer_idx - 1
        selected_experts = selected_experts.long().flatten()
        ones = torch.ones_like(selected_experts, dtype=torch.int32)
        self.activation_counters[layer_idx].index_add_(
            0,
            selected_experts,
            ones,
        )
        if layer_idx == self.layer_num - 1:
            self._record_lookahead_ready()
            self.route_version += 1

    def update_from_bound_routes(self):
        if not self._draft_routes:
            return

        selected_by_layer = []
        for route in self._draft_routes:
            selected_experts = getattr(route, "_selected_experts", None)
            if selected_experts is None:
                raise RuntimeError(
                    "Draft route did not expose _selected_experts"
                )
            selected_by_layer.append(
                selected_experts.detach().reshape(-1).long()
            )

        selected = torch.stack(selected_by_layer, dim=0)
        if selected.device != self.activation_counters.device:
            raise RuntimeError(
                "Draft routes and activation counters must share a device"
            )
        ones = torch.ones_like(selected, dtype=torch.int32)
        self.activation_counters.scatter_add_(1, selected, ones)
        self._record_lookahead_ready()
        self.route_version += 1

    def clear(self):
        # Top-M may still read the previous tensor on the prefetch stream.
        self.activation_counters = torch.zeros_like(
            self.activation_counters
        )

    def get_all_layer_topm_uids(self, count):
        layer_count, _ = self.activation_counters.shape
        counters = self.activation_counters
        scores = counters
        if self.sticky_tiebreak:
            # Keep count ordering exact; persistence only breaks equal scores.
            scores = counters * 2 + self._sticky_prefetch_mask.to(
                counters.dtype
            )
        candidate_count = (
            self.num_experts if self.resident_first_selection else count
        )
        _, indices = torch.topk(
            scores,
            k=candidate_count,
            dim=1,
            largest=True,
            sorted=self.resident_first_selection,
        )

        selected_counts = counters.gather(1, indices)
        mask = selected_counts >= self.prefetch_min_count
        layer_ids = torch.arange(
            layer_count,
            device=self.activation_counters.device,
        ).unsqueeze(1)
        selected_layers = layer_ids.expand(-1, candidate_count)[mask]
        selected_experts = indices[mask]
        packed = torch.stack(
            (selected_layers, selected_experts),
            dim=1,
        )

        # A single D2H synchronization materializes the complete plan.
        pairs = packed.cpu().tolist()
        uidss = [[] for _ in range(layer_count)]
        sticky_pairs = []
        if self.resident_first_selection:
            registered = getattr(
                self.expert_manager,
                "registered_experts",
                {},
            )
            candidates = [[] for _ in range(layer_count)]
            for layer_idx, expert_idx in pairs:
                target_layer_idx = (
                    layer_idx + 1 if self.skip_fisrt_layer else layer_idx
                )
                uid = (target_layer_idx, expert_idx)
                info = registered.get(uid)
                resident = info is not None and not info.offloaded
                candidates[layer_idx].append((not resident, uid))
            for layer_idx, layer_candidates in enumerate(candidates):
                # Prediction count remains the primary order within each
                # class. Keeping predicted residents avoids replacing a
                # useful cache entry merely to prefetch another candidate.
                selected = sorted(
                    layer_candidates,
                    key=lambda item: item[0],
                )[:count]
                uidss[layer_idx] = [uid for _, uid in selected]
                sticky_pairs.extend(
                    (layer_idx, uid[1]) for _, uid in selected
                )
        else:
            for layer_idx, expert_idx in pairs:
                target_layer_idx = (
                    layer_idx + 1 if self.skip_fisrt_layer else layer_idx
                )
                uidss[layer_idx].append(
                    (target_layer_idx, expert_idx)
                )
                sticky_pairs.append((layer_idx, expert_idx))
        if self.sticky_tiebreak:
            self._sticky_prefetch_mask.zero_()
            if sticky_pairs:
                sticky_indices = torch.tensor(
                    sticky_pairs,
                    dtype=torch.long,
                    device=self.activation_counters.device,
                )
                self._sticky_prefetch_mask[
                    sticky_indices[:, 0], sticky_indices[:, 1]
                ] = True
        return uidss
