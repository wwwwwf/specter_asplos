"""Restore the initial expert residents and LRU order outside timed regions."""
from collections import OrderedDict
import torch


class ResidentReset:
    def __init__(self, manager):
        self.manager = manager
        manager.enable_specter_async_loading()
        self.residents = {group_id: tuple(group.main_infos) for group_id, group in manager.group_infos.items() if group.main_infos}

    def restore(self):
        torch.cuda.synchronize()
        for group_id, uids in self.residents.items():
            self.manager.prefetch_experts_async(*uids, unordered=False)
            group = self.manager.group_infos[group_id]
            if set(group.main_infos) != set(uids):
                raise RuntimeError(f'Resident reset failed: {group_id}')
            group.main_infos = OrderedDict((uid, self.manager.registered_experts[uid]) for uid in uids)
            group.hits = group.misses = 0
        torch.cuda.synchronize()


