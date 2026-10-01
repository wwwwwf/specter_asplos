"""Exercise measured-memory search without torch, CUDA, or model checkpoints."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from baselines import memory
from baselines.protocol import settings


class FakeBackend:
    def __init__(self):
        self.metadata = {'expert_slot_bytes': 10, 'cache_group_count': 1}
        self.resident = 1
        self.resizes = []

    def resize_residency(self, resident):
        self.resident = resident
        self.resizes.append(resident)


class BaselineMemoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='baseline-memory-')
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.backend = FakeBackend()
        self.fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock()))
        self.calls = []

    def calibrate(self, peak, budget=100, maximum=16, items=None):
        items = items if items is not None else [('GK', 0, 'text', 'ids')]
        counts = {}

        def measure(backend, ids, tokens, seed, method, record_process_memory=False):
            key = (backend.resident, ids)
            counts[key] = counts.get(key, 0) + 1
            self.calls.append((backend.resident, ids, tokens, seed, method, record_process_memory))
            return {'gpu_memory': {'runtime_gpu_bytes': peak(backend.resident, ids, counts[key])}}

        with patch.dict(sys.modules, {'torch': self.fake_torch}), \
             patch.object(memory.gc, 'collect'), contextlib.redirect_stdout(io.StringIO()):
            return memory.calibrate(self.backend, items, 128, budget, self.output,
                                    measure, 'specmoeoff', maximum)

    def test_nonlinear_growth_does_not_collapse_open_bracket_to_minimum(self):
        # The first byte estimate jumps 1 -> 9; the second estimates 9 -> 1.
        # Capacities 2..8 still need probing; capacity 5 fits exactly.
        result = self.calibrate(lambda resident, *_: 20 * resident)
        self.assertEqual(self.backend.resizes, [1, 9, 5, 6, 5])
        self.assertEqual(result['resident_per_layer'], 5)
        self.assertTrue(result['observations'][-1]['within_budget'])

    def test_exact_maximum_is_restored_and_rechecked_after_failed_neighbor(self):
        result = self.calibrate(lambda resident, *_: 20 + 10 * resident, budget=70, maximum=8)
        self.assertEqual(result['resident_per_layer'], 5)
        self.assertEqual(self.backend.resizes[-2:], [6, 5])
        self.assertEqual(result['observations'][-1]['runtime_gpu_bytes'], 70)
        self.assertEqual(self.backend.resident, 5)

    def test_full_capacity_stops_without_probing_out_of_range(self):
        result = self.calibrate(lambda resident, *_: 7 + 9 * resident, maximum=4)
        self.assertEqual(result['resident_per_layer'], 4)
        self.assertEqual(self.backend.resizes, [1, 4])

    def test_minimum_infeasible_is_persisted_and_raises(self):
        with self.assertRaisesRegex(RuntimeError, 'even with one resident'):
            self.calibrate(lambda resident, *_: 20 + 10 * resident, budget=29)
        report = json.loads((self.output / 'memory_budget.json').read_text())
        self.assertFalse(report['feasible'])
        self.assertEqual(self.backend.resizes, [1])
        self.assertEqual(report['observations'][0]['runtime_gpu_bytes'], 30)

    def test_shrinking_estimate_continues_searching_higher_feasible_candidates(self):
        result = self.calibrate(lambda resident, *_: 10 * resident + resident ** 2, budget=70, maximum=10)
        self.assertEqual(self.backend.resizes, [1, 6, 3, 5, 4])
        self.assertEqual(result['resident_per_layer'], 4)

    def test_every_probe_uses_worst_input_and_preserves_tokens_seeds(self):
        items = [('GK', 0, 'first', 'a'), ('WT', 0, 'second', 'b')]
        result = self.calibrate(lambda resident, ids, _: 10 * resident + (20 if ids == 'b' else 0),
                                budget=70, maximum=8, items=items)
        self.assertEqual(result['resident_per_layer'], 5)
        self.assertEqual(len(self.calls), 2 * len(result['observations']))
        for first, second in zip(self.calls[::2], self.calls[1::2]):
            self.assertEqual(first[2:], (128, 42, 'specmoeoff', True))
            self.assertEqual(second[2:], (128, 1051, 'specmoeoff', True))
        lines = (self.output / 'memory_calibration.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), len(self.calls))

    def test_restored_capacity_that_now_exceeds_budget_is_not_returned(self):
        def peak(resident, _, count):
            return 20 * resident + (10 if resident == 5 and count > 1 else 0)
        result = self.calibrate(peak)
        self.assertEqual(result['resident_per_layer'], 4)
        self.assertEqual(result['observations'][-1]['runtime_gpu_bytes'], 80)

    def test_empty_workload_and_invalid_backend_storage_fail_before_probing(self):
        with self.assertRaises(ValueError):
            self.calibrate(lambda *_: 0, items=[])
        self.backend.metadata['expert_slot_bytes'] = 0
        with self.assertRaises(ValueError):
            self.calibrate(lambda *_: 0)
        self.assertEqual(self.backend.resizes, [])

    def test_reference_requires_requested_tier_cache_depth_and_buffers(self):
        args = SimpleNamespace(model='dsv2lite', memory='low', datasets=['GK'],
                               num_data=1, repeats=1, tokens=128, prefix_tokens=16)
        selected = settings(args.model, args.memory, 'specter')
        config = {'args': vars(args).copy(), 'model_case': {
            key: selected[key] for key in ('offload_per_layer', 'gamma', 'buffer_size')}}
        records = {('GK', 0, 0): {'kind': 'specter', 'gpu_memory': {'runtime_gpu_bytes': 100}}}
        with patch('baselines.compare.read_run', return_value=(config, [], records)):
            self.assertEqual(memory.read_reference(self.output, args)['budget_bytes'], 100)
            for key in ('offload_per_layer', 'gamma', 'buffer_size'):
                with self.subTest(key=key), patch.dict(config['model_case'], {key: selected[key] + 1}), \
                     self.assertRaisesRegex(ValueError, 'tier mismatch'):
                    memory.read_reference(self.output, args)


if __name__ == '__main__':
    unittest.main()
