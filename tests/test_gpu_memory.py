"""No CUDA or torch installation is required for process-memory accounting tests."""
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchmarks.gpu_memory import memory_footprint, memory_snapshot


MIB = 2 ** 20


class GpuMemoryTests(unittest.TestCase):
    def fake_torch(self, gpu_uuid='GPU-selected', allocated=3 * MIB, reserved=4 * MIB):
        properties = SimpleNamespace() if gpu_uuid is None else SimpleNamespace(uuid=gpu_uuid)
        return SimpleNamespace(cuda=SimpleNamespace(
            current_device=Mock(return_value=0),
            get_device_properties=Mock(return_value=properties),
            memory_allocated=Mock(return_value=allocated),
            memory_reserved=Mock(return_value=reserved)))

    def snapshot(self, output, module=None, device=None):
        with patch('benchmarks.gpu_memory.os.getpid', return_value=123), \
             patch('benchmarks.gpu_memory.subprocess.check_output', return_value=output) as query:
            result = memory_snapshot(module or self.fake_torch(), device)
        self.assertEqual(query.call_args.kwargs['timeout'], 10)
        self.assertIn('--query-compute-apps=pid,used_memory,gpu_uuid', query.call_args.args[0])
        return result

    def test_selected_physical_uuid_and_own_pid_not_card_or_visible_index(self):
        module = self.fake_torch()
        result = self.snapshot('999, 40000, GPU-selected\n123, 900, GPU-other\n123, 6, GPU-selected\n', module, 'cuda:0')
        self.assertEqual(result, {'allocated_bytes': 3 * MIB, 'reserved_bytes': 4 * MIB,
                                 'process_bytes': 6 * MIB, 'native_overhead_bytes': 2 * MIB,
                                 'gpu_uuid': 'GPU-selected', 'pid': 123})
        module.cuda.get_device_properties.assert_called_once_with('cuda:0')
        module.cuda.current_device.assert_not_called()

    def test_no_torch_uuid_requires_unique_own_pid_row(self):
        result = self.snapshot('999, 4, GPU-other\n123, 6.5, GPU-selected\n', self.fake_torch(None))
        self.assertEqual(result['process_bytes'], int(6.5 * MIB))
        with self.assertRaisesRegex(RuntimeError, 'found 2'):
            self.snapshot('123, 6, GPU-selected\n123, 7, GPU-other\n', self.fake_torch(None))

    def test_uuid_prefix_case_and_raw_bytes_are_supported(self):
        identifier = '00112233-4455-6677-8899-aabbccddeeff'
        raw = bytes.fromhex(identifier.replace('-', ''))
        for value in (identifier.upper(), ('GPU-' + identifier).encode(), raw):
            result = self.snapshot(f'123, 6, GPU-{identifier}\n', self.fake_torch(value))
            self.assertEqual(result['gpu_uuid'], 'GPU-' + identifier)

    def test_missing_wrong_gpu_and_duplicate_rows_fail_closed(self):
        for output in ('', '999, 4, GPU-selected\n', '123, 6, GPU-other\n',
                       '123, 6, GPU-selected\n123, 6, GPU-selected\n'):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                self.snapshot(output)

    def test_unavailable_or_malformed_memory_is_not_zero(self):
        for value in ('N/A', '[Not Supported]', 'NaN', 'Infinity', '-1', ''):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.snapshot(f'123, {value}, GPU-selected\n')
        for output in ('123, 6\n', 'invalid, 6, GPU-selected\n'):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                self.snapshot(output)

    def test_process_memory_rounding_cannot_make_native_overhead_negative(self):
        result = self.snapshot('123, 3, GPU-selected\n')
        self.assertEqual(result['native_overhead_bytes'], 0)

    def test_telemetry_tool_failure_and_timeout_raise(self):
        failures = (FileNotFoundError(), subprocess.CalledProcessError(1, 'nvidia-smi'),
                    subprocess.TimeoutExpired('nvidia-smi', 10))
        for error in failures:
            with self.subTest(error=type(error)), \
                 patch('benchmarks.gpu_memory.subprocess.check_output', side_effect=error), \
                 self.assertRaisesRegex(RuntimeError, 'telemetry is required'):
                memory_snapshot(self.fake_torch())

    def test_footprint_counts_allocator_peak_and_native_overhead_once(self):
        before = self.snapshot('123, 6, GPU-selected\n')
        after = self.snapshot('123, 8, GPU-selected\n', self.fake_torch(reserved=5 * MIB))
        result = memory_footprint(before, after, 10 * MIB, 7 * MIB)
        self.assertEqual(result['runtime_gpu_bytes'], 13 * MIB)
        self.assertEqual(result['peak_allocated_bytes'], 7 * MIB)
        self.assertEqual(result['before'], before)
        self.assertIsNot(result['before'], before)

    def test_footprint_preserves_larger_endpoint_when_peak_was_reset(self):
        before = self.snapshot('123, 20, GPU-selected\n', self.fake_torch(reserved=18 * MIB))
        after = self.snapshot('123, 6, GPU-selected\n')
        self.assertEqual(memory_footprint(before, after, 8 * MIB, 4 * MIB)['runtime_gpu_bytes'], 20 * MIB)

    def test_footprint_rejects_mismatched_or_inconsistent_evidence(self):
        before = self.snapshot('123, 6, GPU-selected\n')
        for change in ({'pid': 456}, {'gpu_uuid': 'GPU-other'}, {'native_overhead_bytes': 0},
                       {'process_bytes': -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                memory_footprint(before, dict(before, **change), 8 * MIB, 4 * MIB)
        with self.assertRaises(ValueError):
            memory_footprint(before, before, -1, 4 * MIB)


if __name__ == '__main__':
    unittest.main()
