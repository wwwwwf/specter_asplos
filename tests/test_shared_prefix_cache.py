"""CPU checks for authoritative KV, scratch lifetime, and allocation behavior."""
import unittest
from unittest import mock

import torch
from transformers import DynamicCache

from speculative_inference_controller.shared_prefix_cache import SharedPrefixDraftCache


def committed_cache(length, key_width=3, value_width=5, heads=2, layers=2, dtype=torch.float32):
    cache = DynamicCache()
    for layer in range(layers):
        key = torch.arange(2 * heads * length * key_width, dtype=dtype).reshape(2, heads, length, key_width)
        value = torch.arange(2 * heads * length * value_width, dtype=dtype).reshape(2, heads, length, value_width)
        cache.update(key + 100 * layer, value + 1000 * layer, layer)
    return cache


class SharedPrefixCacheTests(unittest.TestCase):
    def test_reserved_scratch_reuses_storage_and_only_materializes_attention(self):
        target = committed_cache(3, layers=1)
        prefix_key, prefix_value = target.key_cache[0].clone(), target.value_cache[0].clone()
        branch = SharedPrefixDraftCache(target, scratch_capacity=4)
        keys, values = [], []
        storage = None
        for step in range(4):
            key = torch.full((2, 2, 1, 3), -10. - step)
            value = torch.full((2, 2, 1, 5), -20. - step)
            keys.append(key)
            values.append(value)
            with mock.patch('speculative_inference_controller.shared_prefix_cache.torch.cat', wraps=torch.cat) as concatenate:
                actual_key, actual_value = branch.update(key, value, 0)
            # Two full attention materializations remain, but neither appending
            # scratch nor capacity management calls cat.
            self.assertEqual(concatenate.call_count, 2)
            self.assertTrue(actual_key.is_contiguous())
            self.assertTrue(actual_value.is_contiguous())
            self.assertTrue(torch.equal(actual_key, torch.cat([prefix_key, *keys], -2)))
            self.assertTrue(torch.equal(actual_value, torch.cat([prefix_value, *values], -2)))
            current = (branch._scratch_key_buffers[0].data_ptr(), branch._scratch_value_buffers[0].data_ptr())
            if storage is None:
                storage = current
            self.assertEqual(current, storage)
            self.assertEqual(branch._scratch_key_buffers[0].shape[-2], 4)
            self.assertIs(branch.key_cache[0], target.key_cache[0])
        self.assertTrue(torch.equal(target.key_cache[0], prefix_key))
        self.assertTrue(torch.equal(target.value_cache[0], prefix_value))
        self.assertEqual(target.get_seq_length(), 3)

    def test_rebase_discards_all_old_scratch_and_refreshes_target_aliases(self):
        original = committed_cache(2)
        branch = SharedPrefixDraftCache(original, scratch_capacity=8)
        for layer in range(2):
            branch.update(torch.full((2, 2, 6, 3), -999.), torch.full((2, 2, 6, 5), -777.), layer)
        old_buffers = [(k.data_ptr(), v.data_ptr()) for k, v in zip(
            branch._scratch_key_buffers, branch._scratch_value_buffers)]
        # Both a longer target prefix and a shorter prefix after rollback must
        # replace all draft-derived state; old scratch is intentionally poisoned.
        for length in (5, 1, 7):
            target = committed_cache(length)
            branch.rebase(target)
            self.assertEqual(branch.get_seq_length(), length)
            self.assertEqual(branch._seen_tokens, length)
            self.assertEqual(branch.scratch_keys, [None, None])
            for layer in range(2):
                self.assertIs(branch.key_cache[layer], target.key_cache[layer])
                key = torch.full((2, 2, 1, 3), 50. + layer)
                value = torch.full((2, 2, 1, 5), 70. + layer)
                actual_key, actual_value = branch.update(key, value, layer)
                self.assertTrue(torch.equal(actual_key, torch.cat((target.key_cache[layer], key), -2)))
                self.assertTrue(torch.equal(actual_value, torch.cat((target.value_cache[layer], value), -2)))
                self.assertEqual((branch._scratch_key_buffers[layer].data_ptr(),
                    branch._scratch_value_buffers[layer].data_ptr()), old_buffers[layer])
                self.assertEqual(branch.get_seq_length(layer), length + 1)

    def test_dynamic_growth_preserves_contents_and_independent_kv_widths(self):
        target = committed_cache(1, layers=1, dtype=torch.float16)
        branch = SharedPrefixDraftCache(target)
        keys, values = [], []
        for tokens in (1, 1, 3, 1):
            key = torch.full((2, 2, tokens, 3), float(tokens), dtype=torch.float16)
            value = torch.full((2, 2, tokens, 5), float(-tokens), dtype=torch.float16)
            keys.append(key)
            values.append(value)
            actual_key, actual_value = branch.update(key, value, 0)
            self.assertTrue(torch.equal(actual_key, torch.cat([target.key_cache[0], *keys], -2)))
            self.assertTrue(torch.equal(actual_value, torch.cat([target.value_cache[0], *values], -2)))
            self.assertGreaterEqual(branch._scratch_key_buffers[0].shape[-2], sum(x.shape[-2] for x in keys))
        self.assertEqual(branch.get_seq_length(), 7)
        self.assertEqual(branch._seen_tokens, 7)

    def test_dtype_changes_preserve_cat_promotion_without_changing_target(self):
        target = committed_cache(1, layers=1, dtype=torch.float16)
        branch = SharedPrefixDraftCache(target, scratch_capacity=4)
        key1, value1 = torch.ones(2, 2, 1, 3, dtype=torch.float16), torch.ones(2, 2, 1, 5, dtype=torch.float16)
        branch.update(key1, value1, 0)
        key2, value2 = torch.full((2, 2, 1, 3), 1.0001), torch.full((2, 2, 1, 5), 2.0001)
        key, value = branch.update(key2, value2, 0)
        self.assertTrue(torch.equal(key, torch.cat((target.key_cache[0], key1, key2), -2)))
        self.assertTrue(torch.equal(value, torch.cat((target.value_cache[0], value1, value2), -2)))
        self.assertEqual(key.dtype, torch.float32)
        self.assertEqual(target.key_cache[0].dtype, torch.float16)

    def test_phi_qwen_and_deepseek_layouts_and_short_window(self):
        for heads, key_width, value_width in ((8, 128, 128), (16, 128, 128), (16, 192, 128)):
            with self.subTest(heads=heads, key_width=key_width, value_width=value_width):
                target = committed_cache(3, key_width, value_width, heads, dtype=torch.float16)
                branch = SharedPrefixDraftCache(target, scratch_capacity=16)
                for layer in range(2):
                    key = torch.randn(2, heads, 2, key_width, dtype=torch.float16)
                    value = torch.randn(2, heads, 2, value_width, dtype=torch.float16)
                    actual_key, actual_value = branch.update(key, value, layer)
                    self.assertTrue(torch.equal(actual_key, torch.cat((target.key_cache[layer], key), -2)))
                    self.assertTrue(torch.equal(actual_value, torch.cat((target.value_cache[layer], value), -2)))
                    self.assertEqual(actual_key.shape[-2], 5)  # no unused capacity exposed
                branch.rebase(target)
                key = torch.zeros(2, heads, 1, key_width, dtype=torch.float16)
                value = torch.zeros(2, heads, 1, value_width, dtype=torch.float16)
                self.assertEqual(branch.update(key, value, 0)[0].shape[-2], 4)
                self.assertEqual(branch.get_seq_length(1), 3)

    def test_invalid_update_does_not_advance_visible_state(self):
        target = committed_cache(2, layers=1)
        branch = SharedPrefixDraftCache(target, scratch_capacity=2)
        with self.assertRaises(ValueError):
            branch.update(torch.zeros(2, 2, 2, 3), torch.zeros(2, 2, 1, 5), 0)
        self.assertEqual(branch.get_seq_length(), 2)
        self.assertEqual(branch._seen_tokens, 2)
        self.assertIsNone(branch.scratch_keys[0])
        with self.assertRaises(ValueError):
            branch.update(torch.zeros(2, 3, 1, 3), torch.zeros(2, 2, 1, 5), 0)
        with self.assertRaises(RuntimeError):
            branch.crop(1)
        with self.assertRaises(RuntimeError):
            branch.update(torch.zeros(2, 2, 1, 3), torch.zeros(2, 2, 1, 5), 1)
        with self.assertRaises(ValueError):
            SharedPrefixDraftCache(target, scratch_capacity=-1)


if __name__ == '__main__':
    unittest.main()
