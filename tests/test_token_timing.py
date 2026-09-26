import contextlib
import io
import unittest
import torch
from benchmarks.token_timing import TokenCommitClock, observe_spec_inf
from speculative_inference_controller.state import spec_inf
from tests.test_state import ToyModel, Tokenizer


class CountingClock:
    def __init__(self, prefix):
        self.prefix = prefix
        self.counts = []

    def commit(self, total):
        self.counts.append(total - self.prefix)


class TokenTimingTests(unittest.TestCase):
    def test_tpot_denominators_and_single_commit(self):
        # Fixed event times separate TTFT, first-batch size, and E2E averaging.
        class FixedStart:
            def elapsed_time(self, event):
                return event
        clock = TokenCommitClock.__new__(TokenCommitClock)
        clock.start = FixedStart()
        clock.commits = [(5, 100.0), (9, 180.0)]
        result = clock.result(9)
        self.assertEqual(result['ttft_ms'], 100.0)
        self.assertEqual(result['tpot_ms'], 10.0)
        self.assertEqual(result['post_first_batch_ms_per_token'], 20.0)
        with self.assertRaises(ValueError):
            clock.result(10)
        clock.commits = [(1, 100.0)]
        result = clock.result(1)
        self.assertIsNone(result['tpot_ms'])
        self.assertIsNone(result['post_first_batch_ms_per_token'])
        clock.commits = [(5, 100.0)]
        self.assertEqual(clock.result(5)['tpot_ms'], 0.0)

    def test_instrumentation_preserves_output_and_covers_tail(self):
        for offset in [0, 1]:
            for tokens in [1, 4, 9]:
                ids = torch.tensor([[1, 2, 3]])
                clock = CountingClock(ids.shape[1])
                observed = observe_spec_inf(spec_inf, clock)
                with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
                    expected = spec_inf(ToyModel(offset), ToyModel(), ids, tokens, 4, Tokenizer(), sampling_strategy='greedy')
                    actual = observed(ToyModel(offset), ToyModel(), ids, tokens, 4, Tokenizer(), sampling_strategy='greedy')
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(clock.counts[-1], tokens)
                self.assertTrue(all(a < b for a, b in zip([0] + clock.counts, clock.counts)))

    def test_cuda_events_capture_ttft_and_post_first_commit_intervals(self):
        clock = TokenCommitClock(16)
        clock.begin()
        torch.cuda._sleep(1000000)
        clock.commit(21)
        torch.cuda._sleep(1000000)
        clock.commit(25)
        torch.cuda.synchronize()
        result = clock.result(9)
        self.assertGreater(result['ttft_ms'], 0)
        self.assertGreater(result['decode_after_first_commit_ms'], 0)
        self.assertEqual(result['first_commit_tokens'], 5)
        self.assertEqual(result['tokens_after_first_commit'], 4)
        self.assertAlmostEqual(result['post_first_batch_ms_per_token'], 2 * result['tpot_ms'])

if __name__ == '__main__':
    unittest.main()
