"""Workload pairing and validation for optional experiment plans."""
import unittest

from benchmarks.experiment_plan import build_cases, seed_for, select_inputs
from benchmarks.run_experiments import build_parser


class Tokenizer:
    def encode(self, text):
        return [int(token) for token in text.split()]


def arguments(experiment, *options):
    return build_parser().parse_args(['--experiment', experiment, *options])


class ExperimentPlanTests(unittest.TestCase):
    def test_length_sweep_uses_common_qualifying_source_inputs(self):
        cases = build_cases(arguments('sensitivity', '--datasets', 'GK', '--prefix-tokens', '2',
                                     '--prefix-lengths', '2', '4'))
        prompts = ['', '1 2', '1 2 3 4 5', '4 3 2 1', '7']
        required = max(case.prefix_tokens for case in cases)
        samples = select_inputs(prompts, Tokenizer(), count=2, required_tokens=required)
        self.assertEqual([sample['source_index'] for sample in samples], [2, 3])
        for sample in samples:
            short = sample['input_ids'][:2]
            long = sample['input_ids'][:4]
            self.assertEqual(short, long[:2])
            self.assertEqual(sample['source_tokens'], len(sample['input_ids']))
        with self.assertRaisesRegex(ValueError, 'longer prompts'):
            select_inputs(prompts, Tokenizer(), count=3, required_tokens=required)

    def test_allow_short_is_explicit_and_keeps_original_indices(self):
        prompts = [None, ' ', '1 2', '3 4 5 6']
        strict = select_inputs(prompts, Tokenizer(), 1, 4)
        short = select_inputs(prompts, Tokenizer(), 1, 4, allow_short=True)
        self.assertEqual(strict[0]['source_index'], 3)
        self.assertEqual(short[0]['source_index'], 2)
        self.assertEqual(short[0]['input_ids'], [1, 2])

    def test_seeds_pair_by_source_and_repeat_independent_of_case_order(self):
        cases = build_cases(arguments('cache', '--policies', 'none', 'early', 'adaptive'))
        paired = {case.label: seed_for(42, case.dataset, 17, 2) for case in cases}
        reversed_pairs = {case.label: seed_for(42, case.dataset, 17, 2) for case in reversed(cases)}
        self.assertEqual(paired, reversed_pairs)
        self.assertEqual(len(set(paired.values())), 1)
        self.assertNotEqual(seed_for(42, 'GK', 17, 2), seed_for(42, 'GK', 18, 2))
        self.assertNotEqual(seed_for(42, 'GK', 17, 2), seed_for(42, 'GK', 17, 3))
        self.assertNotEqual(seed_for(42, 'GK', 17, 2), seed_for(42, 'WT', 17, 2))
        self.assertTrue(0 <= seed_for(2**50, 'GK', 17, 100) < 2**31)

    def test_invalid_error_rates_and_duplicate_sweeps_are_rejected(self):
        for rate in ('nan', 'inf', '-inf', '-0.1', '1.1'):
            args = arguments('oracle')
            args.error_rates = [float(rate)]
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                build_cases(args)
        for args in (
            arguments('oracle', '--error-rates', '0.5', '0.5'),
            arguments('working-set', '--depths', '4', '4'),
            arguments('cache', '--residents', '2', '2'),
            arguments('cache', '--policies', 'none', 'none'),
            arguments('routing', '--datasets', 'GK', 'GK'),
        ):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, 'unique'):
                build_cases(args)

    def test_model_capacity_and_nonempty_measurements_are_validated(self):
        for options in (('--resident', '0'), ('--resident', '17'), ('--depth', '0'),
                        ('--prefix-tokens', '0'), ('--tokens', '1'), ('--num-data', '0'),
                        ('--repeats', '0'), ('--residents', '2', '17')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_cases(arguments('cache', '--model', 'phimoe', *options))

    def test_custom_prompts_and_partial_ablation_names_are_explicit(self):
        custom = build_cases(arguments('routing', '--prompts-json', 'prompts.json'))
        self.assertEqual([case.dataset for case in custom], ['CUSTOM'])
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            build_cases(arguments('routing', '--prompts-json', 'prompts.json', '--datasets', 'GK'))
        variants = build_cases(arguments('ablation'))
        self.assertEqual([(case.policy, case.kv_mode, case.expert_mode) for case in variants],
                         [('adaptive', 'shared', 'overlap'), ('none', 'shared', 'overlap'),
                          ('none', 'copied', 'overlap'), ('none', 'copied', 'serial')])

    def test_token_budget_preserves_full_depth_and_later_commit(self):
        for experiment in ('routing', 'working-set', 'sensitivity', 'cache', 'ablation',
                           'latency', 'memory', 'oracle'):
            options = ['--depths', '4', '16'] if experiment == 'working-set' else ['--depth', '16']
            for budget in (2, 16, 17):
                with self.subTest(experiment=experiment, budget=budget), self.assertRaisesRegex(ValueError, 'maximum depth'):
                    build_cases(arguments(experiment, *options, '--tokens', str(budget)))
            cases = build_cases(arguments(experiment, *options, '--tokens', '18'))
            self.assertEqual(max(case.depth for case in cases), 16)
        # Matched-input fidelity does not report generated-token latency.
        self.assertTrue(build_cases(arguments('fidelity', '--depth', '16', '--tokens', '2')))

    def test_oracle_requires_a_window_that_can_prefetch(self):
        with self.assertRaisesRegex(ValueError, 'submits predictions'):
            build_cases(arguments('oracle', '--depth', '1', '--tokens', '3'))
        self.assertTrue(build_cases(arguments('oracle', '--depth', '2', '--tokens', '4')))
        self.assertTrue(build_cases(arguments('routing', '--depth', '1', '--tokens', '3')))


if __name__ == '__main__':
    unittest.main()
