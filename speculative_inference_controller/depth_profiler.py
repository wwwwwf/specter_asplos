"""SIC offline K selection by mean committed-token decode TPOT."""
import math
import statistics

class LightweightSpeculativeDepthProfiler:
    candidates = (2, 4, 6, 8, 12, 16, 20, 24, 32)

    def profile(self, evaluate, prompts, repeats=3):
        if not prompts or repeats < 1:
            raise ValueError('Representative prompts and positive repeats required')
        records = []
        for repeat in range(repeats):
            for prompt_index, prompt in enumerate(prompts):
                # Rotate order to reduce warm-cache and clock bias.
                offset = (repeat + prompt_index) % len(self.candidates)
                order = self.candidates[offset:] + self.candidates[:offset]
                for depth in order:
                    value = float(evaluate(depth, prompt, repeat))
                    if not math.isfinite(value) or value < 0:
                        raise ValueError('TPOT must be finite and nonnegative')
                    records.append(dict(depth=depth, prompt=prompt_index, repeat=repeat, tpot_ms=value))
        means = {k: statistics.mean(r['tpot_ms'] for r in records if r['depth'] == k) for k in self.candidates}
        return {'selected_depth': min(means, key=means.get), 'mean_tpot_ms': means, 'records': records}
