"""Observe actual SD commitment boundaries without changing either algorithm."""
import inspect
import torch


class TokenCommitClock:
    def __init__(self, prefix_tokens):
        self.prefix_tokens = int(prefix_tokens)
        self.start = torch.cuda.Event(enable_timing=True)
        self.commits = []

    def begin(self):
        self.start.record()

    def commit(self, total_tokens):
        generated = int(total_tokens) - self.prefix_tokens
        if generated <= 0 or (self.commits and generated <= self.commits[-1][0]):
            raise ValueError('Committed output lengths must increase')
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.commits.append((generated, event))

    def result(self, expected_tokens):
        if not self.commits or self.commits[-1][0] != expected_tokens:
            raise ValueError('Commit trace does not cover exactly the requested output')
        times = [{'generated': count, 'time_ms': self.start.elapsed_time(event)} for count, event in self.commits]
        first, last = times[0], times[-1]
        elapsed = last['time_ms'] - first['time_ms']
        remaining = expected_tokens - first['generated']
        return {
            'ttft_ms': first['time_ms'],
            'generation_until_last_commit_ms': last['time_ms'],
            'decode_after_first_commit_ms': elapsed,
            'tpot_ms': elapsed / (expected_tokens - 1) if expected_tokens > 1 else None,
            'first_commit_tokens': first['generated'],
            'tokens_after_first_commit': remaining,
            'post_first_batch_ms_per_token': elapsed / remaining if remaining else None,
            'commits': times,
        }


def observe_spec_inf(function, clock):
    """Compile an observed copy; preserve the production/reference sources."""
    source = inspect.getsource(function)
    terminal = '            token_len += 1\n            break'
    iteration = '        step += 1\n'
    if source.count(terminal) != 1 or source.count(iteration) != 1:
        raise RuntimeError('Decoder structure changed: refusing to guess timing boundaries')
    source = source.replace(terminal, '            token_len += 1\n            _commit_clock.commit(token_len)\n            break')
    source = source.replace(iteration, '        step += 1\n        _commit_clock.commit(token_len)\n')
    namespace = dict(function.__globals__)
    namespace['_commit_clock'] = clock
    exec(compile(source, inspect.getsourcefile(function) + ':commit_timing', 'exec'), namespace)
    return namespace[function.__name__]
