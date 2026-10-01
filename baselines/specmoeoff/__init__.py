"""Author-adapted B=1 SpecMoEOff comparison, not the official system."""


def load_backend(case, device):
    from .backend import load_backend as load
    return load(case, device)
