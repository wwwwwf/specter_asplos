"""Adapted Mixtral-Offloading autoregressive baseline (see PROVENANCE.md)."""


def load_backend(case, device):
    from .backend import load_backend as load
    return load(case, device)
