"""Build/load only the project-local Specter extension; never fall back."""
from functools import lru_cache
import os
from pathlib import Path

@lru_cache(None)
def load_extension():
    os.environ.setdefault('CUDA_HOME', '/usr/local/cuda')
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent
    v = root / 'csrc/vllm'
    from configuration import get_config, external_path
    build = external_path(get_config().cache_dir / 'kernels' / 'specter_cuda')
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    os.environ.setdefault('MAX_JOBS', '2')
    os.environ.setdefault('CUDA_HOME', '/usr/local/cuda')
    sources = [root/'csrc/bindings.cpp', v/'moe/moe_align_sum_kernels.cu',
               v/'moe/marlin_moe_wna16/ops.cu',
               v/'moe/marlin_moe_wna16/kernel_fp16_ku4b8.cu',
               v/'moe/marlin_moe_wna16/kernel_bf16_ku4b8.cu',
               root/'csrc/gptq/marlin_repack.cu']
    return load(name='specter_cuda', sources=[str(p) for p in sources],
        extra_include_paths=[str(v)], build_directory=str(build),
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--use_fast_math', '-DMARLIN_NAMESPACE_NAME=specter_marlin_moe'],
        verbose=os.environ.get('SPECTER_BUILD_VERBOSE') == '1')

if __name__ == '__main__':
    print(load_extension().__file__)
