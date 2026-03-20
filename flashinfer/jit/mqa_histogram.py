"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags


def gen_mqa_histogram_module() -> JitSpec:
    # Kernels use SM100a-specific instructions: TMA, tcgen05 tensor cores.
    nvcc_flags = sm100a_nvcc_flags + ["-lineinfo"]
    return gen_jit_spec(
        "mqa_histogram",
        [
            jit_env.FLASHINFER_CSRC_DIR / "mqa_metadata.cu",
            jit_env.FLASHINFER_CSRC_DIR / "mqa_v2_hist.cu",
            jit_env.FLASHINFER_CSRC_DIR / "mqa_v2.cu",
            jit_env.FLASHINFER_CSRC_DIR / "fast_topk_clusters.cu",
            jit_env.FLASHINFER_CSRC_DIR / "fast_topk_clusters_exact.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mqa_histogram_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )
