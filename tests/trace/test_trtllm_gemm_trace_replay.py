# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay the committed generated-TRTLLM GEMM traces on Blackwell."""

import json
from pathlib import Path

import pytest
import torch

import flashinfer
from flashinfer.trace.templates.gemm import (
    mm_fp4_trtllm_identity_bf16_r128c4_trace,
    mm_fp4_trtllm_relu2_bf16_r128c4_trace,
    mm_fp4_trtllm_relu2_bf16_r8c4_trace,
    mm_fp4_trtllm_relu2_nvfp4_r128c4_trace,
    mm_fp8_trtllm_identity_bf16_trace,
    mm_fp8_trtllm_relu2_bf16_trace,
    mm_fp8_trtllm_relu2_fp8_trace,
)
from flashinfer.utils import get_compute_capability


_TRACE_DIR = Path(__file__).parent / "fi_trace_out"
_BASE_INIT = {"M": 8, "device": "cuda", "seed": 7}
_FP4_R128_INIT = {**_BASE_INIT, "A_scale_rows": 128}
_FP4_R8_INIT = {**_BASE_INIT, "A_scale_rows": 8}
_FP4_QUANTIZED_INIT = {
    **_FP4_R128_INIT,
    "N_packed": 64,
    "M_scale": 128,
    "N_scale": 8,
}
_TRACE_CASES = (
    (
        "mm_fp8_trtllm_identity_bf16_k128_n128.json",
        flashinfer.mm_fp8,
        mm_fp8_trtllm_identity_bf16_trace,
        _BASE_INIT,
    ),
    (
        "mm_fp8_trtllm_relu2_bf16_k128_n128.json",
        flashinfer.mm_fp8,
        mm_fp8_trtllm_relu2_bf16_trace,
        _BASE_INIT,
    ),
    (
        "mm_fp8_trtllm_relu2_fp8_k128_n128.json",
        flashinfer.mm_fp8,
        mm_fp8_trtllm_relu2_fp8_trace,
        _BASE_INIT,
    ),
    (
        "mm_fp4_trtllm_identity_bf16_r128c4_kp64_n128_bs16.json",
        flashinfer.mm_fp4,
        mm_fp4_trtllm_identity_bf16_r128c4_trace,
        _FP4_R128_INIT,
    ),
    (
        "mm_fp4_trtllm_relu2_bf16_r128c4_kp64_n128_bs16.json",
        flashinfer.mm_fp4,
        mm_fp4_trtllm_relu2_bf16_r128c4_trace,
        _FP4_R128_INIT,
    ),
    (
        "mm_fp4_trtllm_relu2_bf16_r8c4_kp64_n128_bs16.json",
        flashinfer.mm_fp4,
        mm_fp4_trtllm_relu2_bf16_r8c4_trace,
        _FP4_R8_INIT,
    ),
    (
        "mm_fp4_trtllm_relu2_nvfp4_r128c4_kp64_n128_bs16.json",
        flashinfer.mm_fp4,
        mm_fp4_trtllm_relu2_nvfp4_r128c4_trace,
        _FP4_QUANTIZED_INIT,
    ),
)


def _require_generated_trtllm_gemm() -> None:
    if not torch.cuda.is_available():
        pytest.skip("generated TRTLLM GEMM traces require CUDA")
    if get_compute_capability(torch.device("cuda")) not in {(10, 0), (10, 3)}:
        pytest.skip("generated TRTLLM GEMM traces require SM100 or SM103")


def _exec_in_fresh_namespace(source: str) -> dict:
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize(
    "filename,api,template,init_kwargs",
    _TRACE_CASES,
    ids=[case[0].removesuffix(".json") for case in _TRACE_CASES],
)
def test_trtllm_gemm_committed_trace_init_replays(
    filename, api, template, init_kwargs
) -> None:
    """The serialized init must construct a directly runnable physical contract."""

    _require_generated_trtllm_gemm()
    document = json.loads((_TRACE_DIR / filename).read_text())
    init_namespace = _exec_in_fresh_namespace(document["init"])
    inputs = init_namespace[template.init.__name__](**init_kwargs)

    result = api(**inputs)
    torch.cuda.synchronize()
    assert result.data_ptr() == inputs["out"].data_ptr()

    reference_namespace = _exec_in_fresh_namespace(document["reference"])
    reference_inputs = {
        name: value
        for name, value in inputs.items()
        if name not in {"out", "out_scale"}
    }
    reference = reference_namespace[template.reference.__name__](**reference_inputs)
    actual = (result, inputs["out_scale"]) if "out_scale" in inputs else result
    assert template.check(reference, actual)
