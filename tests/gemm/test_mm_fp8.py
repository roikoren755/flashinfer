from typing import Dict
from flashinfer.utils import get_compute_capability
import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_fp8
from tests.utils_fp8 import to_float8
from flashinfer import prepare_low_latency_gemm_weights

_cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("n", [2560, 5120])
@pytest.mark.parametrize("k", [8192, 16384, 32768])
@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mat2_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16])
def test_mm_fp8(
    m: int,
    n: int,
    k: int,
    input_dtype: torch.dtype,
    mat2_dtype: torch.dtype,
    res_dtype: torch.dtype,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("mm_fp8 is only supported on Blackwell GPUs.")

    torch.manual_seed(123)
    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    res = torch.zeros([m, n], device="cuda", dtype=res_dtype)
    global_scale = input_inv_s * mat2_inv_s

    prepared_weights = prepare_low_latency_gemm_weights(
        mat2_fp8, _cache_permute_indices
    )
    with autotune():
        mm_fp8(
            input_fp8,
            prepared_weights,
            global_scale,
            out=res,
        )

    reference = torch.mm(input, mat2.transpose(-2, -1))
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > 0.99


@pytest.mark.parametrize(
    ("out", "message"),
    [
        (torch.empty((2, 5), dtype=torch.bfloat16), "Output shape mismatch"),
        (torch.empty((2, 4), dtype=torch.float32), "Unsupported output dtype"),
    ],
)
def test_mm_fp8_trtllm_validates_destination(out, message):
    a = torch.empty((2, 8), dtype=torch.float8_e4m3fn)
    weight = torch.empty((4, 8), dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match=message):
        mm_fp8(
            a,
            weight.T,
            torch.ones(1, dtype=torch.float32),
            out=out,
            backend="trtllm",
        )


def test_mm_fp8_trtllm_relu2():
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("mm_fp8 is only supported on Blackwell GPUs.")
    device = torch.device("cuda")
    torch.manual_seed(2234)
    m, n, k = 8, 128, 128
    a_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device) / (k**0.25)
    b_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device) / (k**0.25)
    a, a_inverse_scale = to_float8(a_bf16)
    b, b_inverse_scale = to_float8(b_bf16)
    alpha = a_inverse_scale * b_inverse_scale
    expected = torch.relu(a_bf16.float() @ b_bf16.float().T).square()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)

    result = mm_fp8(
        a,
        b.T,
        alpha,
        out=out,
        backend="trtllm",
        activation="relu2",
    )

    assert result.data_ptr() == out.data_ptr()
    cosine = F.cosine_similarity(
        result.float().reshape(-1), expected.reshape(-1), dim=0
    )
    assert cosine > 0.99


def test_mm_fp8_trtllm_quantized_output_is_gemm_input():
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("mm_fp8 is only supported on Blackwell GPUs.")
    device = torch.device("cuda")
    torch.manual_seed(2235)
    m, intermediate_size, hidden_size = 8, 128, 128
    a_bf16 = torch.randn((m, hidden_size), dtype=torch.bfloat16, device=device) / (
        hidden_size**0.25
    )
    up_weight_bf16 = torch.randn(
        (intermediate_size, hidden_size), dtype=torch.bfloat16, device=device
    ) / (hidden_size**0.25)
    a, a_inverse_scale = to_float8(a_bf16)
    up_weight, up_weight_inverse_scale = to_float8(up_weight_bf16)
    up_alpha = a_inverse_scale * up_weight_inverse_scale
    expected_up = torch.relu(a_bf16.float() @ up_weight_bf16.float().T).square()
    output_quant_scale = (448.0 / expected_up.abs().amax().clamp_min(1e-6)).reshape(1)
    up_out = torch.empty(
        (m, intermediate_size), dtype=torch.float8_e4m3fn, device=device
    )

    up_result = mm_fp8(
        a,
        up_weight.T,
        up_alpha,
        out_dtype=torch.float8_e4m3fn,
        out=up_out,
        backend="trtllm",
        activation="relu2",
        output_quant_scale=output_quant_scale,
    )

    down_weight_bf16 = torch.randn(
        (hidden_size, intermediate_size), dtype=torch.bfloat16, device=device
    ) / (intermediate_size**0.25)
    down_weight, down_weight_inverse_scale = to_float8(down_weight_bf16)
    down_alpha = output_quant_scale.reciprocal() * down_weight_inverse_scale
    down_out = torch.empty((m, hidden_size), dtype=torch.bfloat16, device=device)
    down_result = mm_fp8(
        up_out,
        down_weight.T,
        down_alpha,
        out=down_out,
        backend="trtllm",
    )
    expected_down = expected_up @ down_weight_bf16.float().T

    assert up_result.data_ptr() == up_out.data_ptr()
    assert down_result.data_ptr() == down_out.data_ptr()
    cosine = F.cosine_similarity(
        down_result.float().reshape(-1), expected_down.reshape(-1), dim=0
    )
    assert cosine > 0.99


def test_mm_fp8_trtllm_quantized_output_cuda_graph():
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("mm_fp8 is only supported on Blackwell GPUs.")
    device = torch.device("cuda")
    torch.manual_seed(2236)
    m, n, k = 8, 128, 128
    a_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device) / (k**0.25)
    b_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device) / (k**0.25)
    a, a_inverse_scale = to_float8(a_bf16)
    b, b_inverse_scale = to_float8(b_bf16)
    alpha = a_inverse_scale * b_inverse_scale
    expected = torch.relu(a_bf16.float() @ b_bf16.float().T).square()
    output_quant_scale = (448.0 / expected.abs().amax().clamp_min(1e-6)).reshape(1)
    out = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=device)

    def run(destination):
        return mm_fp8(
            a,
            b.T,
            alpha,
            out_dtype=torch.float8_e4m3fn,
            out=destination,
            backend="trtllm",
            activation="relu2",
            output_quant_scale=output_quant_scale,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(out)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_result = run(out)

    a_new_bf16 = torch.randn_like(a_bf16) / (k**0.25)
    a_new = (
        (a_new_bf16.float() / a_inverse_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    a.copy_(a_new)
    out.view(torch.uint8).zero_()
    graph.replay()
    torch.cuda.synchronize()

    eager_out = torch.empty_like(out)
    eager_result = run(eager_out)
    torch.cuda.synchronize()

    assert graph_result.data_ptr() == out.data_ptr()
    assert eager_result.data_ptr() == eager_out.data_ptr()
    assert torch.equal(out.view(torch.uint8), eager_out.view(torch.uint8))


if __name__ == "__main__":
    pytest.main([__file__])
