import pytest
import torch
import torch.nn.functional as F
from flashinfer import (
    SfLayout,
    autotune,
    mm_fp4,
    nvfp4_quantize,
    mxfp4_quantize,
)
from flashinfer.utils import (
    get_compute_capability,
    is_sm12x_supported,
    version_at_least,
    LibraryError,
)
from flashinfer.gemm.gemm_base import CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR


def _skip_if_trtllm_unsupported():
    device = torch.device("cuda")
    major, minor = get_compute_capability(device)
    compute_capability = major * 10 + minor
    if not mm_fp4.is_backend_supported("trtllm", compute_capability):
        pytest.skip(
            "Skipping test for trtllm because it is not supported on compute "
            f"capability {compute_capability}."
        )


def _quantize_nvfp4(value, *, shuffle, layout=SfLayout.layout_128x4):
    global_scale = ((448 * 6) / value.float().abs().amax().clamp_min(1e-6)).reshape(1)
    quantized, block_scale = nvfp4_quantize(
        value,
        global_scale,
        sfLayout=layout,
        do_shuffle=shuffle,
    )
    return quantized, block_scale, global_scale


def _test_mm_fp4(
    m,
    n,
    k,
    res_dtype,
    backend,
    use_128x4_sf_layout,
    auto_tuning,
    fp4_type,
    activation="none",
):
    use_nvfp4 = fp4_type == "nvfp4"

    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not mm_fp4.is_backend_supported(backend, compute_capability_number):
        pytest.skip(
            f"Skipping test for {backend} because it is not supported on compute capability {compute_capability_number}."
        )

    if backend == "trtllm":
        if res_dtype == torch.float16:
            pytest.skip("Skipping test for trtllm fp4 with float16")
        if compute_capability[0] in [11, 12]:
            pytest.skip("trtllm gemm does not support SM110/SM120/SM121 GPUs.")
    if backend == "cute-dsl":
        if not use_128x4_sf_layout:
            pytest.skip("cute_dsl backend only supports 128x4 SF layout")
        if compute_capability[0] not in [10]:
            pytest.skip("cute_dsl backend only supports SM100/SM103 GPUs.")
    if backend == "b12x":
        if not use_128x4_sf_layout:
            pytest.skip("b12x backend only supports 128x4 SF layout")
        if compute_capability[0] != 12:
            pytest.skip("b12x backend only supports SM120/SM121 GPUs.")
        if not use_nvfp4:
            pytest.skip("b12x backend only supports NVFP4 (sf_vec_size=16).")
        if torch.version.cuda and int(torch.version.cuda.split(".")[0]) < 13:
            pytest.skip("b12x backend requires CUDA 13+.")
    if not use_128x4_sf_layout and backend != "trtllm":
        pytest.skip("Skipping test for non-trtllm fp4 with use_128x4_sf_layout=False")
    if not use_nvfp4 and backend not in ["cudnn", "auto", "cute-dsl"]:
        pytest.skip("mx_fp4 is only supported for cudnn, cute-dsl, and auto backends")

    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_sf_layout = SfLayout.layout_128x4 if use_128x4_sf_layout else SfLayout.layout_8x4

    global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()

    # for trtllm, we need to shuffle mat2 because we swap A, B.
    do_shuffle_b = backend == "trtllm"

    block_size = 16 if use_nvfp4 else 32
    has_alpha = fp4_type == "mxfp4_alpha" or fp4_type == "nvfp4"

    if use_nvfp4:
        input_fp4, input_inv_s = nvfp4_quantize(
            input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
        )
        mat2_fp4, mat2_inv_s = nvfp4_quantize(
            mat2,
            global_sf_mat2,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=do_shuffle_b,
        )
    else:
        input_fp4, input_inv_s = mxfp4_quantize(input)
        mat2_fp4, mat2_inv_s = mxfp4_quantize(mat2)

    alpha = 1.0 / (global_sf_input * global_sf_mat2) if has_alpha else None

    reference = torch.mm(input, mat2.T)
    if activation == "relu2":
        reference = torch.relu(reference).square()

    res = torch.empty([m, n], device="cuda", dtype=res_dtype)

    try:
        with autotune(auto_tuning):
            mm_fp4(
                input_fp4,
                mat2_fp4.T,
                input_inv_s,
                mat2_inv_s.T,
                alpha,
                res_dtype,
                res,
                block_size=block_size,
                use_8x4_sf_layout=not use_128x4_sf_layout,
                backend=backend,
                use_nvfp4=use_nvfp4,
                skip_check=False,
                activation=activation,
            )

        cos_sim = F.cosine_similarity(
            reference.float().reshape(-1), res.float().reshape(-1), dim=0
        )
        assert cos_sim > 0.97
    except LibraryError as e:
        # TODO: Remove this check once cuDNN backend version is updated to 9.14.0
        if str(e) == CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR:
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))


# TODO: Consdier splitting this function up for the various backends
@pytest.mark.parametrize(
    "m",
    [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 17, 20, 24, 31, 32, 48, 64, 128, 256, 512],
)
@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["trtllm", "cudnn", "cutlass", "cute-dsl", "b12x"])
@pytest.mark.parametrize("use_128x4_sf_layout", [False, True])
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("fp4_type", ["nvfp4", "mxfp4", "mxfp4_alpha"])
def test_mm_fp4(
    m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
):
    # Non-auto backends
    _test_mm_fp4(
        m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
    )


# Split tests for checking auto functionality
@pytest.mark.parametrize("m", [1, 48, 256, 512])
@pytest.mark.parametrize("n", [256, 512])
@pytest.mark.parametrize("k", [256, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_128x4_sf_layout", [True])
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("fp4_type", ["nvfp4", "mxfp4", "mxfp4_alpha"])
def test_mm_fp4_backend_auto(
    m, n, k, res_dtype, use_128x4_sf_layout, auto_tuning, fp4_type
):
    # Some test cases for auto backend.
    _test_mm_fp4(m, n, k, res_dtype, "auto", use_128x4_sf_layout, auto_tuning, fp4_type)


# Regression (#3560): b12x must accept ragged K (real floor K%32==0, not tile_k=128).
# K=192 (packed_k=96) is the shape #3560 broke; both auto_tuning values hit distinct paths.
@pytest.mark.parametrize("k", [96, 192])
@pytest.mark.parametrize("auto_tuning", [False, True])
def test_mm_fp4_b12x_ragged_k(k, auto_tuning):
    _test_mm_fp4(
        m=64,
        n=512,
        k=k,
        res_dtype=torch.bfloat16,
        backend="b12x",
        use_128x4_sf_layout=True,
        auto_tuning=auto_tuning,
        fp4_type="nvfp4",
    )


# K % 32 != 0 violates TMA 16-byte alignment; explicit b12x must reject cleanly.
def test_mm_fp4_b12x_misaligned_k_raises():
    device = torch.device("cuda")
    if not (
        is_sm12x_supported(device) and version_at_least(torch.version.cuda, "13.0")
    ):
        pytest.skip("b12x backend requires SM120/SM121 + CUDA 13+.")
    m, n, k = 64, 512, 112  # k % 32 == 16
    _, _, a_fp4, a_s, b_fp4, b_s, alpha = _nvfp4_operands(m, n, k)
    res = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of 32"):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            alpha,
            torch.bfloat16,
            res,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="b12x",
            use_nvfp4=True,
            skip_check=False,
        )


def _nvfp4_operands(m, n, k):
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    g_in = (448 * 6) / a.float().abs().nan_to_num().max()
    g_w = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_s = nvfp4_quantize(
        a, g_in, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_s = nvfp4_quantize(
        b, g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    return a, b, a_fp4, a_s, b_fp4, b_s, 1.0 / (g_in * g_w)


def test_mm_fp4_b12x_short_k_multi_wave():
    # One K tile and more work tiles than SMs stress the epilogue smem
    # handoff between a persistent CTA's work tiles, a regime the
    # parametrized shapes never reach. Repeats, since a bad handoff shows
    # up as a timing-dependent mismatch.
    device = torch.device("cuda")
    if not (
        is_sm12x_supported(device) and version_at_least(torch.version.cuda, "13.0")
    ):
        pytest.skip("b12x backend requires SM120/SM121 + CUDA 13+.")
    m, n, k = 1024, 4096, 128
    for _ in range(3):
        a, b, a_fp4, a_s, b_fp4, b_s, alpha = _nvfp4_operands(m, n, k)
        res = mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            alpha,
            torch.bfloat16,
            None,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="b12x",
            use_nvfp4=True,
        )
        reference = torch.mm(a, b.T)
        cos_sim = F.cosine_similarity(
            reference.reshape(-1).float(), res.reshape(-1).float(), dim=0
        ).item()
        assert cos_sim > 0.97


def test_mm_fp4_cute_dsl_misaligned_n_raises():
    device = torch.device("cuda")
    if get_compute_capability(device)[0] != 10:
        pytest.skip("cute_dsl backend only supports SM100/SM103 GPUs.")
    m, n, k = 16, 130, 128  # n % 8 == 2
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    g_in = (448 * 6) / a.float().abs().nan_to_num().max()
    g_w = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_s = nvfp4_quantize(
        a, g_in, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_s = nvfp4_quantize(
        b, g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    res = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="N % 8 == 0"):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            1.0 / (g_in * g_w),
            torch.bfloat16,
            res,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cute-dsl",
            use_nvfp4=True,
            skip_check=False,
        )


def test_mm_fp4_trtllm_requires_alpha():
    _skip_if_trtllm_unsupported()
    device = torch.device("cuda")

    a = torch.empty((1, 64), dtype=torch.uint8, device=device)
    b = torch.empty((64, 64), dtype=torch.uint8, device=device)
    scale = torch.empty((1, 4), dtype=torch.uint8, device=device)

    with pytest.raises(ValueError, match="backend='trtllm' requires alpha"):
        mm_fp4(
            a,
            b,
            scale,
            scale,
            backend="trtllm",
            activation="relu2",
        )


@pytest.mark.parametrize("backend", ["auto", "cutlass"])
def test_mm_fp4_quantized_output_requires_trtllm(backend):
    device = torch.device("cuda")
    a = torch.empty((1, 64), dtype=torch.uint8, device=device)
    b = torch.empty((64, 64), dtype=torch.uint8, device=device)
    scale = torch.empty((1, 4), dtype=torch.uint8, device=device)

    with pytest.raises(
        ValueError, match="packed NVFP4 output requires explicit backend='trtllm'"
    ):
        mm_fp4(
            a,
            b,
            scale,
            scale,
            torch.ones(1, dtype=torch.float32, device=device),
            out_dtype=torch.uint8,
            backend=backend,
            activation="relu2",
            skip_check=True,
        )


def test_mm_fp4_trtllm_relu2():
    _test_mm_fp4(
        m=8,
        n=128,
        k=128,
        res_dtype=torch.bfloat16,
        backend="trtllm",
        use_128x4_sf_layout=True,
        auto_tuning=False,
        fp4_type="nvfp4",
        activation="relu2",
    )


def test_mm_fp4_trtllm_quantized_output_is_gemm_input():
    _skip_if_trtllm_unsupported()
    device = torch.device("cuda")
    torch.manual_seed(1235)
    m, intermediate_size, hidden_size = 8, 128, 128
    a_bf16 = torch.randn((m, hidden_size), dtype=torch.bfloat16, device=device) / (
        hidden_size**0.25
    )
    up_weight_bf16 = torch.randn(
        (intermediate_size, hidden_size), dtype=torch.bfloat16, device=device
    ) / (hidden_size**0.25)
    a, a_scale, a_global_scale = _quantize_nvfp4(a_bf16, shuffle=False)
    up_weight, up_weight_scale, up_weight_global_scale = _quantize_nvfp4(
        up_weight_bf16, shuffle=True
    )
    up_alpha = (a_global_scale * up_weight_global_scale).reciprocal()
    expected_up = torch.relu(a_bf16.float() @ up_weight_bf16.float().T).square()
    output_quant_scale = ((448 * 6) / expected_up.abs().amax().clamp_min(1e-6)).reshape(
        1
    )
    up_out = torch.empty((m, intermediate_size // 2), dtype=torch.uint8, device=device)
    up_out_scale = torch.empty(
        ((m + 127) // 128 * 128, (intermediate_size // 16 + 3) // 4 * 4),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    up_result = mm_fp4(
        a,
        up_weight.T,
        a_scale,
        up_weight_scale.T,
        up_alpha,
        out_dtype=torch.uint8,
        out=up_out,
        backend="trtllm",
        activation="relu2",
        output_quant_scale=output_quant_scale,
        out_scale=up_out_scale,
    )

    down_weight_bf16 = torch.randn(
        (hidden_size, intermediate_size), dtype=torch.bfloat16, device=device
    ) / (intermediate_size**0.25)
    down_weight, down_weight_scale, down_weight_global_scale = _quantize_nvfp4(
        down_weight_bf16, shuffle=True
    )
    down_alpha = (output_quant_scale * down_weight_global_scale).reciprocal()
    down_out = torch.empty((m, hidden_size), dtype=torch.bfloat16, device=device)
    down_result = mm_fp4(
        up_out,
        down_weight.T,
        up_out_scale,
        down_weight_scale.T,
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
    assert cosine > 0.97


def test_mm_fp4_trtllm_quantized_output_cuda_graph():
    _skip_if_trtllm_unsupported()
    device = torch.device("cuda")
    torch.manual_seed(1236)
    m, n, k = 8, 128, 128
    a_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device) / (k**0.25)
    b_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device) / (k**0.25)
    a, a_scale, a_global_scale = _quantize_nvfp4(a_bf16, shuffle=False)
    b, b_scale, b_global_scale = _quantize_nvfp4(b_bf16, shuffle=True)
    alpha = (a_global_scale * b_global_scale).reciprocal()
    expected = torch.relu(a_bf16.float() @ b_bf16.float().T).square()
    output_quant_scale = ((448 * 6) / expected.abs().amax().clamp_min(1e-6)).reshape(1)
    out = torch.empty((m, n // 2), dtype=torch.uint8, device=device)
    out_scale = torch.empty(
        ((m + 127) // 128 * 128, (n // 16 + 3) // 4 * 4),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    def run(destination, destination_scale):
        return mm_fp4(
            a,
            b.T,
            a_scale,
            b_scale.T,
            alpha,
            out_dtype=torch.uint8,
            out=destination,
            backend="trtllm",
            activation="relu2",
            output_quant_scale=output_quant_scale,
            out_scale=destination_scale,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(out, out_scale)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_result = run(out, out_scale)

    a_new_bf16 = torch.randn_like(a_bf16) / (k**0.25)
    a_new, a_scale_new = nvfp4_quantize(
        a_new_bf16,
        a_global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    a.copy_(a_new)
    a_scale.copy_(a_scale_new)
    out.zero_()
    out_scale.view(torch.uint8).zero_()
    graph.replay()
    torch.cuda.synchronize()

    eager_out = torch.zeros_like(out)
    eager_out_scale = torch.zeros_like(out_scale)
    eager_result = run(eager_out, eager_out_scale)
    torch.cuda.synchronize()

    assert graph_result.data_ptr() == out.data_ptr()
    assert eager_result.data_ptr() == eager_out.data_ptr()
    assert torch.equal(out, eager_out)
    assert torch.equal(out_scale.view(torch.uint8), eager_out_scale.view(torch.uint8))


def test_mm_fp4_trtllm_rejects_invalid_quantized_output_metadata():
    _skip_if_trtllm_unsupported()
    device = torch.device("cuda")
    torch.manual_seed(1237)
    m, n, k = 1, 128, 128
    a_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    a, a_scale, a_global_scale = _quantize_nvfp4(a_bf16, shuffle=False)
    a_r8, a_scale_r8, a_r8_global_scale = _quantize_nvfp4(
        a_bf16,
        shuffle=False,
        layout=SfLayout.layout_8x4,
    )
    b, b_scale, b_global_scale = _quantize_nvfp4(b_bf16, shuffle=True)
    alpha = (a_global_scale * b_global_scale).reciprocal()
    output_quant_scale = torch.ones(1, dtype=torch.float32, device=device)
    out = torch.empty((m, n // 2), dtype=torch.uint8, device=device)
    out_scale = torch.empty((128, n // 16), dtype=torch.float8_e4m3fn, device=device)

    with pytest.raises(ValueError, match="R128c4 input, weight, and output"):
        mm_fp4(
            a_r8,
            b.T,
            a_scale_r8,
            b_scale.T,
            (a_r8_global_scale * b_global_scale).reciprocal(),
            out_dtype=torch.uint8,
            backend="trtllm",
            use_8x4_sf_layout=True,
            activation="relu2",
            output_quant_scale=output_quant_scale,
            out_scale=out_scale,
        )

    with pytest.raises(ValueError, match="packed NVFP4 out must have shape"):
        mm_fp4(
            a,
            b.T,
            a_scale,
            b_scale.T,
            alpha,
            out_dtype=torch.uint8,
            out=torch.empty((m, n), dtype=torch.uint8, device=device),
            backend="trtllm",
            activation="relu2",
            output_quant_scale=output_quant_scale,
            out_scale=out_scale,
        )

    with pytest.raises(ValueError, match="packed NVFP4 out_scale must have shape"):
        mm_fp4(
            a,
            b.T,
            a_scale,
            b_scale.T,
            alpha,
            out_dtype=torch.uint8,
            out=out,
            backend="trtllm",
            activation="relu2",
            output_quant_scale=output_quant_scale,
            out_scale=torch.empty(
                (128, n // 16 + 4), dtype=torch.float8_e4m3fn, device=device
            ),
        )


if __name__ == "__main__":
    pytest.main([__file__])
