# SPDX-License-Identifier: Apache-2.0

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader, ReaderTensor, dequantize
from huggingface_hub import snapshot_download
from sgl_kernel import (
    ggml_dequantize,
    ggml_moe_a8,
    ggml_moe_a8_vec,
    ggml_moe_get_block_size,
    ggml_mul_mat_a8,
    ggml_mul_mat_vec_a8,
    moe_align_block_size,
)

GGUF_SAMPLE = snapshot_download("Isotr0py/test-gguf-sample")
GGUF_SAMPLE_MOE = snapshot_download("SzymonOzog/test-gguf-moe-sample")


def get_gguf_sample_tensors(
    hidden_size: int, quant_type: GGMLQuantizationType
) -> list[ReaderTensor]:
    sample_dir = GGUF_SAMPLE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


def get_gguf_MoE_tensors(
    hidden_size: int, quant_type: GGMLQuantizationType
) -> list[ReaderTensor]:
    sample_dir = GGUF_SAMPLE_MOE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


DTYPES = [torch.bfloat16]  # [torch.half, torch.bfloat16, torch.float32]
# Hidden_size for testing, must match the sample file in HF repo,
# we have `hidden_size = 256, 1024` for test in HF repo currently.
HIDDEN_SIZES = [256, 1024]
NUM_TOKENS = [7, 2050]  # Arbitrary values for testing
SEEDS = [0]
IQUANT_MMQ_TYPES = [
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ3_S,
]
IS_NVIDIA_CUDA = torch.version.cuda is not None and torch.version.hip is None
ACTIVE_IQUANT_MMQ_TYPES = IQUANT_MMQ_TYPES if IS_NVIDIA_CUDA else []
QUANT_TYPES = [
    # i-matrix
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    # k-quants
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    # standard quantization
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q8_0,
]

# Exercise both exact and ragged 4x32 CUDA tile boundaries, plus multiple
# quantization blocks along K. These cases stay tiny enough for any CUDA GPU.
IQUANT_DENSE_CASES = [
    (3, 31, 256),
    (4, 32, 256),
    (5, 33, 256),
    (33, 65, 256),
    (7, 33, 1024),
]


def _assert_quantized_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, atol=1.5, rtol=1e-1)


def _mmvq_rows(
    qweight: torch.Tensor,
    x: torch.Tensor,
    quant_type: GGMLQuantizationType,
    output_rows: int,
) -> torch.Tensor:
    """Use the established vector kernel as an independent row-wise oracle."""
    return torch.cat(
        [
            ggml_mul_mat_vec_a8(
                qweight,
                x[row : row + 1],
                quant_type,
                output_rows,
            ).to(x.dtype)
            for row in range(x.shape[0])
        ],
        dim=0,
    )


def _make_expert_spread_topk_ids(
    num_tokens: int, top_k: int, num_experts: int
) -> torch.Tensor:
    """Route across first, middle, and last experts to exercise expert strides."""
    assert num_experts >= 4
    selected_experts = torch.tensor(
        [0, 1, num_experts // 2, num_experts - 1],
        dtype=torch.int32,
        device="cuda",
    ).unique()
    flat_indices = torch.arange(num_tokens * top_k, device="cuda")
    return selected_experts[flat_indices % selected_experts.numel()].reshape(
        num_tokens, top_k
    )


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_dequantize(
    hidden_size: int, dtype: torch.dtype, quant_type: GGMLQuantizationType
):
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    for tensor in tensors:
        shape_str = tensor.name.split("_")[-1]
        shape = map(int, shape_str.split("x"))

        ref_output = torch.tensor(
            dequantize(tensor.data, quant_type), device="cuda"
        ).to(dtype)
        output = ggml_dequantize(
            torch.tensor(tensor.data, device="cuda"), quant_type, *list(shape), dtype
        )

        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=4e-2)


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_mmvq(hidden_size: int, dtype: torch.dtype, quant_type: GGMLQuantizationType):
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((1, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type), device="cuda").to(
            dtype
        )
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ggml_mul_mat_vec_a8(qweight, x, quant_type, qweight.shape[0]).to(dtype)

        # NOTE(FlamingoPg): There can be occasional errors, Loosen the granularity of gguf bf16 verification.
        atols = {torch.half: 1, torch.bfloat16: 1.5, torch.float: 1}
        rtols = {torch.half: 1e-1, torch.bfloat16: 3e1, torch.float: 1e-1}

        torch.testing.assert_close(
            output, ref_output, atol=atols[dtype], rtol=rtols[dtype]
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "quant_type",
    [
        # k-quants
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
        # standard quants
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q5_0,
        GGMLQuantizationType.Q8_0,
    ],
)
@torch.inference_mode()
def test_mmq(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
):
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type), device="cuda").to(
            dtype
        )
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ggml_mul_mat_a8(qweight, x, quant_type, qweight.shape[0])
        atols = {torch.half: 1, torch.bfloat16: 1.5, torch.float: 1.2}
        # test matrix has inputs centered around 0 and lower precision from
        # bfloat16 tends to accumulate and can greatly inflate rtol
        # since outputs are also very close to 0
        rtols = {torch.half: 1e-1, torch.bfloat16: 1e4, torch.float: 2e1}
        torch.testing.assert_close(
            output, ref_output, atol=atols[dtype], rtol=rtols[dtype]
        )


@pytest.mark.parametrize("quant_type", ACTIVE_IQUANT_MMQ_TYPES)
@pytest.mark.parametrize(
    "num_tokens,output_rows,hidden_size",
    IQUANT_DENSE_CASES,
)
@torch.inference_mode()
def test_iq_mmq_matches_independent_oracles(
    quant_type: GGMLQuantizationType,
    num_tokens: int,
    output_rows: int,
    hidden_size: int,
):
    torch.manual_seed(0)
    tensor = get_gguf_sample_tensors(hidden_size, quant_type)[0]
    assert tensor.data.shape[0] >= output_rows

    packed_weight = tensor.data[:output_rows]
    qweight = torch.tensor(packed_weight, device="cuda")
    weight = torch.tensor(
        dequantize(packed_weight, quant_type),
        dtype=torch.bfloat16,
        device="cuda",
    )
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    output = ggml_mul_mat_a8(qweight, x, quant_type, output_rows)
    mmvq_reference = _mmvq_rows(qweight, x, quant_type, output_rows)
    dequant_reference = x @ weight.T
    torch.cuda.synchronize()

    _assert_quantized_close(output, mmvq_reference)
    _assert_quantized_close(output, dequant_reference)


@pytest.mark.parametrize("quant_type", ACTIVE_IQUANT_MMQ_TYPES)
@torch.inference_mode()
def test_iq_mmq_zero_input(quant_type: GGMLQuantizationType):
    hidden_size, num_tokens, output_rows = 256, 5, 33
    tensor = get_gguf_sample_tensors(hidden_size, quant_type)[0]
    qweight = torch.tensor(tensor.data[:output_rows], device="cuda")
    x = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    output = ggml_mul_mat_a8(qweight, x, quant_type, output_rows)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


@pytest.mark.skipif(
    not IS_NVIDIA_CUDA, reason="I-quant MMQ is currently enabled on NVIDIA CUDA only"
)
@pytest.mark.parametrize("quant_type", IQUANT_MMQ_TYPES)
@pytest.mark.parametrize("output_rows", [32, 33])
@torch.inference_mode()
def test_iq_moe_mmq_matches_independent_oracles(
    quant_type: GGMLQuantizationType,
    output_rows: int,
):
    # More than 64 tokens exercises the same batched MoE path used by prefill.
    num_tokens, hidden_size, top_k = 65, 512, 2
    model_hidden_size = 1024
    torch.manual_seed(0)

    w13 = get_gguf_MoE_tensors(hidden_size, quant_type)[0]
    assert w13.data.shape[1] >= output_rows
    packed_weight = w13.data[:, :output_rows]
    qweight = torch.tensor(packed_weight, device="cuda")
    weight = torch.tensor(
        dequantize(packed_weight, quant_type),
        dtype=torch.bfloat16,
        device="cuda",
    )
    num_experts = qweight.shape[0]

    x = torch.randn(
        (num_tokens, model_hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    topk_ids = _make_expert_spread_topk_ids(num_tokens, top_k, num_experts)

    block_size = ggml_moe_get_block_size(quant_type)
    assert block_size > 0
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size, num_experts
    )
    output = ggml_moe_a8(
        x,
        qweight,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        output_rows,
        top_k,
        num_tokens,
    ).reshape(num_tokens, top_k, output_rows)

    mmvq_reference = ggml_moe_a8_vec(
        x,
        qweight,
        topk_ids,
        top_k,
        quant_type,
        output_rows,
        num_tokens,
    ).reshape(num_tokens, top_k, output_rows)

    dequant_reference = torch.empty_like(output)
    for expert in torch.unique(topk_ids).tolist():
        token_indices, slots = torch.where(topk_ids == expert)
        dequant_reference[token_indices, slots] = x[token_indices] @ weight[expert].T
    torch.cuda.synchronize()

    _assert_quantized_close(output, mmvq_reference)
    _assert_quantized_close(output, dequant_reference)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
