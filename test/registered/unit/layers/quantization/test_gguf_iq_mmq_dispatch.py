"""Dispatch contracts for GGUF I-quant MMQ support."""

import unittest
from unittest import mock

import torch
from gguf import GGMLQuantizationType as WeightType

from sglang.srt.layers.quantization import gguf
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=3, stage="base-a", runner_config="1-gpu-small")

_SUPPORTED_IQUANT_MMQ_TYPES = {
    WeightType.IQ2_XS,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
}


class TestGGUFIQuantMMQDispatch(CustomTestCase):
    def test_platform_gate_matches_cuda_backend(self):
        expected = _SUPPORTED_IQUANT_MMQ_TYPES if gguf._is_cuda else set()

        self.assertEqual(gguf.IQUANT_MMQ_QUANT_TYPES, expected)
        self.assertTrue(gguf.IQUANT_MMQ_QUANT_TYPES <= gguf.MMQ_QUANT_TYPES)

    @unittest.skipUnless(gguf._is_cuda, "I-quant MMQ is NVIDIA CUDA-only")
    def test_large_supported_iq_batch_uses_mmq(self):
        x = torch.empty((17, 256), dtype=torch.float32)
        qweight = torch.empty((32, 64), dtype=torch.uint8)
        expected = torch.empty((17, 32), dtype=torch.float32)

        with (
            mock.patch.object(gguf, "ggml_mul_mat_a8", return_value=expected) as mmq,
            mock.patch.object(gguf, "ggml_mul_mat_vec_a8") as mmvq,
            mock.patch.object(gguf, "dequantize_gguf_weight") as dequantize,
        ):
            output = gguf.fused_mul_mat_gguf(x, qweight, WeightType.IQ2_XS)

        self.assertIs(output, expected)
        mmq.assert_called_once_with(qweight, x, WeightType.IQ2_XS, qweight.shape[0])
        mmvq.assert_not_called()
        dequantize.assert_not_called()

    @unittest.skipUnless(gguf._is_cuda, "I-quant MMQ is NVIDIA CUDA-only")
    def test_small_supported_iq_batch_keeps_mmvq(self):
        x = torch.empty((16, 256), dtype=torch.float32)
        qweight = torch.empty((32, 64), dtype=torch.uint8)
        expected = torch.empty((16, 32), dtype=torch.float32)

        with (
            mock.patch.object(
                gguf, "ggml_mul_mat_vec_a8", return_value=expected
            ) as mmvq,
            mock.patch.object(gguf, "ggml_mul_mat_a8") as mmq,
            mock.patch.object(gguf, "dequantize_gguf_weight") as dequantize,
        ):
            output = gguf.fused_mul_mat_gguf(x, qweight, WeightType.IQ2_XS)

        self.assertIs(output, expected)
        mmvq.assert_called_once_with(qweight, x, WeightType.IQ2_XS, qweight.shape[0])
        mmq.assert_not_called()
        dequantize.assert_not_called()

    @unittest.skipUnless(gguf._is_cuda, "I-quant MMQ is NVIDIA CUDA-only")
    def test_unimplemented_iq_format_keeps_dequantize_fallback(self):
        x = torch.ones((17, 256), dtype=torch.float32)
        qweight = torch.empty((32, 64), dtype=torch.uint8)
        dequantized = torch.ones((32, 256), dtype=torch.float32)
        expected = x @ dequantized.T

        with (
            mock.patch.object(gguf, "ggml_mul_mat_a8") as mmq,
            mock.patch.object(gguf, "ggml_mul_mat_vec_a8") as mmvq,
            mock.patch.object(
                gguf, "dequantize_gguf_weight", return_value=dequantized
            ) as dequantize,
        ):
            output = gguf.fused_mul_mat_gguf(x, qweight, WeightType.IQ1_S)

        torch.testing.assert_close(output, expected)
        dequantize.assert_called_once_with(qweight, WeightType.IQ1_S, x.dtype)
        mmq.assert_not_called()
        mmvq.assert_not_called()


if __name__ == "__main__":
    unittest.main()
