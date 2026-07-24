import numpy as np
import torch
import torch_npu

import custom_ops_lib


torch.npu.config.allow_internal_format = False


def run_op(x, dims):
    shape = list(torch.permute(x, dims).shape)
    return custom_ops_lib.custom_op(x.npu(), dims, shape).cpu()


def check_equal(name, x, dims):
    expected = torch.permute(x, dims).cpu()
    actual = run_op(x, dims)
    if not torch.equal(actual, expected):
        raise AssertionError(f"{name}: result mismatch")
    print(f"PASS {name}")


def check_fp16_bits(name, bits):
    x_np = bits.view(np.float16).reshape(128, 256)
    actual = run_op(torch.from_numpy(x_np), (1, 0)).numpy()
    expected = x_np.transpose(1, 0)
    if not np.array_equal(actual.view(np.uint16), expected.view(np.uint16)):
        mismatch = np.flatnonzero(
            actual.view(np.uint16).ravel() != expected.view(np.uint16).ravel()
        )
        raise AssertionError(f"{name}: bit mismatch at {mismatch[:10].tolist()}")
    print(f"PASS {name}")


rng = np.random.default_rng(20260723)
check_fp16_bits(
    "random_raw_fp16_bits",
    rng.integers(0, 65536, size=128 * 256, dtype=np.uint16),
)

special = np.array(
    [
        0x0000,
        0x8000,
        0x0001,
        0x03FF,
        0x0400,
        0x7BFF,
        0x7C00,
        0xFC00,
        0x7E00,
        0x7D01,
        0xFE00,
        0xFD01,
    ],
    dtype=np.uint16,
)
check_fp16_bits(
    "special_fp16_bits",
    np.resize(special, 128 * 256),
)

coords = torch.arange(128 * 256, dtype=torch.int32).reshape(128, 256)
check_equal("coordinate_order_int32_fallback", coords, (1, 0))
check_equal("fp32_fallback", torch.randn(17, 31, dtype=torch.float32), (1, 0))
check_equal("int32_fallback", torch.arange(77, dtype=torch.int32).reshape(7, 11), (1, 0))
check_equal("three_dimensional_fallback", torch.randn(3, 5, 7), (2, 0, 1))
check_equal("identity_fallback", torch.randn(128, 256, dtype=torch.float16), (0, 1))

base_npu = torch.randn(256, 128, dtype=torch.float16).npu()
check_equal("noncontiguous_fallback", base_npu.transpose(0, 1), (1, 0))

offset_base_npu = torch.randn(129, 256, dtype=torch.float16).npu()
offset_view = offset_base_npu[1:]
assert offset_view.is_contiguous() and offset_view.storage_offset() != 0
check_equal("storage_offset_fallback", offset_view, (1, 0))

print("ALL TRANSPOSE VALIDATIONS PASSED")
