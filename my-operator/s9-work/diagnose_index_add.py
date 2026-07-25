import numpy as np
import torch
import torch_npu

import custom_ops_lib


np.random.seed(20260723)
input_cpu = torch.from_numpy(
    np.random.uniform(-50, 50, [32, 128]).astype(np.int8)
)
index_cpu = torch.from_numpy(
    np.random.randint(low=0, high=32, size=120).astype(np.int32)
)
source_cpu = torch.from_numpy(
    np.random.uniform(-10, 10, [120, 128]).astype(np.int8)
)

golden = torch.index_add(input_cpu, 0, index_cpu, source_cpu)
input_npu = input_cpu.npu()
index_npu = index_cpu.npu()
source_npu = source_cpu.npu()

builtin = torch.index_add(input_npu, 0, index_npu, source_npu).cpu()
custom = custom_ops_lib.custom_op(input_npu, index_npu, source_npu, 0).cpu()


def report(name, actual):
    mismatch = actual != golden
    positions = mismatch.nonzero()[:12]
    print(f"{name}: mismatch={int(mismatch.sum())}/{golden.numel()}")
    for row, col in positions.tolist():
        print(
            f"  [{row}, {col}] actual={int(actual[row, col])} "
            f"golden={int(golden[row, col])}"
        )


print(f"duplicate_index_count={index_cpu.numel() - index_cpu.unique().numel()}")
report("torch_npu", builtin)
report("custom_op", custom)
