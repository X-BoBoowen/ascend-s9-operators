import torch
import torch_npu

import index_add_validation_lib


torch.npu.config.allow_internal_format = False
torch.manual_seed(20260727)

def run(name, self_tensor, index, source):
    expected = torch.index_add(self_tensor, 0, index, source)
    actual = index_add_validation_lib.index_add(
        self_tensor.npu(),
        0,
        index.npu(),
        source.npu(),
    ).cpu()
    mismatch = actual != expected
    positions = torch.nonzero(mismatch, as_tuple=False)
    print(f"{name}: MISMATCH_COUNT={positions.shape[0]}")
    for position in positions[:64]:
        row = int(position[0])
        column = int(position[1])
        print(
            f"row={row} column={column} "
            f"self={int(self_tensor[row, column])} "
            f"expected={int(expected[row, column])} "
            f"actual={int(actual[row, column])}"
        )


run(
    "aligned_single",
    torch.randint(-128, 128, (1, 256), dtype=torch.int8),
    torch.tensor([0], dtype=torch.int32),
    torch.randint(-128, 128, (1, 256), dtype=torch.int8),
)
run(
    "tail_single",
    torch.randint(-128, 128, (1, 37), dtype=torch.int8),
    torch.tensor([0], dtype=torch.int32),
    torch.randint(-128, 128, (1, 37), dtype=torch.int8),
)
run(
    "tail_repeated",
    torch.randint(-128, 128, (7, 37), dtype=torch.int8),
    torch.tensor([4, 0, 4, 2, 0, 4], dtype=torch.int32),
    torch.randint(-128, 128, (6, 37), dtype=torch.int8),
)
