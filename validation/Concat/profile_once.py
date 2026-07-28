import torch
import torch_npu

import concat_validation_lib


torch.npu.config.allow_internal_format = False
values = [
    torch.randn((128, size), dtype=torch.float16).npu()
    for size in (27, 40, 63, 24, 50, 26, 19, 2, 5)
]
for _ in range(10):
    concat_validation_lib.concat(values, -1)
torch.npu.synchronize()
for _ in range(5):
    concat_validation_lib.concat(values, -1)
torch.npu.synchronize()
