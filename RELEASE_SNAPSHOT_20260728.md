# S9 五算子发布快照（2026-07-28）

## 范围

本快照固化五个正式提交契约：

- `Concat`
- `Greater`
- `IndexAdd`
- `Transpose`
- `SquareSumV1`

`submission-src/` 中五题源码均来自各自最终云端实验版本。二进制
`.run` 和 ZIP 位于仓库外的 `D:\29722\Desktop\GCC\提交相关材料`，
不进入 Git。

## 验证摘要

| 算子 | 正确性覆盖 | 代表性 910B4 结果 |
| --- | --- | --- |
| Concat | 9 定向 + 100 随机，逐位相等 | 16 核在 8/16/32 核 A/B 中保留 |
| Greater | 246 组，覆盖广播、NaN/Inf/±0 和五种 dtype | 保留按 dtype/广播分类的快路 |
| IndexAdd | 23 定向 + 170 原随机 + 345 扩展 = 538 | public 约 9.108µs |
| Transpose | 48 定向 + 200 随机 + 152 扩展 = 400，逐位相等 | public FP16 128×256 约 4.44–4.60µs |
| SquareSumV1 | 46 定向 + 150 原随机 + 726 扩展 = 922 | public 6.490µs；rank-5 三稀疏轴 49.221µs |

SquareSumV1 的扩展矩阵覆盖三种 dtype、rank 1–5、负轴、多轴、
`keep_dims`、31/32/33、63/64/65、8191/8192/8193、10000、
NaN/Inf/±0 和 FP16 平方溢出。最终源码从不含 `build_out` 的独立目录
重新构建，安装生成的 `.run` 后再次通过全部 922 项。

## 最终提交包 SHA-256

| 文件 | SHA-256 |
| --- | --- |
| `Concat.zip` | `82e17ebf1f062c42f61d64f0788b7c2ed6a2633ff26a9abd44fae5b6da7fe814` |
| `Greater.zip` | `6c61f94e838843b052dd72e4cd5622cf309f5d3811370e15a6a15d2353df2539` |
| `IndexAdd.zip` | `d2b087093f5a7e3bc1559cadacd703f7bf3592fbd1a5a5dcf72280a2d2ac2a5f` |
| `Transpose.zip` | `0b88b825e7ee6e5cb598e5c3eb4638d4f652a4a0eb32671087a10a4aa3ff57e8` |
| `SquareSumV1.zip` | `af9ae474bdedb0b5a7b55d39fc0de9a3bd448df39a01b08926304513412e642b` |

## 最终安装包 SHA-256

| 算子 | `.run` SHA-256 |
| --- | --- |
| Concat | `60bdb960e7158bd4f259dc29ac9f95a904c813e3bc3edcfbd2c11dbb80a69095` |
| Greater | `ccc3e22b98dd6a39f5921ab100dd67d60933d12bc9492bd291fee92bd987af03` |
| IndexAdd | `75fda0d2a5f034610a9b3db50503750538ec427a46a168ea52e5364702644931` |
| Transpose | `1edad240181a0e73e1fe4ad607ea0457b12197cc38aa16218057873017287214` |
| SquareSumV1 | `6f06cf701976d2bf3f43bfd05bf82c13c070bf06be09355ab63e6fafc645b71d` |

## SquareSumV1 源码 SHA-256

| 文件 | SHA-256 |
| --- | --- |
| `op_host/square_sum_v1_tiling.h` | `4d74ab0ee3959d12c43a274b9c9261414a8ec6a088087642c44653307afa0bed` |
| `op_host/square_sum_v1.cpp` | `7608e5361505155cb2f922c92733090764a3698d0a272a4a8eb0019ec4447ed5` |
| `op_kernel/square_sum_v1.cpp` | `3794279e548dec2153e2cfdacb72b8b0ceaa999052dfdfc9d1123ea4f5e0491d` |

`SquareSumV1.zip` 只有顶层 `SquareSumV1_zip/`、两个源码目录和一个
可执行 `.run`；本地解包后的源码及 `.run` 哈希与独立构建完全一致。
旧包保存在
`D:\29722\Desktop\GCC\提交相关材料\旧包备份_20260728-1326`。

## 尚未闭环

- 五题仍需逐题上传赛事平台，以 Case1–Case5、`Zip_Check` 和榜单计时
  作为最终结论。
- SquareSumV1 的生成式 ACLNN 验证封装拒绝零长度 `IntArray`；
  Host 已实现空 axis 全维归约，但端到端注入仍需平台确认。
- 不把本地 profiler 时间表述为平台最终成绩。
