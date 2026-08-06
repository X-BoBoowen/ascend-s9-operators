# SquareSumV1-S02BN 云端验证与待测评交接

更新时间：2026-08-06（Asia/Shanghai）

## 1. 结论

S02BN 是 S02BM 的严格后继候选。它补齐 fastPath3 的另一类低并行度
布局：尾部连续归约段大于 16384、自然归约行较少、输出不超过 16 时，
旧实现无法使用 S02BI 的“按自然行 split-K”，通常只启动少量 AIV，
公开同结构矩阵耗时约为 149～364 us。

S02BN 将工作拆成“自然行 × 4096 元素尾块”，均匀分给最多 40 个 AIV，
每核写 FP32 partial workspace，再复用既有树归并。目标项紧邻 A/B 合计
由 `5191.856 us` 降到 `899.054 us`，加速 `5.775x`；6 个未改道控制项
合计比值为 `0.9993x`。候选在 Ascend 910B4、CANN 社区版 8.5.0 上
原生构建并通过累计 `1571/1571` 门禁。

正式平台五个 Case 尚未返回，因此不宣称达到榜单目标。正式基线仍为
S02BA 的 `3985.330 us`，目标为 `1195.599 us`。

## 2. 通用路由与实现

新路由同时满足：

```text
fastPath == 3
inputElements >= 262144
reduceElements >= 32768
1 <= outputElements <= 16
trailingReduceElements > 16384
尾部归约维物理连续
naturalRows * ceil(trailingReduceElements / 4096) >= 40
```

满足时使用 `reduceMode=4`、tiling key 4 和 40 核 FP32 workspace 树归并。
每个工作单元只归约一个自然行内的一段连续尾块，因此不跨越物理间隔；
`ReduceInputOffset` 仍负责通用 rank/axis 映射。输出 17、工作单元少于 40、
短尾和其他 fastPath 均保持 S02BM 路径。

这些条件只依赖 shape、axis、dtype 和物理布局，不依赖 Case 编号、输入值
或隐藏数据。候选源码：

```text
candidates/squaresum_s02bn_long_tail_splitk_20260806/SquareSumV1
```

实现提交：`af57972`；补充结构门禁提交：`8aba0d7`。

## 3. 真机紧邻 A/B

在同一云实例中先用已安装的 S02BM 跑完整矩阵，随后安装 S02BN 原样复跑：

```text
目标项 21 项：5191.856 -> 899.054 us，5.775x
控制项  6 项：1377.906 -> 1378.870 us，0.9993x
全部   27 项：6569.762 -> 2277.924 us，2.884x
最高单项加速：8.529x
```

代表性结果：

| 布局 | dtype | S02BM（us） | S02BN（us） | 加速 |
| --- | --- | ---: | ---: | ---: |
| rows=4, outputs=8, tail=65536 | FP16 | 267.690 | 41.106 | 6.512x |
| rows=4, outputs=8, tail=65536 | BF16 | 342.686 | 41.280 | 8.302x |
| rows=4, outputs=8, tail=65536 | FP32 | 265.082 | 42.836 | 6.188x |
| rows=4, outputs=9, tail=65536 | BF16 | 343.698 | 43.724 | 7.861x |
| rows=4, outputs=16, tail=65536 | BF16 | 344.038 | 40.982 | 8.395x |
| rows=4, outputs=8, tail=65539 | BF16 | 363.716 | 42.646 | 8.529x |
| rows=2, outputs=8, tail=131072 | BF16 | 343.568 | 42.390 | 8.105x |

输出 17 控制项约 `267～344 us`，改动前后小于 0.3%；工作单元不足控制项
约 `148～195 us`，改动前后小于 0.3%。这证明收益来自新路由而不是时段
波动或全局内核变化。

## 4. 正确性门禁

```text
既有基础门禁                    1295/1295
fastPath3 短尾 split-K 专项        39/39
非连续跨路径边界专项               36/36
S02BM workspace 阈值专项          102/102
S02BN 长尾性能/正确性矩阵           27/27
S02BN 结构正确性专项                72/72
累计                              1571/1571
```

结构专项覆盖自然行 1/2/3/4/8、输出 1/2/5/8/9/15/16、尾长边界与
非整除尾块、rank 3/4/5、多个尾部归约维、交错保留维、负轴、轴重排，
以及 `keep_dims=False/True`，三种 dtype 全部通过。

## 5. 构建、安装与提交包

云端实验：

```text
/home/ma-user/work/s9/experiments/squaresum_s02bn_cloud_20260806_1235
```

实际安装并完成门禁的 RUN SHA-256：

```text
F5E515F32453292EF2CFD3C636C0203DB045C046BCA01EA546D3BDAEAC9FA0E2
```

提交包：

```text
D:\29722\Desktop\GCC\提交相关材料\20260806\S02BN\SquareSumV1.zip
大小：672935 bytes
SHA-256：0595AB4EBE7CC9CB61D14E828D6D2DD5E599D3307341B906734BE54F3E9926BE
```

ZIP 只有一个顶层 `SquareSumV1_zip/`，包含五份 host/kernel 源文件和
一个 `.run`，共 6 个文件。五份源码逐文件哈希与 S02BN 候选一致，
`.run` 与云端实际安装版本一致，ZIP 中 `.run` 权限为 `0755`。

## 6. 结论边界与下一步

S02BN 证明长尾 fastPath3 的低并行度问题可以通用修复，但无法从公开
矩阵推断隐藏 Case 是否采用该布局。下一步只做一件事：上传本报告指定的
S02BN ZIP，记录官方 Case1–Case5。只有五项全 Pass 且稳定平台耗时显著
低于 S02BA，才把它晋级为新的正式基线。
