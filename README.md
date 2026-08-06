# 昇腾 AI 创新大赛 S9 算子优化

本仓库记录 Ascend 910B、CANN 社区版 8.5.0 环境下五道 S9
算子的通用实现、真机验证、性能实验和正式提交源码。

当前工作原则：

- 先满足题面全部 dtype、rank、轴、广播、空输入和非对齐约束；
- 不根据隐藏样例猜 shape，不写公开 Case 专用分支；
- 每题在独立实验目录中构建、安装、回归和打包；
- 五道题分别按各自 `prof_sum` 排名并换算为 100、90、…、10、0 分，
  队伍赛季总分相加的是五题名次积分，不是五个 `prof_sum`；每题单项奖
  也独立评定；
- 只有平台 Case1–Case5 才能作为最终正确性和性能结论；
- Git 仓库保存源码、测试与文档，不保存 `.run`、ZIP、wheel 和
  profiler 数据库。

> 状态时间：2026-08-06（Asia/Shanghai）
>
> 当前阶段：SquareSumV1-S02BN 已完成 910B4/CANN 8.5.0 构建、安装、
> 紧邻 A/B、累计 `1571/1571` 门禁和提交包审计，等待官方评测。它在
> S02BM 基础上把 fastPath3 的长连续尾部拆成“自然行 × 4096 元素尾块”
> 分配给 40 个 AIV；21 项目标矩阵合计由 `5191.856 μs` 降至
> `899.054 μs`（`5.775x`），6 个控制项合计保持 `0.9993x`。
> S02BK 官方 `3984.7995 μs` 只比基线快 `0.5305 μs（0.013%）`，判定
> 未晋级。正式优化基线仍为 S02BA 的
> `3985.330 μs`，目标为 `≤1195.599 μs`。证据见
> [`SQUARESUM_S02BN_CLOUD_REPORT_20260806.md`](./SQUARESUM_S02BN_CLOUD_REPORT_20260806.md)。

下一轮完整执行方案见
[`OPTIMIZATION_PLAN_20260730.md`](./OPTIMIZATION_PLAN_20260730.md)。
该方案已按“每题独立排名并换算积分”重新起草，不再使用五题
`prof_sum` 合计安排开发顺序。

## 1. 当前状态

| 题目 | 工程状态 | 真机回归 | 当前提交包 | 平台状态 |
| --- | --- | ---: | --- | --- |
| Concat | 本轮完成 | 9 定向 + 100 随机 + 168 扩展 | 已生成并审计 | 5/5 Pass，688.6425 |
| Greater | 本轮完成 | 26 定向 + 220 随机 + 9 个 int32 扩展；扩展集连续复跑 3 次 | 已生成并审计 | 5/5 Pass，18595.372 |
| IndexAdd | 本轮完成 | 23 定向 + 170 随机 + 345 扩展 = 538 | 已生成并审计 | 5/5 Pass，119427.8755 |
| Transpose | 本轮完成 | 48 定向 + 200 随机 + 152 扩展 + 84 边界 = 484 | 已生成并审计 | 5/5 Pass，16303.227 |
| SquareSumV1 | S02BN 待官方评测 | S02BN 累计 1571/1571 | S02BN 已独立打包审计 | S02BN 待测；S02BK 3984.7995 未晋级，正式基线仍为 S02BA 3985.330 |

这里的“本轮完成”表示：

1. 正式算子名称、Host、Kernel、Tiling 和 CMake 契约一致；
2. 在 910B4、CANN 8.5.0 环境重新构建出 `.run`；
3. 安装该 `.run` 后，用自定义扩展直接调用正式算子；
4. 完成定向、随机、扩展和大规模压力回归；
5. ZIP 结构、文件权限、源码哈希和 `.run` 哈希已审计；
6. 本地 ZIP 与云端发布 ZIP 完全一致。

它不表示已经获得新的榜单成绩。五题的新包必须逐题上传，平台
Case1–Case5 全 Pass 后才能确认隐藏用例闭环。

## 2. 本轮五题的提交清单

正式 ZIP 位于本地仓库外：

```text
D:\29722\Desktop\GCC\提交相关材料\
|-- Concat.zip
|-- Greater.zip
|-- IndexAdd.zip
|-- Transpose.zip
`-- SquareSumV1.zip
```

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `Concat.zip` | 410639 B | `b43e88f82ba5230f9172b1f1f2f4c07381d4f14cd1fb5a4de3cf3871c55d25b2` |
| `Greater.zip` | 398168 B | `316797810d06b57d18c898d1fe449c1b0b51565c6a842845dded75524f7f868d` |
| `IndexAdd.zip` | 417920 B | `cf8c08a60d3b356686a07e136766dd06128afa87dccf67d9c9940330843d990b` |
| `Transpose.zip` | 401378 B | `52850235386690dcd89bf0fff039a4e510345a725aa2616d5f2a7c4d4fb5f1d0` |
| `SquareSumV1.zip` | 473756 B | `eecd9f1fd6b4c0617b6ec2ec632f24bb0f310d5a9d2f2125f03f6ec86ecfaf5b` |

包内 `.run`：

| 题目 | `.run` SHA-256 |
| --- | --- |
| Concat | `9dca80b07e85eacf9b90ac98349f64bf0ba0db19565c440f0c16f12cbf1ca873` |
| Greater | `d8ab6f81aa1eaa775cbe69078db09901037ccad667216697371475a42dbb4298` |
| IndexAdd | `58a5be8bbbc8db62708973fea21bd6630e1d938ae9c8a4ad28132f3014ce679d` |
| Transpose | `2baf9ecb7e38716b5d4a7ca8e89c890be2392c4b8c9352d8ca67d8d23ca9a9af` |
| SquareSumV1 | `aaa8096582170bab688607e40651e8c1d8b9b86e7f889f61948c8d1ce7320563` |

五个 ZIP 均只有一个顶层 `<Operator>_zip/`，包含 `op_host/`、
`op_kernel/` 和一个非空 `custom_opp_euleros_aarch64.run`。
`IndexAdd` 的 `.run` 在 ZIP 中保持 `0750`，其余四个为 `0755`。

## 3. 仓库结构

```text
.
|-- submission-src/
|   |-- Concat/             # 本轮正式源码
|   |-- Greater/            # 本轮正式源码
|   |-- IndexAdd/           # 本轮正式源码
|   |-- Transpose/          # 上一轮稳定源码
|   `-- SquareSumV1/        # 本轮正式源码
|-- validation/
|   |-- Concat/             # 定向、随机、profile 与扩展封装
|   |-- Greater/
|   |-- IndexAdd/
|   |-- Transpose/
|   `-- SquareSumV1/
|-- case_910b/              # 官方公开 case 与运行脚本
|-- operator-descriptors/   # 开发期工程描述
|-- EXPERT_OPINION_SYNTHESIS_20260727.md
|-- EXPERT_REVIEW_TRACKER.md
|-- RELEASE_SNAPSHOT_20260728_FIRST_THREE.md
`-- README.md
```

正式源码入口始终是 `submission-src/<Operator>/`。包内源码已与这里
逐文件计算 SHA-256，五题全部匹配。

## 4. Concat

### 4.1 题面覆盖

- dtype：`float32`、`float16`、`int32`、`int8`；
- 动态输入列表；
- rank 1–8；
- 正轴和负轴；
- 任意合法拼接轴；
- 首个或中间输入为空；
- 非 32B 对齐；
- 当前 Tiling 容量最多 256 个输入。

### 4.2 实现

Host 将任意拼接轴折叠为：

```text
[outer, input_dim * inner] -> [outer, sum(input_dim) * inner]
```

以字节为单位传递每个输入的行宽，避免不同 dtype 下重复维护元素
对齐逻辑。Kernel 通过 `ListTensorDesc` 读取动态输入地址。

当前有两类工作划分：

- `outer` 足够时按行分核；
- `outer` 很小而总字节数很大时按全局 32B 工作块分核，避免超宽单行
  只占用一个核。

大规模多行搬运使用 `DataCopyPad` 二维搬运；行跨度超过硬件 stride
上限时自动回退逐行搬运。Host 根据总字节量和实际 Vector Core 数量
动态选择核数，小任务避免过度起核，大任务可超过旧版固定 16 核。

### 4.3 验证

- 9 个定向用例；
- 100 个随机用例；
- 168 个扩展用例；
- 全部与 `torch.cat` 逐位相等。

覆盖内容包括 33/256 输入、rank 1–8、dim 0/中间/末轴、负轴、
首个空输入、多个空输入、非对齐宽度、宽行分块和大 outer。

本地事件计时只用于版本 A/B，不等价于平台成绩。代表性 p50：

| 场景 | p50 |
| --- | ---: |
| public 非对齐 | 110.09 us |
| 对齐多行 | 107.22 us |
| dim0 大输入 | 101.76 us |
| outer=128 大输入 | 101.04 us |
| 256 输入 | 484.60 us |

## 5. Greater

### 5.1 题面覆盖

- dtype：`float32`、`bfloat16`、`float16`、`int32`、`int8`；
- 输出严格为 bool；
- rank 0–8；
- NumPy/PyTorch 广播；
- 同形、标量、末维广播、交错广播；
- NaN、Inf、`+0/-0`、相等值；
- 空输出和非 32B 对齐。

### 5.2 实现

Host 计算广播后的输出 shape，并为两个输入生成 stride。被广播的维度
stride 为 0。Kernel 识别并优化：

- 连续同形输入；
- 标量输入；
- 可连续搬运的重复 run；
- 通用坐标映射广播。

计算按 4096 元素分块，最多使用 40 个 Vector Core。浮点路径使用
原生 `Compare(GT)`，因此 NaN 参与比较为 false。int8 先无损转 half
再比较。int32 不转 float，避免 24 位有效精度造成全位宽整数误判；
最终版本通过 `max(self, other) == self` 与 `self == other` 两个掩码
组合出严格大于。

### 5.3 int32 修复记录

早期候选曾尝试用高/低 16 位掩码组合比较，在随机全 32 位 bit pattern
中出现 621 个错误。该候选未进入正式包。

最终 int32 路径在以下扩展集全部通过，并连续复跑 3 次：

- 边界值笛卡尔积；
- 257×1021 随机全位宽 bit pattern；
- 200003 个全相等元素；
- 左/右标量极值；
- trailing、interleaved 和 rank-8 广播；
- 空输出。

### 5.4 验证

- 26 个定向用例；
- 220 个随机用例；
- 9 个 int32 扩展用例；
- 扩展集额外连续复跑 3 次；
- 共 255 个不同用例全部通过。

本地扩展调用事件计时约 23–25 us，仅用于回归和 A/B。

## 6. IndexAdd

### 6.1 题面覆盖

- self/source dtype：`float32`、`bfloat16`、`float16`、`int32`、`int8`；
- index dtype：`int32`；
- rank 1–8；
- 任意正/负 dim；
- 重复、乱序和空 index；
- 未命中输出保持 self；
- int8/int32 溢出回绕；
- 题面 index 上限 8000；
- 非 32B 对齐 inner。

### 6.2 实现

Host 将张量折叠为：

```text
outer = product(shape[:dim])
dimSize = shape[dim]
inner = product(shape[dim + 1:])
```

Kernel 自身从 self 构造完整输出，不依赖 Python wrapper 预复制。每个
任务负责一组互不重叠的输出行，因此重复 index 不存在跨核写冲突。

核心优化：

- `inner` 按 256 元素分块；
- `dim` 每组最多 64 行；
- 最多 40 核；
- 行宽与地址满足条件时使用二维 `DataCopyPad` 批量搬入/搬出；
- int8 使用分批累加路径，保持逐位回绕语义；
- float16/bfloat16 在需要时使用 float 累加缓冲；
- index 在核内缓存；
- 当 index 较大且 inner 有多个分块时，为一个 dim 组预构建命中列表，
  在多个 inner chunk 间复用；
- 只有浮点 dimGroups≥8、整数 dimGroups≥16 时才启用命中复用，
  该门限来自通用压力矩阵 A/B，不绑定某个隐藏 shape；
- 连续任务分配减少大 shape 下的重复扫描与负载不均衡。

当前使用 6 个 TilingKey，组合区分普通/批量 int8、命中复用和单
inner chunk 快路。

### 6.3 验证

- 23 个定向用例；
- 170 个随机用例；
- 3 个 seed 共 345 个扩展用例；
- 合计 538 个用例全部通过；
- 另有大 index、大 dim、大 inner、outer>1 的压力和性能矩阵。

代表性本地事件计时：

| 场景 | 中位数 |
| --- | ---: |
| FP16, M=8000, inner=4096 | 721.37 us |
| FP32, outer=2, M=8000, inner=2048 | 845.98 us |
| BF16, outer=2, M=8000, inner=2048 | 756.24 us |
| FP32, M=8000, inner=4096 | 819.30 us |
| int8, dim=4096, M=1024, inner=2048 | 129.68 us |

一个“int8 批量 source 队列 + 无条件命中列表”候选曾在随机 case18
出现 41 个错误，已经丢弃，未进入正式源码和提交包。

## 7. Transpose

### 7.1 题面覆盖

- dtype：`float32`、`float16`、`int32`、`int8`；
- rank 1–6；
- 任意合法 `dims`，支持正轴和负轴；
- 单位维、空输入和非 32B 对齐；
- `N`、`N2` 至 10000，`N3`–`N5` 至 1000；
- 结果与 `torch.permute(...).contiguous()` 逐位一致。

### 7.2 实现

Host 先合并保持不变的前缀和可连续搬运的尾部，再根据排列结构选择
5 个 TilingKey：

- 通用坐标映射、identity 和连续 run 搬运；
- 单矩阵、16×16 对齐的 fp16 硬件 Transpose；
- 循环置换折叠后的二维分块转置；
- 输出末维可聚合时的 gather 快路；
- 多 batch、16×16 对齐的 fp16 硬件 Transpose。

通用路径不依赖隐藏 shape；矩阵行列、batch 和 stride 均来自 Host
Tiling。循环置换折叠后的二维分块根据 dtype 使用三种向量数据流：

- FP16 的 16×16 输入块先由 `DataCopyPad` 补齐，完整块和边界块均调用
  硬件 `Transpose`，只回写有效行列；
- FP32/INT32 的 8×8 块预生成 8 组列掩码，一次 `GatherMask` 完成
  64 个元素的转置；
- INT8 的 32×32 块将相邻字节视为 `uint16` 列对，`GatherMask` 抽取
  32 行的同一列对，再用移位与向量转换拆成两条输出行。

不能安全折叠的任意排列继续走通用坐标映射或 gather 回退，不依赖
隐藏 shape，也没有 Case 专用分支。

### 7.3 验证、性能与拒绝候选

- 48 个定向用例；
- 200 个固定 seed 随机用例；
- 152 个循环置换、任意排列和特殊 bit pattern 扩展用例；
- 84 个阈值、行列分块和 3–5 维双向旋转边界用例；
- 合计 484 个用例，在最终候选和正式目录无缓存构建上各完整通过一次；
- 三个优化阶段均在独立实验目录中构建、安装和回归，正式发布包再次
  完整回归 484 例。

相同扩展调用、相同输入的 NPU Event 中位数：

| 场景 | 优化前 | 正式版本 |
| --- | ---: | ---: |
| FP16, 1024×1024 | 28.934 us | 29.111 us |
| FP16, 1000×1000 尾块 | 176.429 us | 51.082 us |
| FP32, 1024×1024 | 677.332 us | 188.231 us |
| INT32, 1024×1024 | 677.395 us | 188.276 us |
| INT8, 1024×1024 | 479.078 us | 88.978 us |
| FP32, (32,256,512) 末两维交换 | 2745.314 us | 785.701 us |
| INT8, (32,256,512) 末两维交换 | 1909.640 us | 349.423 us |

设备侧 `msprof` 的同形状内核中位数进一步确认提升来自 Kernel：

| 场景 | 优化前 Kernel | 正式候选 Kernel |
| --- | ---: | ---: |
| FP16, 1024×1024 | 28.411 us | 28.451 us |
| FP16, 1000×1000 尾块 | 176.154 us | 50.321 us |
| FP32, 1024×1024 | 676.153 us | 185.894 us |
| INT8, 1024×1024 | 478.500 us | 88.492 us |

曾实现 CANN 8.5 增强 Transpose API 的 UB 大块候选。它虽然通过同一
484 例正确性回归，但设备级 profiling 显示明显回退：

| 场景 | 稳定版平均 kernel | 增强候选平均 kernel |
| --- | ---: | ---: |
| float32, 128×256 | 25.756 us | 243.971 us |
| int8, 128×256 | 19.536 us | 337.596 us |
| fp16, 127×257 尾块 | 20.913 us | 480.164 us |

因此增强候选没有进入正式源码、Git 提交或发布 ZIP。

## 8. SquareSumV1

### 8.1 题面覆盖

- dtype：`float16`、`bfloat16`、`float32`；
- rank 1–5；
- 单轴、多轴、负轴和任意合法轴组合；
- `keep_dims=true/false`；
- `N`、`N2` 至 10000，`N3` 至 1000，`N4` 至 200；
- 非 32B 对齐、单位维和标量输出；
- 语义严格对应
  `torch.sum(torch.square(x), dim=axis, keepdim=keep_dims)`。

### 8.2 实现

Host 将轴集合规范化，并识别连续末轴、连续前缀/中间轴和通用稀疏轴。
Kernel 保留通用坐标映射回退，同时为常见布局提供连续向量归约、分组
归约和按行树形归约。

大输入、小输出是旧版的主要瓶颈。当前根据数据类型和输出规模选择三种
通用模式：

- 普通模式：按输出分核，适合输出数充足或归约较短的场景；
- FP32 原子模式：当输出不超过 8 且归约足够长时，40 核分别计算局部
  和，再原子累加输出；
- 工作区模式：FP16/BF16 的长连续归约，以及输出不超过 1024 的长
  前缀/中间归约，40 核写入 FP32 局部和，跨核同步后由 0 号核收尾。

工作区大小同时包含 CANN 系统工作区与用户局部和，Kernel 通过
`GetUserWorkspace` 获取正确偏移，避免早期实验中出现的 DDR 越界。
归约长度超过 8192 时切换到更大的 UB 配置；两个 TilingKey 均有
8192/8193 边界回归。

S02F 为 fastPath3 增加第 5 个 TilingKey。仅当分组尾部归约满足
32B 对齐、尾长不超过 64、最内层保留输出连续且可按 8 个输出成组、
二维搬运跨度合法时，Kernel 才会：

- 一次二维 DMA 搬入 8 个相邻输出的分组数据；
- 用 `WholeReduceSum<float>` 批量得到 8 组局部和；
- 在 UB 中按 batch 行做二叉树归约，最后一次写回 8 个输出。

尾长 64 的上限来自 CANN 8.5.0 `WholeReduceSum<float>` 的单 repeat
256B/64-float 硬件约束。尾长 1024 曾在候选边界测试中暴露错误，
随后 Host 门控被收紧；该形状现已验证正确回退旧路径。其他不满足
条件的形状也全部保留 S02E2 路径。

数值语义方面：

- FP16 先在 FP16 中平方，再转 FP32 累加，保留 FP16 平方溢出；
- BF16 先转 FP32 完成乘法，再以 `CAST_RINT` 回写 BF16，随后转
  FP32 累加，等价于先执行 BF16 `torch.square`；
- FP32 全程使用 FP32；原子模式只用于 FP32，避免半精度原子写入
  的硬件兼容问题。

### 8.3 正确性验证

- 46 个定向用例；
- 150 个固定 seed 随机用例；
- 726 个边界、布局和大维度扩展用例；
- 922 个基础、随机和扩展用例；
- 10 个工作区模式探针；
- 10 个 TilingKey/回退边界探针；
- 4 个 FP32 原子稳定性探针；
- 4 个 BF16 严格逐位语义探针；
- 16 个 S02F 专项，覆盖 batch 31/32/33、40 核任务不均分、
  `keep_dims`、负轴、三种 dtype、尾长 64 上限和尾长 1024 回退；
- S02F 候选累计 `966/966 Pass`；正式源码独立重建后再次通过
  16 个专项和 7 组性能矩阵的正确性检查。

BF16 专项包含曾经能暴露错误的
`63 × 0.10009765625`，修复前输出 `0.6328125`，PyTorch 期望
`0.62890625`；正式版本逐位一致。

### 8.4 性能

以下为相同扩展调用、相同输入的 NPU Event 中位数，仅用于版本 A/B：

| 场景 | 初始版本 | 正式版本 | 提升 |
| --- | ---: | ---: | ---: |
| FP16, shape=(10000,64), axis=0 | 218.1 us | 27.524 us | 7.9× |
| FP32, shape=(10000,64), axis=0 | 202.7 us | 27.142 us | 7.5× |
| FP16, shape=(200,1000,64), axis=(0,1) | 4321.5 us | 79.189 us | 54.6× |
| FP32, shape=(200,1000,64), axis=(0,1) | 4015.3 us | 47.647 us | 84.3× |
| FP16, shape=(200,1000,64), 全轴 | 1109.0 us | 38.273 us | 29.0× |
| FP32, shape=(200,1000,64), 全轴 | 1049.0 us | 35.246 us | 29.8× |

S02E2 与 S02F 在相邻时间窗口重新安装各自正式 `.run` 后的 fastPath3
同形状 A/B（NPU Event 中位数）：

| 场景 | S02E2 | S02F | 改善 |
| --- | ---: | ---: | ---: |
| small aligned FP16 | 43.614 us | 27.567 us | 36.79% |
| medium aligned FP16 | 92.608 us | 68.257 us | 26.30% |
| large aligned FP16 | 98.771 us | 77.377 us | 21.66% |
| wide output FP16 | 187.409 us | 125.721 us | 32.92% |
| medium aligned BF16 | 100.449 us | 78.185 us | 22.16% |
| medium aligned FP32 | 88.014 us | 67.717 us | 23.06% |
| 未命中尾长 33 FP16 | 101.707 us | 101.552 us | 0.15% 波动 |

前 6 个命中形状合计由 610.865 降至 444.824 us，改善 27.18%。

设备侧 profiling 的稳定区间中位数：

| 场景 | SquareSumV1 kernel |
| --- | ---: |
| 公开样例 `(123,31)` FP16 | 6.440 us |
| 大前缀归约 BF16 | 76.952 us |
| 大全轴归约 BF16 | 44.551 us |
| 大全轴归约 FP32 | 32.251 us |

平台隐藏形状未知，因此这些数据只证明同形状优化有效，不作为榜单成绩。

## 9. 源码哈希

### 9.1 Concat

| 文件 | SHA-256 |
| --- | --- |
| `op_host/CMakeLists.txt` | `d1b100115b8c34135ccdfc54f91597847a7823ec76cdca995e2b80f5c6092cd2` |
| `op_host/concat.cpp` | `cd98d736052c05987bb3ebc9fcb32e0a43694e8ed35961cfba45772e384741ce` |
| `op_host/concat_tiling.h` | `5b07ef0615f269c0ea951977916152748bc16fef7d652c276619d287620efb9a` |
| `op_kernel/CMakeLists.txt` | `10b4df9e22540a42e443602357cf8a7bfa71b4c9c7198fa7da6b3f4343b00118` |
| `op_kernel/concat.cpp` | `6ff66e8845a1dd60f148536bd3770677a1627d8a9d9d89a9371b8e0ab86353d8` |

### 9.2 Greater

| 文件 | SHA-256 |
| --- | --- |
| `op_host/CMakeLists.txt` | `d1b100115b8c34135ccdfc54f91597847a7823ec76cdca995e2b80f5c6092cd2` |
| `op_host/greater.cpp` | `1f1b69f6d65128d1c7746a38af2ceb9df501f2f69c95eefa486ae27d60b62a52` |
| `op_host/greater_tiling.h` | `964b0256aef614b62e285afbffd6c967507276aac52493cd0047121b68f93c6b` |
| `op_kernel/CMakeLists.txt` | `de2557844234c8ca7d5952ec124d4c53f0196d9066d5e172f86f62808fb776a6` |
| `op_kernel/greater.cpp` | `0f2aaab2f2a0d804281a177bed09d915bd329d55b402c5fa6a596aef8380c5e0` |

### 9.3 IndexAdd

| 文件 | SHA-256 |
| --- | --- |
| `op_host/CMakeLists.txt` | `d1b100115b8c34135ccdfc54f91597847a7823ec76cdca995e2b80f5c6092cd2` |
| `op_host/index_add.cpp` | `33e4eb4c27f633fcc8031a1ef4e353ff1e98f86ece32d666ff0e424f20831476` |
| `op_host/index_add_tiling.h` | `85d66f140d24855b97d281a8b6f715b81ead115fab7e2aa976a57614616ccb6b` |
| `op_kernel/CMakeLists.txt` | `fb5e4b00af77bb885d29dff3faa1f1c094e79f19f6a366eca0d381a209819627` |
| `op_kernel/index_add.cpp` | `e9ee4fdd157ef6a92ae843eef49eb8179a558885695dc0bdcbfcece82bca7aa8` |

### 9.4 Transpose

| 文件 | SHA-256 |
| --- | --- |
| `op_host/CMakeLists.txt` | `d1b100115b8c34135ccdfc54f91597847a7823ec76cdca995e2b80f5c6092cd2` |
| `op_host/transpose.cpp` | `d68ec597fa68861a04f5da11f17b481674500567328fef57500a387786fdc260` |
| `op_host/transpose_tiling.h` | `71d17bd19c58ada5ee29e6a1b3640e2f3c97ab8451cb9de6611d9a8befd4d5e1` |
| `op_kernel/CMakeLists.txt` | `dc5e6d36cbd092eed6fdc008a40896ede683299a3affeb91d693343bd6f29597` |
| `op_kernel/transpose.cpp` | `c59c71398d16431b7dd2d6b428dcc7969aae6f01d30ed0819df183f61278b581` |

### 9.5 SquareSumV1

| 文件 | SHA-256 |
| --- | --- |
| `op_host/CMakeLists.txt` | `0dd1acfd256e9666299b0b634c4f16362aa61702395fb68a504a12ddec123f23` |
| `op_host/square_sum_v1.cpp` | `6c4731323e66d3e7a02044fa214386d7c10167a0ba145c6aa8d3902535f5f367` |
| `op_host/square_sum_v1_tiling.h` | `e669820ab05878a8701fe2e6f33f4b44076ee93e74c1a308c3b5e8fcf54079f6` |
| `op_kernel/CMakeLists.txt` | `62c979b878b6c91819cb9df2d8d49544a2d1f41be5a7031de6088688f2f0afd6` |
| `op_kernel/square_sum_v1.cpp` | `b668b6a2217130e1878063713b3d03f663c7626b7c175fb6c8e503119002255a` |

## 10. 云端状态

云环境：

```text
ModelArts: my_env2
NPU: Ascend 910B4
CANN: /home/ma-user/Ascend/cann-8.5.0
工作根目录: /home/ma-user/work/s9
```

最终实验：

```text
/home/ma-user/work/s9/experiments/concat_dynamic_rows_20260728_1417
/home/ma-user/work/s9/experiments/greater_int32_masks_20260728_1439
/home/ma-user/work/s9/experiments/indexadd_hitreuse_20260728_1514
/home/ma-user/work/s9/experiments/transpose_release_20260729_2027
/home/ma-user/work/s9/experiments/squaresum_workspace_best_20260729_1854
/home/ma-user/work/s9/experiments/squaresum_s02f_grouped_vector8_20260730_1603
/home/ma-user/work/s9/release/squaresum_s02f_20260730_1629
```

云端发布包：

```text
/home/ma-user/work/s9/releases/concat_20260728_1446/Concat.zip
/home/ma-user/work/s9/releases/greater_20260728_1511/Greater.zip
/home/ma-user/work/s9/releases/indexadd_20260728_1622/IndexAdd.zip
/home/ma-user/work/s9/releases/transpose_20260729_2032/Transpose.zip
/home/ma-user/work/s9/release/squaresum_s02f_20260730_1629/SquareSumV1.zip
```

SquareSumV1 的正式回归证据：

```text
/home/ma-user/work/s9/validation/regression_922_formal_final_20260729
/home/ma-user/work/s9/experiments/squaresum_workspace_best_20260729_1854/validation/profile_final_*
/home/ma-user/work/s9/experiments/squaresum_s02f_grouped_vector8_20260730_1603/*.log
/home/ma-user/work/s9/release/squaresum_s02f_20260730_1629/formal_*.log
```

Transpose 的正式回归和性能证据：

```text
/home/ma-user/work/s9/experiments/transpose_release_20260729_2027/release_*.log
/home/ma-user/work/s9/experiments/transpose_release_20260729_2027/release_benchmark.log
/home/ma-user/work/s9/profiles/transpose_true_baseline_20260729
/home/ma-user/work/s9/profiles/transpose_vector_final_20260729
```

Transpose 的正式源码、独立构建源码和包内源码逐文件哈希一致；包内
`.run` 与正式构建产物 SHA-256 一致。旧提交包已备份为：

```text
提交相关材料/历史版本/Transpose_pre_vectorized_20260729-2033.zip
```

截至 2026-07-29 19:34，SquareSumV1 的正式源码、独立构建源码、
包内源码逐文件哈希一致；包内 `.run` 与云端正式构建产物 SHA-256
一致。旧提交包已备份为：

```text
提交相关材料/历史版本/SquareSumV1_pre_workspace_parallel_20260729-1933.zip
```

所有云端命令通过本地 `ssh_visible.ps1` 执行，并记录在：

```text
/home/ma-user/work/s9/codex-visible-terminal.log
```

每次通过 `ssh_visible.ps1` 启动新的服务器运行时，还会先清空仓库根目录
的 `artifact/`，再将本次运行结果写入：

```text
artifact/result.json
artifact/run.log
```

`result.json` 使用 UTF-8 和缩进格式，记录远端命令、开始/结束时间、
运行时长、退出码和成功状态；`run.log` 保存本次运行的完整终端输出。
两个正式文件都先写入同目录临时文件，再以替换方式发布，避免中途
中断留下半写文件。`artifact/` 只代表最新一次运行，不保留历史记录。

## 11. 平台正式结果与解释

2026-07-30 已收到 `提交相关材料/20260729/` 中五个正式 ZIP 的平台结果。
五个 ZIP 的 SHA-256 与本 README 第 2 节登记值完全一致，Case 求和也逐题
等于平台 `prof_sum`：

| 题目 | Case1 | Case2 | Case3 | Case4 | Case5 | 总和 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Concat | 14.13 | 38.9005 | 19.07 | 105.102 | 511.44 | 688.6425 |
| Greater | 5.24 | 16026.332 | 2262.108 | 92.344 | 209.348 | 18595.372 |
| IndexAdd | 9.03 | 37491.429 | 88.1215 | 2568.781 | 79270.514 | 119427.8755 |
| Transpose | 4.62 | 51.161 | 2467.94 | 2195.774 | 11583.732 | 16303.227 |
| SquareSumV1 | 7.72 | 404.592 | 241.66 | 1720.188 | 908.376 | 3282.536 |

五题正确性为 `25/25 Pass`。赛事评分必须逐题处理：

- Concat 只与 Concat 榜单比较；
- Greater 只与 Greater 榜单比较；
- IndexAdd 只与 IndexAdd 榜单比较；
- Transpose 只与 Transpose 榜单比较；
- SquareSumV1 只与 SquareSumV1 榜单比较。

不同题目的绝对耗时量级没有横向评分意义。第一阶段争取进入积分区只需
比较各题当前值与第 10 名门槛；进入前十后，再补第 3、第 1 名目标线。
跨题调度应比较一次可实现优化预计跨过的名次和新增积分，而不是比较绝对
耗时。

2026-07-30 已取得五题第 10 名门槛：

| 题目 | 当前 `prof_sum` | 第10名 | 当前/门槛 | 至少需下降 |
| --- | ---: | ---: | ---: | ---: |
| SquareSumV1 | 3985.330（S02BI 同包最低观测 3982.789） | 1268.598 | 3.14× | 68.17% |
| Concat | 688.6425 | 225.096 | 3.06× | 67.31% |
| Transpose | 16303.227 | 1063.036 | 15.34× | 93.48% |
| Greater | 18595.372 | 572.743 | 32.47× | 96.92% |
| IndexAdd | 119427.8755 | 841.584 | 141.91× | 99.30% |

五题当前均在第 10 名以后，第一阶段目标是至少让一道题进入前十。按门槛
距离和当前源码证据，第一争分题为 SquareSumV1，第二为 Concat；后三题
都需要多路径或算法级重构。

Concat 这次结果与此前 `693.2035` 来自完全相同的 ZIP，本次低
`4.561`（`0.66%`）。由于没有代码变化且只有一次复测，只登记为同包
最低平台观测，不把它编号成 Concat-C02，也不宣称新增代码收益。

## 12. 下一次工作如何接续

### 12.1 已收平台反馈

五题正式 ZIP 的首轮完整平台结果已经登记在第 11 节。正式源码、ZIP
及这组结果继续冻结为可回退基线；后续每个候选都必须使用新迭代编号，
不得覆盖 `submission-src/` 后丢失当前可用版本。

### 12.2 复核本地工作区

```bash
git status --short
git log -1 --oneline
git diff --check
```

正式源码只从 `submission-src/` 取。不要从 `candidates/`、
`current-submission-extracted-*` 或历史实验目录覆盖正式版本。

### 12.3 云端执行纪律

显式加载 CANN 8.5.0：

```bash
source /home/ma-user/Ascend/cann-8.5.0/bin/setenv.bash
source /home/ma-user/Ascend/cann-8.5.0/opp/vendors/customize/bin/set_env.bash
```

登录欢迎信息可能仍显示镜像预置的旧 CANN 字样，不能据此判断实际构建
版本。构建、安装和运行前都要核对环境变量指向 8.5.0。

每轮实验：

1. 从最终实验复制到带时间戳的新目录；
2. 保存旧源码哈希；
3. 只改一个可检验假设；
4. 重建 `.run` 并安装；
5. 先定向，再随机，再扩展，再压力；
6. 做同形状 A/B；
7. 只有正确且有稳定收益才提升为正式源码；
8. 独立无缓存重建后再打包。

### 12.4 后续工作

当前顺序：

1. `SquareSumV1-S02A` 的 atomic 越界修复已完成，并保留在后续正式
   版本中：atomic 目标为对齐 workspace，最终只回写真实输出字节。
2. `SquareSumV1-S02E2` 官方结果为 `3259.5255 us`，相比 S01 的
   `3282.536 us` 只改善 `23.0105 us`（0.70%）；Case4 只改善
   `1.844 us`，证明 fastPath4 没有命中主要瓶颈。
3. `SquareSumV1-S02F` 已完成 fastPath3 连续 8 输出批处理、966 例
   回归、相邻窗口 A/B、正式源码独立重建和打包，当前等待平台评测。
4. 平台结果返回前不得宣称 S02F 的隐藏 Case 或 `prof_sum` 已改善；
   后续继续诊断 `ReduceContiguous` 和未覆盖的 grouped-suffix 长尾，
   目标仍为 SquareSumV1 `prof_sum < 1268.598`。
5. 第二争分题为 Concat；若其他 Case 不变，Case5 必须从 `511.44`
   压到 `47.8935` 以下，题内目标 `< 225.096`。
6. 第三题暂定 Transpose；Greater 与 IndexAdd 先分别完成一次结构诊断，
   再按实测收益决定后续顺序。

隐藏 shape 不可见，不得反推或硬编码；每个候选必须在独立实验目录完成
正确性与同形状 A/B，未经验证不得覆盖对应的 `submission-src/<Op>`。

### 12.5 SquareSumV1-S02E2 历史正式候选

2026-07-30 曾将 `SquareSumV1-S02E2` 晋级为正式候选。该版本包含：

- float atomic 输出改为安全 workspace，消除对真实输出的对齐扩大写；
- 大 workspace 归约使用独立 TilingKey 和 2D DMA + UB 树形 finalizer；
- fastPath4 的 strided-inner 行归约改为“真实行一次搬入、补零到下一个
  2 的幂、UB 二叉树归约”，避免顺序 Add 链和非 2 次幂的额外 DMA。

候选在云端 CANN 8.5.0 上通过：

```text
46 directed
4 bf16 semantic
4 atomic
10 workspace
10 tiling-key
150 random
726 extended
合计 950/950 Pass
```

fastPath4 独立 A/B（NPU Event 中位数，单位 μs）：

| dtype | S02H 对照 | S02E2 | 改善 |
| --- | ---: | ---: | ---: |
| fp16 | 127.246 | 82.364 | 35.27% |
| bf16 | 133.959 | 85.148 | 36.44% |
| fp32 | 123.962 | 77.155 | 37.76% |

连续末轴、中间轴、fastPath3、workspace 大归约控制矩阵未发现稳定回退。
小于约 2% 或随运行频率变化的差异均未记作代码收益。

正式构建与包：

```text
云端 release:
/home/ma-user/work/s9/release/squaresum_s02e2_20260730_1532

.run SHA-256:
C19BA2DBE2E81E0E3D923F0E8A4F5D7EDAF79B6E84CD1C803A1EA857A2B20FFF

历史 ZIP:
提交相关材料/20260730/SquareSumV1.zip
提交相关材料/历史版本/SquareSumV1_S02E2_before_S02F_20260730-1635.zip

ZIP SHA-256:
14B3384ED0A5A740808E008BE0D7922BEA9D5A27CD837CE8CFCC0DBD3D196EE9

上一正式包备份:
提交相关材料/历史版本/SquareSumV1_S01_before_S02E2_20260730-1536.zip

上一正式包 SHA-256:
D5D0407C7F81519DCE36682D18104D1579BF96642EA898E0706C91550942DBB7
```

上述提升只证明公开覆盖矩阵中的 fastPath4 通用路径更快。S02E2 的
平台结果为 3259.5255 us，相比 S01 只改善 0.70%，因此该路径没有
命中主要隐藏瓶颈；这不能用于反推隐藏 shape、dtype 或 TilingKey。

### 12.6 SquareSumV1-S02F 当前正式候选

S02F 只优化源码能够直接确认的 fastPath3 结构：旧路径对每个输出逐个
执行二维 DMA、平方、归约和标量同步；新路径一次处理 8 个连续输出。
Host 使用严格门控，不满足对齐、连续性、尾长和 stride 条件的形状继续
运行 S02E2。

候选测试：

```text
46 directed
4 bf16 semantic
4 atomic
10 workspace
10 tiling-key
150 random
726 extended
16 grouped-vector8 专项
合计 966/966 Pass
```

正式 A/B 中，6 个命中形状逐项改善 21.66%～36.79%，合计改善
27.18%；未命中尾长 33 对照只有 0.15% 波动。尾长 1024 的早期边界
错误由专项测试捕获；依据 CANN 8.5.0 的 256B vector repeat 约束，
新路径上限已收紧为 64，1024 现正确回退。

正式构建与包：

```text
云端 release:
/home/ma-user/work/s9/release/squaresum_s02f_20260730_1629

.run SHA-256:
AAA8096582170BAB688607E40651E8C1D8B9B86E7F889F61948C8D1CE7320563

当前 ZIP:
提交相关材料/SquareSumV1.zip
提交相关材料/20260730/S02F_1635/SquareSumV1.zip

ZIP SHA-256:
EECD9F1FD6B4C0617B6EC2EC632F24BB0F310D5A9D2F2125F03F6EC86ECFAF5B

上一正式包备份:
提交相关材料/历史版本/SquareSumV1_S02E2_before_S02F_20260730-1635.zip

上一正式包 SHA-256:
14B3384ED0A5A740808E008BE0D7922BEA9D5A27CD837CE8CFCC0DBD3D196EE9
```

S02F 的本地 A/B 只能证明该通用路径更快；隐藏 Case 的 shape、dtype 和
TilingKey 仍未知，最终结论必须等待平台 Case1～Case5。

### 12.7 SquareSumV1-S02AY 阶段测评候选

2026-08-05，S02AY 在 S02F 之后继续完成了分段归约、长向量、多输出
grouped、跨步树归约和 workspace 中间归约的通用优化及边界修复。完整
证据、路径、哈希和回传格式见：

```text
SQUARESUM_S02AY_CLOUD_REPORT_20260805.md
```

S02AY 云端独立验证汇总：726 扩展、46 定向、150 随机、4 个 BF16
严格语义、45 个关键路径、6 个 workspace 专项、9 个 grouped 对齐
窄宽专项全部通过。33 个代表性性能基准多数为 24～33 μs；三重多轴、
跨步树归约和大前缀归约约为 65～85 μs，仍是下一阶段优化重点。

阶段测评包：

```text
D:\29722\Desktop\GCC\提交相关材料\20260805\S02AY\SquareSumV1.zip
ZIP SHA-256:
C5A9F237124374D97C78DAD97966833AD6A5783EA9B6C8D67CD53ECDEBE162AC

.run SHA-256:
11066D7CCC6339B904FC052F3C21E90B856E3448838564A7B6E0771B1E18D3F5
```

该包尚未取得官方隐藏 Case 结果，因此仍不得宣称进入前十，也不覆盖
冻结的正式基线。收到 Case1～Case5 后按每题独立排名继续优化。

### 12.8 SquareSumV1-S02BA 单例间隔阶段测评候选

2026-08-05，S02BA 修正了长度为 1 的非归约维度对物理连续组的错误
分类。末轴等价形状直接使用已有连续末轴实现；中间归约只在
`reduceElements >= 8192` 时晋级已有连续中间轴实现，小尺寸继续走
S02AY 路径。Kernel、非单例间隔路径和算子语义均未修改。

三种 dtype 的大单例间隔矩阵从约 `1.07～1.94 ms` 降到
`53.704～87.460 μs`，加速 `17.44～25.17×`；多个单例间隔的全归约
从 `140.560～160.535 μs` 降到 `27.478～28.150 μs`。交叉矩阵确认
8192 及以上稳定获益，2048 及以下不晋级。

云端独立通过 46 定向、150 随机、4 BF16 严格语义、726 扩展、45
关键路径、72 单例语义、15 单例性能形状和 30 交叉点测试。详细形状、
适用边界、A/B 与哈希见：

```text
SQUARESUM_S02BA_CLOUD_REPORT_20260805.md
```

阶段测评包：

```text
D:\29722\Desktop\GCC\提交相关材料\20260805\S02BA\SquareSumV1.zip
ZIP SHA-256:
23007E299E83D1116C783235C924C9CFDE729B4FD15BA78A474E5AAF8DC3114B

.run SHA-256:
71F9895392C989317FF6F1CA22391FE4D6CB6948F06955BDFA43D9C79208E194
```

S02BA 官方结果为 `6.51 / 399.088 / 138.753 / 2550.761 / 890.218 μs`，
`prof_sum=3985.33 μs`。相对 S02E2，Case3 改善 42.07%，但 Case4
恶化 48.44%，总耗时恶化 22.27%，因此 S02BA 不晋级正式基线。

### 12.9 SquareSumV1-S02BB 宽度 8 紧凑搬运候选

S02BB 只优化 workspace 中间连续归约的完整内层宽度 8：FP16/BF16
将 `rows × 8` 作为一个连续块搬入 UB，消除逐行 16B 搬运与 32B 补齐。
FP32、其他宽度、Host Tiling 和非 workspace 路径均不变。

公开同形状 A/B 中，超长归约由 `273.120～1270.352 μs` 降至
`37.602～51.456 μs`，加速 7.26～24.69 倍；两个外层输出组加速约
13 倍。46 定向、150 随机、4 BF16 语义、726 扩展、45 关键路径、
60 宽度 8 专项、72 单例语义、30 交叉和 15 单例性能矩阵全部通过。

详细报告与阶段包：

```text
SQUARESUM_S02BB_CLOUD_REPORT_20260805.md

D:\29722\Desktop\GCC\提交相关材料\20260805\S02BB\SquareSumV1.zip
ZIP SHA-256:
DC7451016F341784E3083653DD9BCCBF32B0345A3F9FAA462E76162682289F7F

.run SHA-256:
50AAF086075564114EA2B3AB015ECC51F17AE6A13DE120C590FCD4CDF0D1A072
```

S02BB 官方结果为 `7.88 / 405.58 / 140.8 / 2551.56 / 902.756 μs`，
`prof_sum=4008.576 μs`。相对 S02BA，五个 Case 分别变慢 `21.04%`、
`1.63%`、`1.48%`、`0.03%`、`1.41%`，总耗时变慢 `23.246 μs`
（`0.58%`）。公开宽度 8 A/B 的收益没有转化为隐藏评测收益，因此
S02BB 不晋级，正式对比基线仍为 S02BA；该结果也不能证明任何隐藏
Case 的 shape、dtype 或执行路径。

### 12.10 SquareSumV1-S02BD 尾部单例连续路由候选

S02BD 将“连续归约组后仅剩长度为 1 的保留维”从中间轴路径改道到物理
等价的连续末轴路径。Kernel 与 S02BB 字节一致，只有 Host 增加 5 行通用
布局判断。无 NPU 的本地地址审计已覆盖 72,168 个 rank/shape/axis/
keepDims 组合，其中 1,644 个实际改道；所有改道布局的扁平输出顺序、
归约连续区间和完整输入覆盖均严格一致，698 个零归约布局保持原路由。

完整本地证据与未来一次性云端 A/B 门禁见：

```text
SQUARESUM_S02BD_LOCAL_AUDIT_20260805.md
```

S02BD 已在 CANN 8.5.0 / 910B4 完成 66/66 A/B：60 个目标布局全部
正确，中位加速 28.17 倍，6 个宽度 2/4 控制项最大绝对波动 1.46%。
定向测试同时发现历史实现的空归约未置零问题，因此 S02BD 不直接提交，
其路由优化由 S02BF 继承。

### 12.11 SquareSumV1-S02BF 尾部单例与空归约候选

S02BF 在 S02BD 通用路由上修复空归约：`reduceElements == 0` 使用独立
tiling key 13 和编译期特化置零，非空 tiling key 不含额外运行时分支。
它在 910B4/CANN 8.5.0 上独立通过 1295/1295 门禁和 66/66 性能矩阵。
相对 S02BB，60 个目标布局的中位加速为 29.62 倍；相对 S02BD 的中位
差为 -0.19%，表明非空热路径已恢复。

```text
SQUARESUM_S02BF_CLOUD_REPORT_20260805.md

D:\29722\Desktop\GCC\提交相关材料\20260805\S02BF\SquareSumV1.zip
大小：594583 bytes
ZIP SHA-256：
B5484C12895CFF3165D865A3E557FCE321221BD5C09FB66DE1C4916A6B35196B
RUN SHA-256：
A6F489BAA532308882A355F59B864E9621B5386E12830D18F79D52EBCCF284EF
```

S02BF 官方结果为 `7.908 / 407.248 / 139.776 / 2541.768 /
904.500 μs`，`prof_sum=4001.200 μs`。相对 S02BA 合计变慢
`15.870 μs`（`+0.398%`），因此不晋级。

### 12.12 SquareSumV1-S02BG FP32 宽度 8 紧凑搬运候选

S02BG 是 S02BF 的严格后继，只删除 `compactFullInner8` 对 FP32 的排除。
FP32 每行 8 个元素正好 32B；静态审计确认所有 DMA 长度和树归约地址
均满足 32B 对齐。相邻安装 A/B 的 7 个目标布局全部正确，最小/中位/
最大加速为 `1.005x / 3.700x / 11.960x`；宽度 4/16 控制项最大绝对
波动 `1.149%`。独立完整门禁为 `1295/1295`。

```text
SQUARESUM_S02BG_CLOUD_REPORT_20260805.md

D:\29722\Desktop\GCC\提交相关材料\20260805\S02BG\SquareSumV1.zip
大小：594087 bytes
ZIP SHA-256：
D3BC2703620C076EC35A173C558926E9D76868E3F05D1BE12BCA090BFA6D0247
RUN SHA-256：
58B03731E7E7D1EF83DB913CE9E31D71BB91BC14571F4EBBD2F64661C0DAFE40
```

S02BG 官方结果为 `6.500 / 399.998 / 138.4125 / 2553.771 /
896.918 μs`，`prof_sum=3995.5995 μs`。相对 S02BA 合计变慢
`10.2695 μs`（`+0.258%`），相对 S02BF 只改善 `5.6005 μs`；因此
S02BG 也不晋级，S02BA 继续作为正式官方基线。

### 12.13 SquareSumV1-S02BI fastPath3 split-K 候选

S02BI 只在 fastPath3、输出不超过 8、尾部连续归约宽度
`1024～16384`、自然分组行不少于 16 且总规模达到 workspace 门槛时，
把归约行分给 40 个 AIV 并复用既有 key 4 的 FP32 树归约。fastPath4、
尾宽 256、自然行 8、输出 9 和阈值下方均不进入新路由。

同机相邻 A/B 的 33 个目标记录合计从 `8899.536` 降到
`1373.074 μs`，约 `6.481x`；单项最小/中位/最大加速为
`1.202x / 4.269x / 19.984x`。既有全量和新专项合计 `1370/1370`
正确。待测 ZIP：

```text
D:\29722\Desktop\GCC\提交相关材料\20260805\S02BI\SquareSumV1.zip
ZIP SHA-256：BB3ECCDFBA6584EBAC4D7D3F5B86C34E5579C935BF775E8C5CA0BC9046B2A565
RUN SHA-256：BBBCD140B73DB45BD89C2E7B6D142FFC5345A917E47FCD8A25BBD8BD2145654E
```

完整门槛、失败入口证据、A/B 和包审计见
[`SQUARESUM_S02BI_CLOUD_REPORT_20260805.md`](./SQUARESUM_S02BI_CLOUD_REPORT_20260805.md)。

官方结果为 `6.470 / 390.5975 / 139.463 / 2552.781 /
893.4775 μs`，`prof_sum=3982.789 μs`。相对 S02BA 的五项变化依次为
`-0.040 / -8.4905 / +0.710 / +2.020 / +3.2595 μs`，合计只改善
`2.541 μs（0.064%）`。Case4/5 反而退化，且总差异远小于平台复测波动，
因此 S02BI 不晋级；这也证明 fastPath3 split-K 没有覆盖隐藏主耗时布局。

### 12.14 SquareSumV1-S02BN fastPath3 长尾 split-K 候选

S02BN 是 S02BM 的严格后继：当 fastPath3 的尾部连续归约段大于 16384、
输出不超过 16 且“自然行 × 4096 元素尾块”至少能形成 40 个工作单元时，
将这些工作单元分配给 40 个 AIV，再复用 FP32 workspace 树归并。
输出 17、工作单元不足、短尾以及其他 fastPath 保持原实现。

真机紧邻 A/B 中，21 个目标项合计从 `5191.856` 降到 `899.054 μs`
（`5.775x`），最高单项 `8.529x`；6 个未改道控制项合计比值为
`0.9993x`。既有、阈值、长尾和补充结构门禁累计 `1571/1571`。

```text
SQUARESUM_S02BN_CLOUD_REPORT_20260806.md

D:\29722\Desktop\GCC\提交相关材料\20260806\S02BN\SquareSumV1.zip
大小：672935 bytes
ZIP SHA-256：
0595AB4EBE7CC9CB61D14E828D6D2DD5E599D3307341B906734BE54F3E9926BE
RUN SHA-256：
F5E515F32453292EF2CFD3C636C0203DB045C046BCA01EA546D3BDAEAC9FA0E2
```

S02BN 尚无官方结果，不能据公开 A/B 宣称命中隐藏主耗时或达到榜单目标。

## 13. 安全与仓库卫生

禁止提交：

- SSH 私钥、云账号、令牌和 `.env`；
- `.run`、`.so`、wheel、目标文件；
- ZIP、profiler 数据库和构建缓存；
- 个人材料、代金券记录和联系方式。

提交前检查：

```bash
git diff --check
git ls-files | grep -Ei '\.(pem|key|run|so|whl|zip)$'
```

## 14. 结论边界

当前能够确认的是：S02BF、S02BG、S02BI、S02BK 官方均为 `5/5 Pass`，
但都没有形成相对 S02BA `3985.330 μs` 的有效稳定提升；S02BN 在通用
fastPath3 长尾矩阵取得最高 `8.529x`、目标合计 `5.775x` 加速并通过
`1571/1571` 门禁，但尚无官方结果。

当前不能确认的是：隐藏 Case 的具体 shape/dtype/TilingKey。后续优化必须
继续依据通用布局、公开矩阵和真机 profiling，不得把平台 Case 当作可猜测
或可硬编码的数据；
只有官方五项结果可以证明是否达到 `0.30 × T_baseline`。
