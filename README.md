# 昇腾 AI 创新大赛 S9 算子优化

本仓库记录 Ascend 910B、CANN 社区版 8.5.0 环境下五道 S9
算子的通用实现、真机验证、性能实验和正式提交源码。

当前工作原则：

- 先满足题面全部 dtype、rank、轴、广播、空输入和非对齐约束；
- 不根据隐藏样例猜 shape，不写公开 Case 专用分支；
- 每题在独立实验目录中构建、安装、回归和打包；
- 只有平台 Case1–Case5 才能作为最终正确性和性能结论；
- Git 仓库保存源码、测试与文档，不保存 `.run`、ZIP、wheel 和
  profiler 数据库。

> 状态时间：2026-07-29 20:34（Asia/Shanghai）
>
> 当前阶段：五道题均已完成本轮工程闭环并分别生成待测 ZIP；
> `Transpose` 已完成 FP16 尾块、FP32/INT32 和 INT8 矩阵路径优化。

## 1. 当前状态

| 题目 | 工程状态 | 真机回归 | 当前提交包 | 平台状态 |
| --- | --- | ---: | --- | --- |
| Concat | 本轮完成 | 9 定向 + 100 随机 + 168 扩展 | 已生成并审计 | 新包待上传 |
| Greater | 本轮完成 | 26 定向 + 220 随机 + 9 个 int32 扩展；扩展集连续复跑 3 次 | 已生成并审计 | 新包待上传 |
| IndexAdd | 本轮完成 | 23 定向 + 170 随机 + 345 扩展 = 538 | 已生成并审计 | 新包已交用户评测 |
| Transpose | 本轮完成 | 48 定向 + 200 随机 + 152 扩展 + 84 边界 = 484 | 已生成并审计 | 新包待上传 |
| SquareSumV1 | 本轮完成 | 922 全量 + 28 专项 + 4 组 BF16 逐位语义 | 已生成并审计 | 新包待上传 |

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
| `SquareSumV1.zip` | 451339 B | `d5d0407c7f81519dce36682d18104d1579bf96642ea898e0706c91550942dbb7` |

包内 `.run`：

| 题目 | `.run` SHA-256 |
| --- | --- |
| Concat | `9dca80b07e85eacf9b90ac98349f64bf0ba0db19565c440f0c16f12cbf1ca873` |
| Greater | `d8ab6f81aa1eaa775cbe69078db09901037ccad667216697371475a42dbb4298` |
| IndexAdd | `58a5be8bbbc8db62708973fea21bd6630e1d938ae9c8a4ad28132f3014ce679d` |
| Transpose | `2baf9ecb7e38716b5d4a7ca8e89c890be2392c4b8c9352d8ca67d8d23ca9a9af` |
| SquareSumV1 | `b8748b428673562c0ef25d7c1c9df7a26c2a5ca30988afdffa8f284aa15fc46e` |

五个 ZIP 均只有一个顶层 `<Operator>_zip/`，包含 `op_host/`、
`op_kernel/` 和一个非空 `custom_opp_euleros_aarch64.run`。
`IndexAdd` 的 `.run` 在 ZIP 中保持 `0750`，其余三个为 `0755`。

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
- 合计 922 个用例在候选构建和正式目录干净构建上各完整通过一次；
- 10 个工作区模式探针；
- 10 个 TilingKey/回退边界探针；
- 4 个 FP32 原子稳定性探针；
- 4 个 BF16 严格逐位语义探针。

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
| `op_host/square_sum_v1.cpp` | `eac0db09a9f929898ecf145580052822041df01b5a76ff05eb701f592ea98abd` |
| `op_host/square_sum_v1_tiling.h` | `e669820ab05878a8701fe2e6f33f4b44076ee93e74c1a308c3b5e8fcf54079f6` |
| `op_kernel/CMakeLists.txt` | `f70a5fa430598410e08085e9e16bfdfd86254a87977bdf4105f9f548b5fa4cee` |
| `op_kernel/square_sum_v1.cpp` | `04d5f4d859cb044f9392e3d69a898b6db313919a9ed5f8a9c45385e856eb149b` |

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
/home/ma-user/work/s9/release/squaresum_final_20260729
```

云端发布包：

```text
/home/ma-user/work/s9/releases/concat_20260728_1446/Concat.zip
/home/ma-user/work/s9/releases/greater_20260728_1511/Greater.zip
/home/ma-user/work/s9/releases/indexadd_20260728_1622/IndexAdd.zip
/home/ma-user/work/s9/releases/transpose_20260729_2032/Transpose.zip
/home/ma-user/work/s9/releases/squaresum_20260729_1938/SquareSumV1.zip
```

SquareSumV1 的正式回归证据：

```text
/home/ma-user/work/s9/validation/regression_922_formal_final_20260729
/home/ma-user/work/s9/experiments/squaresum_workspace_best_20260729_1854/validation/profile_final_*
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

## 11. 平台已知结果与解释

用户此前提供的结果：

| 题目 | Case1 | Case2 | Case3 | Case4 | Case5 | 总和 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Concat | 15.7005 | 46.871 | 123.5425 | 116.7625 | 847.587 | 1150.4635 |
| Greater | 5.312 | 39463.772 | 2261.36 | 94.512 | 210.032 | 42034.988 |
| IndexAdd | 9.7805 | 41352.976 | 90.9915 | 2564.231 | 80160.552 | 124178.531 |
| Transpose | 4.5805 | 257.445 | 2466.05 | 2197.384 | 34623.812 | 39549.2715 |

这些结果对应四题本轮最终 ZIP 之前的版本，不能用来评价本 README
记录的新包。新包上传后必须记录：

- 上传时间；
- ZIP SHA-256；
- Case1–Case5 Pass/Fail；
- 每个 Case 的耗时；
- `prof_sum`；
- 排名变化；
- 若失败，保存平台结果文本，不猜隐藏数据。

## 12. 下一次工作如何接续

### 12.1 先收平台反馈

五题当前正式 ZIP 已交用户逐题上传。上传后分别记录五个 Case 的
Pass/Fail、每个耗时和 `prof_sum`，再决定下一轮优化方向。平台测评
期间不得覆盖这些已验证基线，也不得用旧结果评价当前 ZIP。

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

收到五题平台结果后，先确认每题五个隐藏 Case 是否全 Pass，再按题目
顺序和最慢 Case 的量级，与现有通用路径建立可验证假设。隐藏 shape
不可见，不得反推或硬编码；每个候选必须在独立实验目录完成正确性与
同形状 A/B，未经验证不得覆盖对应的 `submission-src/<Op>`。

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

当前能够确认的是：五题的正式源码、云端构建产物、本地提交包和验证
证据已形成可复现闭环；Transpose 的三类矩阵慢路径已完成向量化，
正式构建通过 484 例回归，并有同形状事件计时和设备 profiling 证据。

当前不能确认的是：新包在官方隐藏 Case 上的最终耗时、排名和是否已经
进入奖励区间。所有平台结论必须等待用户回传五个 Case 的实际结果。
