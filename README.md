# 昇腾 AI 创新大赛 · 算子挑战赛 S9

本仓库记录昇腾 AI 创新大赛算子挑战赛 S9 五道 910B 赛题的适配、验证、性能分析与 Ascend C 优化工作。目标是在满足精度、泛化和提交契约的前提下，持续降低公开及隐藏用例的算子耗时。

> 状态日期：2026-07-28
>
> 目标环境：Ascend 910B、CANN 社区版 8.5.0、GCC 10.3、openEuler/ModelArts
>
> 当前结论：五题均有公开功能通过记录；五套正式算子契约源码均已完成 CANN 8.5.0 构建、独立安装验证和 ZIP 审计。仓库中的 `submission-src/` 已同步为实际生成最终 ZIP 的正式源码。平台 Case1–Case5 仍须逐题上传，以官方结果作为最终结论。

## 1. 当前进度

| 赛题 | 最近稳定公开验证 | 正式提交契约 | 构建/打包状态 | 状态说明 |
|---|---|---|---|---|
| `Greater` | `test pass` / `case1 verify result pass!` | `Greater` | `.run` 和 ZIP 已生成并审计 | 4 个 Vector Core 的 FP16 快路已迁移到正式名称 |
| `IndexAdd` | `test pass` / `case1 verify result pass!` | `IndexAdd` | `.run` 和 ZIP 已生成并审计 | 公共形状使用 15 核，并支持更一般的 index 行数分配 |
| `Transpose` | `test pass` / `case1 verify result pass!` | `Transpose` | `.run` 和 ZIP 已生成并审计 | 32 核二维连续 FP16 分块转置 |
| `Concat` | 早期 `ConcatFast` 稳定版本通过 | `Concat` 动态输入列表 | `.run` 和 ZIP 已生成并审计 | 正式契约已构建；仍需在平台公开 harness 上完成最终闭环 |
| `SquareSumV1` | `test pass` / `case1 verify result pass!` | `SquareSumV1` | 独立重建、922 例回归和 ZIP 审计完成 | 三种 dtype、任意合法轴组合和四类布局均有通用/向量路径 |

“公开验证通过”只表示赛事提供的公开 `case1` 在指定云环境中通过，不代表隐藏用例、平台最终成绩或排名。“已生成并审计”表示正式源码已产出非空 `.run`，并完成 ZIP 路径白名单、正式 `OP_ADD`、内部实验名称残留和源码同步检查；它也不等价于平台上传验收通过。

## 2. 已观察到的性能

以下数据用于版本选择，均来自同一类 Ascend 910B 云实例。不同实例、驱动状态、profiler 采样和 warm-up 会产生波动。

| 算子 | 代表性稳定观察 | 备注 |
|---|---:|---|
| `GreaterFast` | 公开 runner 约 `2.40 us`；稳态 kernel 常见约 `2.0–2.3 us` | 4 核版本优于此前 8 核候选 |
| `IndexAddFast` | 公开 runner 约 `3.08 us` | 公开形状使用 15 核；显著快于 AiCPU 路径 |
| `TransposeFast` | 稳态 kernel 常见约 `4.3–4.9 us` | 二维 `128×256 -> 256×128`，32 核 |
| `SquareSumV1` | public 形状 task-time 中位数约 `6.49 us`；rank-5 三稀疏轴约 `49.22 us` | 相对初始通用版分别约快 3.1 倍和 8.2 倍；典型双稀疏轴约快 51 倍 |
| `Concat` | 尚不发布最终数字 | 在线契约名称迁移和最终复测仍在进行 |

赛事 runner 会在目标算子之间插入较大的 `Mul` 工作负载，因此 runner 输出、kernel profiler 时间和平台排行榜耗时不能直接混用。

## 3. 实现概览

### 3.1 Greater

正式提交算子 `Greater` 源自开发阶段的 `GreaterFast`，面向连续、同形状的 FP16 输入：

- Host tiling 根据元素数量选择 1–4 个 Vector Core；
- 每核处理连续片段；
- kernel 通过向量计算生成严格规范的 bool 输出；
- 不满足快路条件的输入由 PyTorch 扩展回退到 ACLNN。

已覆盖 NaN、Inf、正负零、相等值、FP16 子正规数、FP32、int32、广播及非连续输入等专项测试。

### 3.2 IndexAdd

正式提交算子 `IndexAdd` 源自开发阶段的 `IndexAddFast`，面向 int8 输入、int32 index 和按第 0 维累加：

- 每核基础处理 8 条 source 行；
- Host 根据 index 数量动态选择核数，最多 32 核；
- public case 的 120 条 index 使用 15 核；
- kernel 使用 int8 原子加，保持重复 index 的并发语义；
- tiling key 区分整 8 行和尾块路径；
- wrapper 先复制原 input，再对命中行进行累加。

专项验证包括最坏重复 index、跨核冲突、正负溢出模 256、未命中行保持、随机全范围 int8 以及重复运行竞态检查。

### 3.3 Transpose

正式提交算子 `Transpose` 源自开发阶段的 `TransposeFast`，当前处理二维连续 FP16 转置：

- 将矩阵划分为固定 tile；
- 最多使用 32 个 Vector Core；
- Host 均匀分配 tile，并单独处理余数；
- kernel 使用 Ascend C `Transpose` 指令；
- 使用 4 级队列缓冲；
- 非二维、非 FP16、非连续或不符合快路契约的输入回退到 `aclnnPermute`。

### 3.4 Concat

正式提交算子 `Concat` 使用动态输入列表契约，kernel 算法源自稳定 `ConcatFast`，面向末维拼接：

- 支持多个 FP16 输入；
- Host 通过动态输入列表读取输入数量和各输入形状；
- kernel 通过 `ListTensorDesc` 获取每个输入的 GM 地址；
- Host 计算 outer、各输入末维宽度和输出行跨度；
- 最多使用 16 核；
- kernel 使用双缓冲，在输入和输出 GM 之间按行搬运；
- 输入宽度与偏移由 tiling 数据统一传入。

早期 `ConcatFast` 和 `ConcatD` 仅作为开发过程记录；`submission-src/Concat` 已切换到赛事正式 `Concat` 契约并成功生成提交包。正式契约仍需在平台公开 harness 上完成最终功能和性能复测，因此不发布未经确认的最终耗时。

### 3.5 SquareSumV1

正式提交算子 `SquareSumV1` 是面向完整题面契约的自定义融合实现：

- 支持 float16、bfloat16、float32，rank 1–5，负轴、多轴、`keep_dims`；
- Host 统一规范化归约轴并生成输出/归约维度与输入 stride，保留通用坐标回退；
- 连续后缀归约批量处理多行，长归约自动切回分段归约；
- 连续中间轴使用二维搬运、批量平方和列累加；
- 包含最后一维的分离轴使用分组后缀路径，可覆盖多个非相邻归约轴；
- 不包含最后一维的分离轴使用跨步 inner 批处理；
- FP16 先按输入 dtype 执行平方再转 FP32 累加，保持 `torch.square` 的溢出/舍入语义；
- 小/中规模快路使用 32 核，输入达到约 100 万元素时自适应启用 40 核；
- 多批 UB 复用显式设置 V→MTE2 同步，且 `WholeReduceSum` 每批不超过 255 次硬件上限。

最终候选通过 46 个定向、150 个原随机和 726 个扩展用例，共 922 项；扩展集覆盖 31/32/33、63/64/65、8191/8192/8193、10000 等边界、三种 dtype、rank 1–5、各种轴顺序、负轴、`keep_dims`、NaN/Inf/±0、FP16 溢出和大输出分批。

## 4. 仓库结构

```text
.
|-- case_910b/
|   |-- Greater/            # 官方公开用例、CppExtension 和运行脚本
|   |-- IndexAdd/
|   |-- Transpose/
|   |-- Concat/
|   `-- SquareSumV1/
|-- submission-src/         # 实际生成最终 ZIP 的五题正式契约源码
|   |-- Greater/
|   |-- IndexAdd/
|   |-- Transpose/
|   |-- Concat/
|   `-- SquareSumV1/
|-- operator-descriptors/   # 开发阶段 *Fast 工程使用的 msopgen 描述
|-- greater-fast/           # GreaterFast 早期独立开发快照
|-- index-add-fast/         # IndexAddFast 早期独立开发快照
|-- extra_correctness.py    # 五题附加边界正确性测试
|-- diagnose_index_add.py   # IndexAdd 定向诊断
`-- sheet-inspect/          # 赛题表格的只读检查工具
```

`submission-src/` 是当前最重要的源码入口。它已从五个最终 ZIP 反向核对并同步，只保留可开源的 Host/Kernel 源码，不提交 `.run`。每题包含：

```text
<Operator>/
|-- op_host/
|   |-- CMakeLists.txt
|   |-- <official_name>.cpp
|   `-- <official_name>_tiling.h
`-- op_kernel/
    |-- CMakeLists.txt
    `-- <official_name>.cpp
```

正式文件名映射如下：

| 赛题 | Host/Kernel 文件 | Tiling 文件 | 正式算子/Kernel | 开发阶段对应名称 |
|---|---|---|---|---|
| Greater | `greater.cpp` | `greater_tiling.h` | `Greater` / `greater` | `GreaterFast` / `greater_fast` |
| IndexAdd | `index_add.cpp` | `index_add_tiling.h` | `IndexAdd` / `index_add` | `IndexAddFast` / `index_add_fast` |
| SquareSumV1 | `square_sum_v1.cpp` | `square_sum_v1_tiling.h` | `SquareSumV1` / `square_sum_v1` | `SquareSumFast` / `square_sum_fast` |
| Concat | `concat.cpp` | `concat_tiling.h` | `Concat` / `concat` | `ConcatFast` / `concat_fast` |
| Transpose | `transpose.cpp` | `transpose_tiling.h` | `Transpose` / `transpose` | `TransposeFast` / `transpose_fast` |

平台 `Zip_Check` 会检查正式文件名。正式提交还要求 `OP_ADD`、tiling 注册类、kernel 入口、CMake 生成目标以及 `.run` 中的算子元数据保持一致；仅在 ZIP 内随意改名会导致提交契约不一致。

## 5. 环境要求

| 项目 | 版本/配置 |
|---|---|
| 设备 | Ascend 910B |
| CANN | 社区版 8.5.0 |
| 编译器 | GCC/G++ 10.3 |
| Python | 比赛环境 Python 3.9 |
| PyTorch NPU | 比赛镜像内置版本 |
| 操作系统 | openEuler / ModelArts 比赛环境 |

初始化环境示例：

```bash
source /path/to/Ascend/cann-8.5.0/set_env.sh
export PATH=/path/to/gcc-10.3/bin:$PATH
export CC=/path/to/gcc-10.3/bin/gcc
export CXX=/path/to/gcc-10.3/bin/g++
```

请勿直接复用其他 CANN 版本生成的工程或 `.run`。

## 6. 构建 Ascend C 算子

`operator-descriptors/` 保存的是开发阶段内部 `*Fast` 工程的算子描述。以 GreaterFast 为例：

```bash
msopgen gen \
  -i operator-descriptors/greater_fast.json \
  -f pytorch \
  -c ai_core-ascend910b \
  -out build/Greater \
  -lan cpp
```

这类描述适合复现开发 harness，但不能直接与正式 `submission-src/` 混用。正式提交工程必须使用与赛事签名一致的算子描述生成，再同步对应 `submission-src/<Operator>/op_host` 和 `op_kernel`。需要注意：

- `submission-src` 同时使用平台正式文件名、正式 `OP_ADD` 和正式 kernel 入口；
- `operator-descriptors` 当前保留经过验证的内部 `*Fast` 开发类型，不是最终平台描述；
- 生成工程的输入、输出、属性、动态输入、源文件映射和 include 名称必须与正式契约一致；
- 不应仅改文件名而不重新构建 `.run`。

在生成工程中执行：

```bash
cd build/Greater
bash build.sh
```

成功后应产生：

```text
build_out/custom_opp_euleros_aarch64.run
```

五题正式工程已在 CANN 8.5.0 云环境中成功构建过；仓库公开实际打包使用的核心源码和开发 descriptor，但尚未公开由赛事正式描述生成的完整工程，也尚未把工程生成、编译、验证和打包整合成跨机器的一键脚本。

## 7. 运行公开用例

单题测试入口：

```bash
cd case_910b/<Operator>
bash run.sh 1
```

Greater、IndexAdd 和 Transpose 的 wrapper 需要链接相应 opapi 库，可以通过环境变量指定：

```bash
export GREATER_FAST_LIB_DIR=/path/to/Greater/build_out/op_api/lib
export INDEX_ADD_FAST_LIB_DIR=/path/to/IndexAdd/build_out/op_api/lib
export TRANSPOSE_FAST_LIB_DIR=/path/to/Transpose/build_out/op_api/lib
```

成功运行至少应满足：

- wheel 从当前源码重新构建；
- 日志出现 `test pass`；
- 日志出现 `case1 verify result pass!`；
- profiler 正常退出；
- 目标算子实际调度到预期 kernel；
- 没有复用其他版本留下的 wheel、动态库或缓存。

## 8. 附加正确性验证

[`extra_correctness.py`](extra_correctness.py) 包含公开用例以外的附加测试：

| 算子 | 主要覆盖 |
|---|---|
| Greater | FP16 广播、NaN/Inf、FP32、int32、相等和边界值 |
| IndexAdd | 重复 index、int8 回绕、FP32 非零维度 |
| Concat | 负维度、空分片、FP16/FP32 |
| Transpose | 三维置换、恒等置换、FP16/FP32 |
| SquareSumV1 | 三种 dtype、负轴/多轴、keepdim、rank 1–5、长归约边界、NaN/Inf/±0 和溢出语义 |

这些测试用于验证 wrapper 的泛化回退和算子语义，不等价于赛事隐藏用例。

## 9. 提交包要求

赛事页面要求每次只提交一道题，并将以下内容放入同一个目录：

- `op_host/`
- `op_kernel/`
- 与源码一致的 `custom_*.run`

随后必须使用赛事提供的 `zip_op.sh` 打包：

```bash
bash zip_op.sh Greater
```

Greater 的正确 ZIP 结构示例：

```text
Greater_zip/
|-- op_host/
|   |-- CMakeLists.txt
|   |-- greater.cpp
|   `-- greater_tiling.h
|-- op_kernel/
|   |-- CMakeLists.txt
|   `-- greater.cpp
`-- custom_opp_euleros_aarch64.run
```

截至 2026-07-28，五题均已按这一结构生成最终候选 ZIP。离线审计包含：

- ZIP 可完整读取，且每题只有顶层 `<Operator>_zip/`、`op_host/`、`op_kernel/` 和一个 `custom_opp_euleros_aarch64.run`；
- Host/Kernel 文件名符合 `Zip_Check` 要求；
- Host 源码包含正式 `OP_ADD(<Operator>)`；
- 正式源码中不残留 `Fast`、`ConcatD` 或 `concat_d` 等内部实验名称；
- 每个包内只有一个非空 `.run`，并记录候选包的 SHA-256 以便后续上传核对。

本轮候选包校验值：

| 文件 | SHA-256 |
|---|---|
| `Concat.zip` | `82e17ebf1f062c42f61d64f0788b7c2ed6a2633ff26a9abd44fae5b6da7fe814` |
| `Greater.zip` | `6c61f94e838843b052dd72e4cd5622cf309f5d3811370e15a6a15d2353df2539` |
| `IndexAdd.zip` | `d2b087093f5a7e3bc1559cadacd703f7bf3592fbd1a5a5dcf72280a2d2ac2a5f` |
| `SquareSumV1.zip` | `af9ae474bdedb0b5a7b55d39fc0de9a3bd448df39a01b08926304513412e642b` |
| `Transpose.zip` | `0b88b825e7ee6e5cb598e5c3eb4638d4f652a4a0eb32671087a10a4aa3ff57e8` |

最终候选包仍应逐题上传平台，以平台 `Zip_Check`、精度验证和性能结果作为最终结论。本仓库不提交 `.run`、ZIP、wheel、profile 数据库和构建缓存；二进制提交包必须在 CANN 8.5.0 目标环境中由对应源码重新构建。

## 10. 已知问题

1. Concat 正式动态输入契约已经构建和打包，但尚未完成平台公开 harness 的最终功能与性能闭环。
2. SquareSumV1 的生成式 ACLNN 验证封装会在进入 tiling 前拒绝零长度 `IntArray`，因此空 `axis` 只能以等价的“显式全轴”做端到端测试；Host 已实现空轴全维归约语义，但仍需平台用例确认该属性的实际注入方式。
3. 公开用例通过以及离线包审计均不保证隐藏用例泛化、平台上传验收或榜单成绩。
4. 五题尚未形成跨机器的一键生成、编译、安装、测试和打包流水线。
5. 仓库当前未声明开源许可证；正式对外复用前需要补充合适的 LICENSE。

## 11. 安全与仓库卫生

以下内容不会进入 Git：

- SSH 私钥、`.pem`、`.key`；
- 云账号、令牌和本地 `.env`；
- `.run`、`.so`、wheel、目标文件；
- `PROF*`、数据库和 profiler 日志；
- 本地同步、解压和文档检查临时目录；
- 比赛代金券、个人联系方式和其他提交材料。

提交前建议执行：

```bash
git status --short
git diff --check
git ls-files | grep -Ei '\.(pem|key|run|so|whl)$'
```

## 12. 后续计划

- 逐题上传最终候选 ZIP，记录 `Zip_Check`、精度和性能结果；
- 完成 Concat 正式动态输入契约的公开用例闭环；
- 根据 SquareSumV1 平台实际 Case1–Case5 形状继续做定向性能 A/B；
- 扩展五题隐藏形状、dtype、广播和非连续布局覆盖；
- 增加统一的一键构建、测试、profile 和提交包脚本；
- 按赛事要求准备 GitCode 开源 PR。

## 13. 免责声明

仓库中的性能数据是本地云实例上的工程测量，仅用于优化决策。赛事平台的最终成绩由官方隐藏用例、精度检查和计时环境决定。
