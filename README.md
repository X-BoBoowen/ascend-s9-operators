# 昇腾 AI 创新大赛 · 算子挑战赛 S9

本仓库记录昇腾 AI 创新大赛算子挑战赛 S9 五道 910B 赛题的适配、验证、性能分析与 Ascend C 优化工作。目标是在满足精度、泛化和提交契约的前提下，持续降低公开及隐藏用例的算子耗时。

> 状态日期：2026-07-24
>
> 目标环境：Ascend 910B、CANN 社区版 8.5.0、GCC 10.3、openEuler/ModelArts
>
> 当前结论：五题均有公开功能通过记录；Greater、IndexAdd、Transpose 已形成稳定自定义快路；Concat 与 SquareSumV1 同时保留稳定实现和后续优化分支。

## 1. 当前进度

| 赛题 | 最近稳定公开验证 | 当前实现 | 提交源码 | 状态说明 |
|---|---|---|---|---|
| `Greater` | `test pass` / `case1 verify result pass!` | 4 个 Vector Core 的 FP16 快路 | 已整理 | 稳定版本已进入提交源码 |
| `IndexAdd` | `test pass` / `case1 verify result pass!` | 动态分核的 int8 原子累加快路 | 已整理 | 公共形状使用 15 核，已补充更一般的 index 行数分配 |
| `Transpose` | `test pass` / `case1 verify result pass!` | 32 核二维分块转置 | 已整理 | 支持二维连续 FP16 的泛化分块 |
| `Concat` | 早期 `ConcatFast` 稳定版本通过 | 16 核、双缓冲、末维拼接 | 已整理稳定版 | 最新 `ConcatD` 在线名称迁移尚未完全闭环，因此未宣称为最终版 |
| `SquareSumV1` | `test pass` / `case1 verify result pass!` | FP16 平方并按末维归约的融合核 | 已整理 | 当前本地最快调用路径仍是官方组合实现；自定义融合 v4 已验证正确 |

“公开验证通过”只表示赛事提供的公开 `case1` 在指定云环境中通过，不代表隐藏用例、平台最终成绩或排名。

## 2. 已观察到的性能

以下数据用于版本选择，均来自同一类 Ascend 910B 云实例。不同实例、驱动状态、profiler 采样和 warm-up 会产生波动。

| 算子 | 代表性稳定观察 | 备注 |
|---|---:|---|
| `GreaterFast` | 公开 runner 约 `2.40 us`；稳态 kernel 常见约 `2.0–2.3 us` | 4 核版本优于此前 8 核候选 |
| `IndexAddFast` | 公开 runner 约 `3.08 us` | 公开形状使用 15 核；显著快于 AiCPU 路径 |
| `TransposeFast` | 稳态 kernel 常见约 `4.3–4.9 us` | 二维 `128×256 -> 256×128`，32 核 |
| `SquareSumV1` | 官方组合路径约 `2.58 us` | 该数字不是自定义融合核的最终平台成绩 |
| `Concat` | 尚不发布最终数字 | 在线契约名称迁移和最终复测仍在进行 |

赛事 runner 会在目标算子之间插入较大的 `Mul` 工作负载，因此 runner 输出、kernel profiler 时间和平台排行榜耗时不能直接混用。

## 3. 实现概览

### 3.1 Greater

`GreaterFast` 面向连续、同形状的 FP16 输入：

- Host tiling 根据元素数量选择 1–4 个 Vector Core；
- 每核处理连续片段；
- kernel 通过向量计算生成严格规范的 bool 输出；
- 不满足快路条件的输入由 PyTorch 扩展回退到 ACLNN。

已覆盖 NaN、Inf、正负零、相等值、FP16 子正规数、FP32、int32、广播及非连续输入等专项测试。

### 3.2 IndexAdd

`IndexAddFast` 面向 int8 输入、int32 index 和按第 0 维累加：

- 每核基础处理 8 条 source 行；
- Host 根据 index 数量动态选择核数，最多 32 核；
- public case 的 120 条 index 使用 15 核；
- kernel 使用 int8 原子加，保持重复 index 的并发语义；
- tiling key 区分整 8 行和尾块路径；
- wrapper 先复制原 input，再对命中行进行累加。

专项验证包括最坏重复 index、跨核冲突、正负溢出模 256、未命中行保持、随机全范围 int8 以及重复运行竞态检查。

### 3.3 Transpose

`TransposeFast` 当前处理二维连续 FP16 转置：

- 将矩阵划分为固定 tile；
- 最多使用 32 个 Vector Core；
- Host 均匀分配 tile，并单独处理余数；
- kernel 使用 Ascend C `Transpose` 指令；
- 使用 4 级队列缓冲；
- 非二维、非 FP16、非连续或不符合快路契约的输入回退到 `aclnnPermute`。

### 3.4 Concat

仓库中的稳定 `ConcatFast` 提交源码面向末维拼接：

- 支持多个 FP16 输入；
- Host 计算 outer、各输入末维宽度和输出行跨度；
- 最多使用 16 核；
- kernel 使用双缓冲，在输入和输出 GM 之间按行搬运；
- 输入宽度与偏移由 tiling 数据统一传入。

后续尝试将在线算子名称迁移为 `ConcatD`，但最近一次完整公开 harness 尚未闭环。因此仓库明确区分“稳定提交源码”和“最新实验命名”，避免把失败实验写成已完成成果。

### 3.5 SquareSumV1

自定义 `SquareSumFast` 融合核执行：

1. FP16 输入搬入 UB；
2. 转换为 FP32；
3. 逐元素平方；
4. 对最后一维执行 `WholeReduceSum`；
5. 转回 FP16 并写回。

Host 按 outer 行数最多分配 32 核，并对 reduce 长度进行对齐。该融合版本已通过正确性验证，但当前选中的最快 runner 路径仍是官方 `Mul + ReduceSum` 组合，因此二者不能混为同一个性能结论。

## 4. 仓库结构

```text
.
|-- case_910b/
|   |-- Greater/            # 官方公开用例、CppExtension 和运行脚本
|   |-- IndexAdd/
|   |-- Transpose/
|   |-- Concat/
|   `-- SquareSumV1/
|-- submission-src/         # 当前五题提交源码，使用平台要求的正式文件名
|   |-- Greater/
|   |-- IndexAdd/
|   |-- Transpose/
|   |-- Concat/
|   `-- SquareSumV1/
|-- operator-descriptors/   # msopgen 使用的内部算子描述
|-- greater-fast/           # GreaterFast 早期独立开发快照
|-- index-add-fast/         # IndexAddFast 早期独立开发快照
|-- extra_correctness.py    # 五题附加边界正确性测试
|-- diagnose_index_add.py   # IndexAdd 定向诊断
`-- sheet-inspect/          # 赛题表格的只读检查工具
```

`submission-src/` 是当前最重要的源码入口。每题包含：

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

| 赛题 | Host/Kernel 文件 | Tiling 文件 | 稳定内部算子类型 |
|---|---|---|---|
| Greater | `greater.cpp` | `greater_tiling.h` | `GreaterFast` |
| IndexAdd | `index_add.cpp` | `index_add_tiling.h` | `IndexAddFast` |
| SquareSumV1 | `square_sum_v1.cpp` | `square_sum_v1_tiling.h` | `SquareSumFast` |
| Concat | `concat.cpp` | `concat_tiling.h` | `ConcatFast` |
| Transpose | `transpose.cpp` | `transpose_tiling.h` | `TransposeFast` |

平台 `Zip_Check` 会检查正式文件名。仅在 ZIP 内随意改名会造成源码、CMake 和 `.run` 不一致，因此提交包必须从一致的源码状态重新编译。

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

算子描述位于 `operator-descriptors/`。以 Greater 为例：

```bash
msopgen gen \
  -i operator-descriptors/greater_fast.json \
  -f pytorch \
  -c ai_core-ascend910b \
  -out build/Greater \
  -lan cpp
```

随后将对应 `submission-src/Greater/op_host` 和 `op_kernel` 源码同步到生成工程。需要注意：

- `submission-src` 使用平台正式文件名；
- descriptor 当前保留经过验证的内部 `*Fast` 算子类型；
- 生成工程的源文件映射、CMake 路径和 include 名称必须保持一致；
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

目前仓库已经公开核心源码和 descriptor，但尚未把五题工程生成、文件映射、编译与打包整合成完全可移植的一键脚本。

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
| SquareSumV1 | 负轴、keepdim、多轴归约、FP16/FP32 |

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

本仓库不提交 `.run`、wheel、profile 数据库和构建缓存。二进制提交包应由参赛者在 CANN 8.5.0 的目标环境中从对应提交源码重新构建。

## 10. 已知问题

1. 最新“内部算子类型直接改成官方名称”的完整工程仍有链接问题，已观察到 `cannot find -lascend_kernels`。因此当前稳定方案保留经过验证的内部 `*Fast` 类型，并在提交层使用平台要求的正式文件名。
2. Concat 的 `ConcatD` 在线名称迁移尚未完成最终公开 harness 验证；仓库保留早期稳定 `ConcatFast` 提交源码。
3. SquareSumV1 当前最快 runner 路径是官方组合实现，而 `submission-src` 中是已验证正确的自定义融合版本。
4. 公开用例通过不保证隐藏用例泛化或榜单成绩。
5. 五题尚未形成跨机器的一键生成、编译、安装、测试和打包流水线。
6. 仓库当前未声明开源许可证；正式对外复用前需要补充合适的 LICENSE。

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

- 修复正式内部算子名称工程的 `ascend_kernels` 链接与生成依赖；
- 完成 Concat 在线契约的公开用例闭环；
- 对 SquareSumV1 融合核继续做性能 A/B；
- 扩展五题隐藏形状、dtype、广播和非连续布局覆盖；
- 增加统一的一键构建、测试、profile 和提交包脚本；
- 按赛事要求准备 GitCode 开源 PR。

## 13. 免责声明

仓库中的性能数据是本地云实例上的工程测量，仅用于优化决策。赛事平台的最终成绩由官方隐藏用例、精度检查和计时环境决定。
