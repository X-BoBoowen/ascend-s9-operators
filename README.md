# 昇腾算子挑战赛 S9

本仓库用于昇腾 AI 创新大赛算子挑战赛 S9 的 910B 赛题开发，记录五道算子的 CANN 8.5 兼容适配、正确性验证、性能分析和 Ascend C 优化过程。

> 当前状态（2026-07-23）：五题公开基线全部通过，11 个通用边界用例全部通过；`Greater` 和 `IndexAdd` 已完成第一轮稳定优化，主线正在优化 `Transpose`。

## 当前进度

| 算子 | 公开用例 | 当前阶段 | 已验证结论 |
|---|---|---|---|
| `Greater` | 通过 | 第一轮优化完成 | 8 核 Ascend C 快路稳定快于 ACLNN 基线约 5.9% |
| `IndexAdd` | 通过 | 第一轮优化完成 | 15 核 int8 原子累加快路稳定快于 AiCPU 基线约 76.3 倍 |
| `Transpose` | 通过 | 优化中 | CANN 8.5 基线使用 `aclnnPermute` |
| `Concat` | 通过 | 待优化 | CANN 8.5 基线使用 `aclnnCat` |
| `SquareSumV1` | 通过 | 待优化 | 基线使用 `aclnnMul(input, input)` + `aclnnReduceSum` |

## 评测环境

| 项目 | 配置 |
|---|---|
| 云端设备 | Huawei Cloud Ascend 910B4 |
| CANN | 社区版 8.5.0 |
| 编译器 | GCC 10.3 |
| 调用层 | PyTorch + torch_npu CppExtension |
| 性能工具 | `msprof` |
| 基础系统 | 比赛提供的 Euler/ModelArts 环境 |

## 公开用例结果

以下数据来自同一云实例上的公开 `case1`。`time_use` 是赛事脚本输出的原始整数；“约耗时”按 `time_use / 1,000,000` 换算为 profiler 中的微秒值。

| 算子 | ACLNN 基线 `time_use` | 优化后三次有效采样 | 优化中位数 | 约耗时 | 本地收益 |
|---|---:|---|---:|---:|---:|
| `Greater` | 2,540,500 | 2,390,500 / 2,380,000 / 2,400,000 | 2,390,500 | 2.3905 us | 耗时下降约 5.9% |
| `IndexAdd` | 235,075,000 | 3,070,000 / 3,090,000 / 3,080,000 | 3,080,000 | 3.0800 us | 约 76.3x |
| `Concat` | 12,620,500 | - | - | 12.6205 us | 待优化 |
| `Transpose` | 12,020,500 | - | - | 12.0205 us | 优化中 |
| `SquareSumV1` | 2,640,500 | - | - | 2.6405 us | 待优化 |

所有基线均同时满足：

- wheel 从当前算子目录干净重建并安装；
- 日志出现 `case1 verify result pass!`；
- profiler 正常退出；
- `time_use` 为非零有效值；
- 无旧 wheel 或其他算子产物复用。

上述结果用于本地迭代和版本选择，不代表赛事平台最终成绩或排名。不同云实例、驱动状态和 profiler 抖动可能造成差异。

## 正确性评测

### 通用附加用例

[`extra_correctness.py`](extra_correctness.py) 当前包含 11 个独立于官方 `test_op.py` 的附加用例，全部通过：

| 算子 | 覆盖内容 |
|---|---|
| `Greater` | FP16 广播与 NaN/Inf、FP32 广播、int32 相等与边界值 |
| `IndexAdd` | int8 重复索引、FP32 在非零维度上的重复索引 |
| `Concat` | 负维度、空分片、FP16/FP32 |
| `Transpose` | 三维置换、恒等置换、FP16/FP32 |
| `SquareSumV1` | 负轴、`keepdim`、多轴归约、FP16/FP32 |

### 优化专项验证

`GreaterFast` 已验证：

- 2,048 个特殊值组合逐元素零错误；
- 覆盖 NaN、正负无穷、正负零、相等值和 FP16 最小子正规数；
- bool 输出底层字节严格规范为 `0/1`；
- 覆盖 2,176 元素单核路径，以及 FP32、int32、广播、非连续张量等 ACLNN 回退路径。

`IndexAddFast` 已验证：

- 120 条 source 全部写入同一输出行的最坏跨核冲突；
- int8 正负溢出按模 256 正确回绕；
- 未命中的输出行保持原始 input；
- 100 组全范围随机 int8 严格测试零失败；
- 最坏冲突连续重复运行未发现原子竞态；
- 公开 profile 中 30 轮均调度 `IndexAddFast`，`InplaceIndexAddAiCpu` 数量为 0。

## 优化实现

### GreaterFast

快路只处理同形状、连续的 FP16 张量，元素数量为 32 的倍数且不超过 16,384；其他输入继续使用 `aclnnGtTensor`。

当前内核使用 `SubRelu + Mins(1) + Cast(CAST_CEIL)` 生成规范 bool 输出。Host tiling 对适合分核的输入最多使用 8 个 Vector Core，非 256 对齐的合法输入走单核快路。

### IndexAddFast

快路严格匹配公开用例的 int8/int32 形状和连续布局：

- `input`: `[32, 128]`, int8；
- `index`: `[120]`, int32；
- `source`: `[120, 128]`, int8；
- `dim`: `0` 或等价负维度。

实现先保留 `result.copy_(input)`，再使用 15 个 Vector Core，每核处理 8 条 source 行，通过 int8 原子写保证重复 index 的并发语义。其他形状、类型、维度或布局回退到 `aclnnIndexAdd`。

分核 A/B 结果：5 核约 3.18 us，15 核约 3.08 us，30 核约 3.98 us，因此最终保留 15 核版本。

## 评测方法

1. 每次运行前清理 `build/`、`dist/`、旧 wheel 和 `PROF*`。
2. 使用 `run.sh` 重建扩展并通过 `msprof` 运行官方用例。
3. 强制检查准确性标记、profiler 退出码和非零 `time_use`。
4. 性能候选至少获得三次有效独立采样，以中位数决定是否保留。
5. `time_use=0` 或导出不完整的 profile 视为无效，不参与统计。
6. 优化快路之外的输入必须继续通过 ACLNN 回退测试。

运行单个公开用例：

```bash
cd case_910b/<Operator>
bash run.sh 1
```

`GreaterFast` 和 `IndexAddFast` 还需要先构建对应的 Ascend C 算子包，并让扩展能够找到生成的 opapi 动态库。当前脚本仍包含比赛云实例路径，跨机器复现前需要完成路径参数化。

## 目录结构

```text
.
|-- case_910b/              # 五题 CppExtension、公开用例与严格 run.sh
|   |-- Greater/
|   |-- IndexAdd/
|   |-- Concat/
|   |-- Transpose/
|   `-- SquareSumV1/
|-- greater-fast/           # GreaterFast Ascend C kernel、Host tiling 与算子定义
|-- index-add-fast/         # IndexAddFast Ascend C kernel、Host tiling 与算子定义
|-- diagnose_index_add.py   # IndexAdd CPU/NPU/自定义扩展定向诊断
|-- extra_correctness.py    # 五题附加边界正确性测试
`-- sheet-inspect/          # 赛题说明表格的只读检查工具
```

## 已知限制与风险

- 当前两个快路都刻意限制适用范围，隐藏用例不匹配时依赖 ACLNN 回退。
- `IndexAdd` 的 FP16 ACLNN 回退与原生 NPU 实现都可能因原子累加顺序产生非确定误差；观察到相对 CPU 的最大差为 `0.0078125`。该问题不属于 int8 快路，但仍需在后续通用化时处理。
- 自定义算子包的构建、安装和路径配置尚未整理为完全可移植的一键流程。
- 仓库尚未选择开源许可证；正式提交前应按赛事要求补充。

## 下一步

- 完成 `Transpose` 的 Ascend C 快路设计、正确性验证和性能 A/B；
- 依次评估 `Concat` 与 `SquareSumV1` 的融合收益；
- 将算子包构建与动态库路径改为环境变量或相对路径；
- 增加可复现的完整构建说明和自动化回归脚本；
- 按赛事要求整理最终提交结构、许可证和 GitCode PR。
