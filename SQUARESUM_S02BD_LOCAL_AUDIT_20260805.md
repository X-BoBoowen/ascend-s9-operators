# SquareSumV1-S02BD 本地审计与云端验证准备

更新时间：2026-08-05（Asia/Shanghai）

## 1. 官方结果与目标

S02BA 官方结果为：

```text
6.51 / 399.088 / 138.753 / 2550.761 / 890.218 μs
prof_sum: 3985.330 μs
```

S02BB 官方结果为：

```text
7.88 / 405.58 / 140.8 / 2551.56 / 902.756 μs
prof_sum: 4008.576 μs
```

S02BB 五个 Case 均未优于 S02BA，总耗时变慢 `23.246 μs`（`0.58%`），
因此不晋级。后续正式对比基线仍固定为 S02BA 的 `3985.330 μs`，目标
保持为：

```text
0.30 × 3985.330 = 1195.599 μs
```

## 2. S02BD 单变量候选

候选目录：

```text
candidates/squaresum_s02bd_trailing_singleton_last_20260805/SquareSumV1
```

候选提交：

```text
b412772 perf(squaresum): route trailing singleton as contiguous
```

S02BD 从 S02BB 复制，只修改 Host 的物理布局分类。原来被识别为
`fastPath=2` 的连续中间归约，如果归约组后的所有保留维乘积为 1，且
归约元素数非零，则物理内存中每个输出对应的归约段实际上与连续末轴
完全相同。S02BD 将该布局改道到已有 `fastPath=1`：

```cpp
if (fastPath == 2U &&
    innerElements == 1U &&
    reduceElements > 0U) {
    fastPath = 1U;
}
```

这不是按 shape、dtype 或 Case 编号硬编码；判断只依赖通用物理布局。
`reduceElements == 0` 明确保留旧路由，避免空归约进入不适用的连续快路。

S02BD Kernel 与 S02BB 的 SHA-256 相同：

```text
B1CAE31AF656F9558AA0A504957EFC38B5A292893D212AD7788155EF3EFEF69B
```

除 `op_host/square_sum_v1.cpp` 上述 5 行外，候选内其余文件与 S02BB
逐文件一致。

## 3. 本地地址与语义审计

新增不依赖 CANN、NPU 或 PyTorch 的审计脚本：

```text
diagnostics/squaresum_s02bd_trailing_singleton_static_20260805.py
```

脚本按当前 Host 规则重建连续性分类，并对 rank 1～5、维度
`{0,1,2,3}`、全部非空归约轴组合和 `keep_dims=true/false` 穷举。对每个
被改道布局，逐地址验证：

1. S02BB 路由必须为 `fastPath=2`，S02BD 必须为 `fastPath=1`；
2. 归约组后的物理元素数必须恰好为 1；
3. 一个输出的全部归约地址必须严格等于 `[base, base+reduceElements)`；
4. 按输出扁平顺序连接所有归约段，必须严格等于整个输入地址
   `[0, inputElements)`，不存在乱序、重复、遗漏或跨输出；
5. keepDims 产生的 stride-0 输出轴不改变地址映射；
6. 负轴、乱序轴和重复轴与规范化轴形式等价；
7. 零归约、尾宽 2 和尾宽 4 控制布局不改道；
8. 候选完整文件集合与 S02BB 一致，除目标 Host 文件外逐文件哈希相同。

本地结果：

```text
SOURCE_GUARDS=PASS
LAYOUTS=72168
PROMOTED=1644
PROMOTED_KEEP_DIMS_EQUIVALENTS=822
ZERO_REDUCE_RETAINED=698
PROMOTED_OUTPUT_CARDINALITIES=11
AXIS_FORM_CASES=3
LARGE_CASES=8
LARGE_PROMOTED=5
LARGE_CONTROLS=3
SUMMARY: S02BD trailing-singleton static audit passed
```

三份 S02BD 专项脚本均已通过 Python 语法编译；源码补丁通过
`git diff --check`。

## 4. 必须由 910B4 裁决的性能边界

语义等价不等于性能必胜。S02BD 将 `fastPath=2` 改为 `fastPath=1` 后，
还会联动已有 Host 调度：

- `outputElements <= 8` 且输入/归约足够大时，新旧版本都使用 40 核
  workspace；S02BD 把每核计算从中间轴短行搬运改为连续归约；
- `outputElements=9…1024` 时，S02BB 可使用 middle workspace，S02BD
  会进入已有 last-axis 非 workspace 路径；该区间存在调度退化风险，不能
  靠静态模型宣称收益；
- `outputElements > 1024` 时，两版均不使用 workspace，但执行的中间轴与
  连续末轴 Kernel 路径不同，仍需真实设备比较；
- 小归约会进入 last-axis rows/segmented 路径，长归约可能进入 width
  1/2/4/8 向量路径或 workspace tree finalizer，必须分别覆盖。

因此云端性能脚本已扩展到 `outputElements=8/9/32/64/256/1024/1025`
边界，并保留小归约、超长归约、多个外层、单例间隔、两个尾部单例、
FP16/BF16/FP32 及尾宽 2/4 控制。总计 22 个布局、66 个带正确性校验的
性能项。

## 5. 云端恢复后的最小验证闭环

云端资源恢复后只执行以下闭环，不直接打包：

1. 在同一台 910B4、同一 CANN 8.5.0 环境重新安装 S02BB；
2. 运行 66 项 trailing-singleton 性能矩阵并保存 7 组样本中位数；
3. 构建、安装 S02BD，原样运行同一矩阵；
4. 逐布局、逐 dtype 比较，重点检查 8/9 和 1024/1025 两个调度边界；
5. 运行 96 项专项正确性（16 个布局 × 3 dtype × keepDims 真/假）；
6. 再运行 46 定向、150 随机、4 BF16 语义、726 扩展和现有关键路径
   门禁；
7. 只有全部正确，目标布局稳定改善，控制布局无实质退化，才冻结二进制、
   审计 ZIP 并交付正式平台测评。

若 `9…1024` 区间出现退化，将依据同机 A/B 结果收紧通用
`outputElements` 门槛，或保留原 workspace 调度并仅替换每核连续归约
实现。没有真实 910B4 数据前不预设门槛。

## 6. 当前结论

S02BD 已完成本地代码、地址等价性、边界覆盖和云端测试准备；尚未完成
CANN 8.5.0 构建、910B4 正确性和性能验证，因此当前不是可提交包，也
不能宣称改善任何官方 Case。华为云资源不足期间不发起 SSH 或云端任务。
