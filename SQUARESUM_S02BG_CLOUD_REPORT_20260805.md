# SquareSumV1-S02BG 云端验证与待测评交接

更新时间：2026-08-05（Asia/Shanghai）

## 1. 状态

S02BG 是 S02BF 的严格后继候选：保留尾部单例连续路由、空归约 tiling
key 13 和全部既有路径，只将 workspace 中间连续归约的 FP32 完整宽度 8
从逐行 DMA 改为一个连续整块 DMA。

它已在 Ascend 910B4、CANN 社区版 8.5.0 上完成独立构建、安装、
`1295/1295` 正确性门禁、相邻安装 A/B 和提交包身份审计，可以作为
S02BF 之后的下一次官方测评候选。

当前官方对比基线仍为 S02BA 的 `3985.330 us`，目标仍为
`1195.599 us`。没有 S02BG 官方 Case1～Case5 前，不宣称达到目标或前十。

## 2. 修改与安全边界

S02BF 的 `ProcessParallelMiddle` 已对 FP16/BF16 的完整宽度 8 使用紧凑
整块搬运，但通过 dtype 判断排除了 FP32。FP32 每行 `8 × 4 = 32B`，天然
满足 DMA 和 UB 向量地址对齐；逐行搬运没有语义必要。

S02BG 只删除这一项 FP32 排除条件：

```text
compactFullInner8 =
    innerElements == 8 && innerIndex == 0 && current == 8
```

Host、TilingData、tiling key、blockDim、workspace 大小、归约顺序、平方
语义和输出转换全部不变。该条件只依赖通用布局属性，不依赖 Case 编号、
固定 shape、输入值或隐藏数据。

静态审计确认：

```text
SOURCE_SCOPE=PASS
PROMOTED_DTYPE=FP32
PROMOTED_INNER_WIDTH=8
TREE_ALIGNMENT_CHECKS=42
SUMMARY passed
```

NORMAL_CHUNK 和 LONG_CHUNK 下，整块 DMA 字节数均为 32B 的整数倍；
树归约每一级 FP32 源偏移也都是 32B 的整数倍。宽度 2/4 不满足最后一级
对齐条件，没有被顺带推广。

候选源码：

```text
candidates/squaresum_s02bg_fp32_middle8_compact_20260805/SquareSumV1
```

## 3. 相邻安装 A/B

在同一台 910B4 上先安装 S02BF，跑完同一 27 项矩阵；随即安装 S02BG
复跑。每项预热后用 NPU Event 取 7 组样本中位数，并同时比较 CPU 参考
结果。两版均为 `27/27` 正确。

FP32 宽度 8：

| 公开布局 | S02BF（us） | S02BG（us） | 加速 |
| --- | ---: | ---: | ---: |
| reduce=8192, outputs=8 | 54.641 | 54.382 | 1.005x |
| reduce=32768, outputs=8 | 37.048 | 30.146 | 1.229x |
| reduce=100000, outputs=8 | 74.106 | 32.574 | 2.275x |
| reduce=200000, outputs=8 | 136.695 | 36.943 | 3.700x |
| reduce=1000000, outputs=8 | 627.508 | 52.468 | 11.960x |
| reduce=200000, plain outputs=8 | 136.475 | 33.629 | 4.058x |
| 两个外层组、总 outputs=16 | 258.560 | 42.128 | 6.137x |

7 个目标布局的最小/中位/最大加速为 `1.005x / 3.700x / 11.960x`。

真正未修改的宽度 4/16 六个控制项最大绝对波动为 `1.149%`；其中唯一
轻微变慢项是 FP32 宽度 16：`103.258 → 103.390 us`（`+0.128%`）。

## 4. 正确性门禁

S02BG 安装后独立通过：

```text
定向矩阵                         46/46
随机矩阵                        150/150
BF16 严格语义                     4/4
扩展边界矩阵                    726/726
关键路径门禁                      45/45
宽度 8 边界专项                   60/60
单例间隔语义专项                  72/72
单例中间交叉矩阵                  30/30
单例间隔性能矩阵                  15/15
宽度 8 性能矩阵                   27/27
尾部单例与空归约定向矩阵          120/120
合计                            1295/1295
```

宽度 8 专项覆盖 workspace 门槛前、门槛点和门槛后，奇数归约长度，
多外层输出组，正轴、负轴、乱序轴，`keep_dims` 真/假，三种 dtype，
以及宽度 7/9/16 控制路径。

## 5. 构建与提交包

云端实验：

```text
/home/ma-user/work/s9/experiments/squaresum_s02bg_cloud_20260805_1900
```

云端发布：

```text
/home/ma-user/work/s9/releases/squaresum_s02bg_20260805_1906
```

本地 ZIP：

```text
D:\29722\Desktop\GCC\提交相关材料\20260805\S02BG\SquareSumV1.zip
大小：594087 bytes
SHA-256：D3BC2703620C076EC35A173C558926E9D76868E3F05D1BE12BCA090BFA6D0247
```

ZIP 根目录为 `SquareSumV1_zip/`，仅包含 `op_host/`、`op_kernel/` 和
`custom_opp_euleros_aarch64.run`。包内五份源码与 S02BG 候选逐文件一致；
包内 `.run` 与云端构建、安装并完成全部验证的二进制一致：

```text
RUN SHA-256：58B03731E7E7D1EF83DB913CE9E31D71BB91BC14571F4EBBD2F64661C0DAFE40
```

源码提交：`b6124ae`。S02BF ZIP 和冻结的 `submission-src/` 均未覆盖。

## 6. 官方测评结果

```text
Case1: Pass, Result: 6.500
Case2: Pass, Result: 399.998
Case3: Pass, Result: 138.4125
Case4: Pass, Result: 2553.771
Case5: Pass, Result: 896.918
prof_sum: 3995.5995
```

相对 S02BA 的五项差值依次为 `-0.010 / +0.910 / -0.3405 /
+3.010 / +6.700 us`，合计变慢 `10.2695 us`（`+0.258%`）。相对 S02BF
合计改善 `5.6005 us`，但仍未优于 S02BA，也远未达到 `1195.599 us`。
这证明宽度 8 紧凑搬运没有命中官方主耗时；S02BG 不晋级，后续转向
小输出、大规模非连续归约的 split-K/workspace 算法。
