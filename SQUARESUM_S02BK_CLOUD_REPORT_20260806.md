# SquareSumV1-S02BK 云端验证与待测评交接

更新时间：2026-08-06（Asia/Shanghai）

## 1. 结论

S02BK 是 S02BI 的严格后继候选。它保留 S02BI 已验证的所有连续、
分组向量、strided、空归约和 fastPath3 split-K 路径，只为 fastPath2
中 `1 <= innerElements < 8` 的 workspace 归约增加对齐的 packed-phase
路径。

该候选已在 Ascend 910B4、CANN 社区版 8.5.0 上原生构建并正确安装，
通过 `1370/1370` 既有全量门禁以及 `45/45` 小宽度专项门禁。相邻安装
A/B 中，专项 45 项中位数合计从 `4676.563 us` 降至 `2661.409 us`，
约 `1.757x`。控制组基本保持原性能级别；正式平台五个 Case 尚未返回，
因此不能宣称达到榜单目标。

## 2. 问题定位与实现

S02BI 仅在 `innerElements == 8` 时把完整中间行紧凑搬入 UB。对
`innerElements == 1..7`，旧路径会把每个很短的 FP32 行分别填充到
32 字节，导致最多约 8 倍的搬运与 UB 占用膨胀。

直接紧凑搬运的 S02BJ 虽通过静态模型，但在 `innerElements == 2` 真机
触发 `UB address accessed by VEC instruction is not aligned`。原因不是 GM
DMA，而是树归约中 `halfRows * innerElements` 形成了非 32B 对齐的 VEC
源地址。S02BJ 已拒绝且未打包，云端安装也在失败后恢复为 S02BI。

S02BK 采用通用 packed-phase 设计：

1. 计算最小二次幂 `phaseRows`，使
   `phaseRows * innerElements` 为 8 个 FP32 元素的整数倍；
2. 紧凑搬运并平方多个完整 phase，在对齐的 phase span 上归约；
3. 每核把 phase 部分和写入独立 FP32 scratch；
4. 以每行 32B padding 重新读入少量 phase 行，执行地址对齐的最终树归约；
5. 把结果写回原有 per-core partial workspace，复用既有多核最终归约。

Host 只在 `fastPath == 2`、workspace reduce 且 `1 <= innerElements < 8`
时增加 scratch 工作区；所有大小计算继续使用溢出检查。路由只依赖 shape、
axis、dtype 和布局，不依赖 Case 编号、输入值或隐藏数据。

候选源码：

```text
candidates/squaresum_s02bk_packed_small_middle_20260806/SquareSumV1
```

源码提交：`562900b`。

## 3. 真机相邻安装 A/B

同一台 910B4 上先运行已恢复的 S02BI，再安装 S02BK。每项先与 PyTorch
结果比较，随后预热并取 7 组 NPU Event 样本中位数。

```text
records:             45
S02BI median sum:    4676.563 us
S02BK median sum:    2661.409 us
aggregate speedup:   1.757x
best speedup:        6.274x
```

代表性结果：

| 布局 | dtype | S02BI（us） | S02BK（us） | 加速 |
| --- | --- | ---: | ---: | ---: |
| inner=2, tail | FP16 | 177.215 | 28.711 | 6.172x |
| inner=2, tail | BF16 | 180.176 | 28.720 | 6.274x |
| inner=2, tail | FP32 | 148.491 | 27.637 | 5.373x |
| inner=3, tail | BF16 | 127.029 | 31.626 | 4.017x |
| outer=3, inner=4 | FP32 | 145.684 | 28.813 | 5.056x |
| outer=5, inner=7 | FP16 | 77.495 | 29.452 | 2.631x |
| singleton gap, inner=4 | FP32 | 104.369 | 29.341 | 3.557x |
| keepdims, inner=4 | FP32 | 104.895 | 28.107 | 3.732x |

`inner=9` 与输入量低于 fastPath2 阈值的控制组保持原性能级别。相邻扫描
中未走新路径的 `inner=1/8` 有约 5%–20% 的单轮波动，源码路由和对应
内核均未改变；不把这类差值视为代码收益或退化。

## 4. 正确性门禁

```text
定向矩阵                          46/46
随机矩阵                         150/150
BF16 严格语义                      4/4
扩展边界矩阵                     726/726
关键路径门禁                       45/45
宽度 8 边界专项                    60/60
单例间隔语义专项                   72/72
尾部单例与空归约定向矩阵           120/120
单例中间交叉矩阵                   30/30
单例间隔性能矩阵                   15/15
宽度 8 性能矩阵                    27/27
既有全量合计                     1295/1295
fastPath3 split-K 专项             39/39
非连续跨路径边界专项               36/36
既有门禁累计                     1370/1370
S02BK 小宽度专项                   45/45
本轮全部证据                     1415/1415
```

## 5. 构建与发布

云端实验：

```text
/home/ma-user/work/s9/experiments/squaresum_s02bk_cloud_20260806_1152
```

云端发布：

```text
/home/ma-user/work/s9/releases/squaresum_s02bk_20260806_1203
```

实际安装并完成全部门禁的 RUN SHA-256：

```text
7F5253C14823DC7C9215862CF3F90F59FE2FF57A108C6947E8E36597A4F73A5A
```

## 6. 提交包

```text
D:\29722\Desktop\GCC\提交相关材料\20260806\S02BK\SquareSumV1.zip
大小：645882 bytes
SHA-256：D283DB77BC37B46B64FD7E9EF0CCFA22BA8DD42397A622378C6674B69A7FACA1
```

ZIP 只有一个顶层 `SquareSumV1_zip/`，其中包含五份 host/kernel 源文件
和一个 `.run`，共 6 个文件。下载后已逐文件核对：五份源码与 S02BK
候选一致，`.run` 与云端实际安装并通过 `1415/1415` 的二进制一致。

## 7. 下一步

上传本报告指定的 S02BK ZIP，记录官方 Case1–Case5 的每项耗时。正式
基线仍为 S02BA `3985.330 us`，目标为 `1195.599 us`。只有官方结果能
判断小宽度路径是否命中隐藏主耗时；若仍未显著改善，继续基于通用布局
和真机 profiling 优化其他未覆盖路径，不反推或硬编码隐藏 shape。

## 8. 官方结果与结论（2026-08-06）

```text
Case1: Pass, Result: 6.570
Case2: Pass, Result: 389.5275
Case3: Pass, Result: 138.733
Case4: Pass, Result: 2554.151
Case5: Pass, Result: 895.818
prof_sum: 3984.7995
```

相对 S02BA `3985.330 us`，S02BK 合计只改善 `0.5305 us（0.013%）`；
五项变化为 `+0.060 / -9.5605 / -0.020 / +3.390 / +5.600 us`。
Case4/5 退化，合计差异远小于同包复测波动，因此 S02BK 判定未晋级。
这说明“已经满足旧 workspace 门槛的 fastPath2 小 inner”没有命中隐藏
主耗时。后继 S02BM 转而修复 S02BK 未覆盖的门槛以下单核性能断崖。
