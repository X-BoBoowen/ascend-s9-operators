# SquareSumV1-S02CD 云端验证与待测评交接

更新时间：2026-08-06（Asia/Shanghai）

## 1. 结论

S02CD 是 S02CA 的结构性后继，目标是消除 fastPath4 中窄 inner 布局的
填充放大和大量小 DMA。它不是针对平台 Case 的尺寸分支；路由只依赖
shape、axis、物理 stride、dtype、输入/输出规模和 UB 容量。

在 Ascend 910B4、CANN 社区版 8.5.0 上，最终合并版通过：

```text
S02BT 自适应边界          69/69
S02CA split-K 边界        36/36
S02CD inner2 专项边界     42/42
最终合并回归             147/147
42 点全路径图谱           42/42
```

正式平台结果尚未返回，因此不能据公开图谱宣称已经达到
`0.30 × T_baseline = 1195.599 us`。

## 2. 实现

S02CD 增加三条紧凑数据通路：

- fastPath4、`inner=2`、短 2 次幂末级归约：按输出任务分核，将多个物理
  连续输出行整块搬入 UB，不再把每行补齐到 8/16 元素；
- fastPath4、`inner=2`、末级归约 128～4096：40 核按外层归约组做
  split-K，每核写 FP32 partial 后统一归并；
- fastPath4、`inner=4/8/16` 的大规模 split-K：把逐归约行的小 DMA
  合并为连续整块 DMA，并在紧凑 FP32 布局中完成树归约。

`lastReduce=2, inner=2` 的 FP32 中间行跨度只有 16B，不满足下一行
32B 向量起址要求，因此明确回退到原填充路径。开发中曾因遗漏这一门限
触发 AICore 异常；修复后原失败用例 3/3 通过，并重新完成全部回归。

源码：

```text
candidates/squaresum_s02bu_strided_grouped_splitk_20260806/SquareSumV1
```

最终源码提交：`527a39e`。

## 3. 性能证据

相同 42 点跨 fastPath 图谱的中位耗时统计：

| 版本 | 42 点中位总和（us） | 最大单点（us） |
| --- | ---: | ---: |
| S02CA | 16965.913 | 3291.287 |
| S02CC | 5230.620 | 763.473 |
| S02CD | 3459.513 | 152.547 |

S02CD 相对 S02CA 的图谱总和改善 `4.904x`，最大单点改善 `21.575x`。
关键公开 4M 输入结构的最终中位耗时包括：

```text
inner2 grouped: FP16 116.193 / FP32 120.260 us
inner2 long:    FP16  88.867 / FP32  69.680 us
inner8 split-K: FP16  96.740 / FP32  91.407 us
inner16 split-K:FP16  92.480 / FP32  71.747 us
rank5 split-K:  FP16  74.840 / FP32  71.993 us
```

图谱中 42 个点全部低于 `153 us`。这些是公开代理性能证据，不代表隐藏
Case 的布局或正式得分。

## 4. 正确性覆盖

新增 inner2 专项覆盖：

- 末级归约长度 128、256、512、1024、2048、4096；
- 分组宽度 8、4、2、1 和输出尾块；
- 外层归约组 39、40、41、64；
- 输出门限内外、非 2 次幂回退；
- rank4/rank5、多个外层输出行、负轴、乱序轴和 keep_dims；
- FP16、BF16、FP32。

此外重新执行 S02BT/S02CA 的 105 项既有边界，最终合计 `147/147`。

## 5. 构建与提交包

云端实验：

```text
/home/ma-user/work/s9/experiments/squaresum_s02cd_power2_compact_20260806_1543
```

云端发布快照：

```text
/home/ma-user/work/s9/releases/squaresum_s02cd_20260806_1550
```

本地提交包：

```text
D:\29722\Desktop\GCC\提交相关材料\20260806\S02CD\SquareSumV1.zip
大小：829902 bytes
ZIP SHA-256：5D847711252AE49F5CA6BA8B65C14983BE703D4AE0493D323B5B61A90EF79D30
RUN SHA-256：39D38CBA228B62E3261A823CFE9E9394E632B21A83E67CEA63531F48299E2489
```

ZIP 只有一个顶层 `SquareSumV1_zip/`，共 6 个文件；`.run` 权限为
`0755`。5 份源码与仓库逐字节一致，云端和本地 ZIP 哈希一致。

## 6. 官方结果上下文

```text
S02BS: 7.416 / 407.388 / 139.676 / 2538.380 / 903.168
prof_sum = 3996.028 us

S02BT: 7.448 / 395.728 / 139.240 / 2548.652 / 904.312
prof_sum = 3995.380 us

S02CA: 6.410 / 390.238 / 138.633 / 2557.111 / 890.177
prof_sum = 3982.569 us
```

三者均没有形成相对 S02BA `3985.330 us` 的有效稳定突破。S02CD 需要
新的五 Case 正式结果；只有五项全部 Pass 且稳定中位总和不超过
`1195.599 us`，才能判定目标完成。
