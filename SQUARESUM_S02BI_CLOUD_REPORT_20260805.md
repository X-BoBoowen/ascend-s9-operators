# SquareSumV1-S02BI 云端验证与待测评交接

更新时间：2026-08-05（Asia/Shanghai）

## 1. 状态

S02BI 是 S02BG 的严格后继候选。它保留 S02BG 的 13 个既有 tiling
入口和全部连续、分组向量、strided、空归约路径，只为一类已实测的
fastPath3 大归约增加 workspace split-K：多个 AIV 各自处理不重叠的
自然分组行，写 FP32 部分和，同步后由既有 LONG_CHUNK 树归约输出。

它已在 Ascend 910B4、CANN 社区版 8.5.0 上完成原生构建、正确安装、
`1295/1295` 全量门禁和 `75/75` split-K 专项门禁。正式官方基线仍为
S02BA 的 `3985.330 us`，目标仍为 `1195.599 us`；收到官方五个 Case
前不宣称 S02BI 晋级或达到目标。

## 2. 路由边界与合规性

新路由只在下列通用条件同时成立时启用：

```text
fastPath == 3
inputElements >= 262144
reduceElements >= 32768
1 <= outputElements <= 8
1024 <= trailingReduceElements <= 16384
naturalGroupedRows >= 16
分组尾部满足真实二维 DMA 步距、对齐和 uint32 stride 边界
```

尾宽 256、自然行数 8、输出数 9、阈值下方、物理上因单例维度重新合并
为连续布局的情况都不进入新路由；fastPath4 明确不进入。所有条件只依赖
shape、axis、dtype 和物理 stride，不依赖 Case 编号、输入值或隐藏数据。

每个核通过商余分配得到 `[firstRow, limitRow)`；跨外层分组时在真实
`batchDim` 边界截断二维 DMA，避免把 source gap 用到下一组。即使核没有
分到自然行，也会写入定义良好的零部分和。最终复用已登记的 tiling key 4
和原有 FP32 树归约，不新增运行时入口。

候选源码：

```text
candidates/squaresum_s02bi_fp3_splitk_20260805/SquareSumV1
```

对应源码提交：`b55d5b1`，入口修复提交：`2295cdc`。

## 3. 相邻安装 A/B

同一台 910B4 上先正确安装 S02BG，再安装 S02BI；每项先对照 PyTorch
结果，然后预热并取 7 组 NPU Event 样本中位数。39 项 fastPath3 扫描
全部正确，其中 33 个目标记录合计从 `8899.536 us` 降到
`1373.074 us`，约 `6.481x`；单项最小/中位/最大加速为
`1.202x / 4.269x / 19.984x`。

| 布局 | dtype | S02BG（us） | S02BI（us） | 加速 |
| --- | --- | ---: | ---: | ---: |
| rows=16, tail=4096, outputs=8 | FP16 | 73.340 | 40.854 | 1.795x |
| rows=16, tail=4096, outputs=8 | BF16 | 92.834 | 42.160 | 2.202x |
| rows=16, tail=4096, outputs=8 | FP32 | 72.894 | 42.800 | 1.703x |
| rows=128, tail=4096, outputs=8 | FP16 | 525.410 | 41.128 | 12.776x |
| rows=128, tail=4096, outputs=8 | BF16 | 676.424 | 40.984 | 16.505x |
| rows=128, tail=4096, outputs=8 | FP32 | 525.550 | 42.138 | 12.472x |
| rows=40, tail=16384, outputs=8 | FP16 | 654.144 | 41.426 | 15.789x |
| rows=40, tail=16384, outputs=8 | BF16 | 840.944 | 42.080 | 19.984x |
| rows=40, tail=16384, outputs=8 | FP32 | 649.398 | 40.948 | 15.859x |

另一套 36 项跨 fastPath3/4 边界矩阵也是 `36/36`。其中真正命中新路由
的 rank-5 输出 8 布局，FP16/BF16/FP32 分别从
`265.884 / 338.688 / 262.578 us` 降到
`42.922 / 42.188 / 44.460 us`，即约 `6.194x / 8.028x / 5.906x`；
全部 fastPath4 项保持原性能级别。

## 4. 正确性门禁

```text
定向矩阵                         46/46
随机矩阵                        150/150
BF16 严格语义                     4/4
扩展边界矩阵                    726/726
关键路径门禁                      45/45
宽度 8 边界专项                   60/60
单例间隔语义专项                  72/72
尾部单例与空归约定向矩阵          120/120
单例中间交叉矩阵                  30/30
单例间隔性能矩阵                  15/15
宽度 8 性能矩阵                   27/27
既有全量合计                    1295/1295
split-K 行数/尾宽专项             39/39
非连续跨路径边界专项              36/36
本轮累计                        1370/1370
```

## 5. 构建、安装与失败证据

云端实验：

```text
/home/ma-user/work/s9/experiments/squaresum_s02bi_cloud_20260805_2028
```

云端发布：

```text
/home/ma-user/work/s9/releases/squaresum_s02bi_20260805_2045
```

第一次构建曾选择新 tiling key 14。源码可以编译，但运行时入口表未登记
14，目标首项报 `BinaryGetFunctionByEntry failed, funcEntry=14`；该版本没有
产生可用目标性能数据，也未打包。随后改为复用语义匹配且已登记的 key 4，
重新独立构建、安装并完成上述全部门禁。另一次环境审计还发现，把安装路径
写到 `.../opp/vendors/customize` 会生成无效嵌套目录；最终所有有效 A/B 和
门禁均使用 `--install-path=.../cann-8.5.0/opp`，并核对外层文件时间戳。

最终 RUN SHA-256：

```text
BBBCD140B73DB45BD89C2E7B6D142FFC5345A917E47FCD8A25BBD8BD2145654E
```

## 6. 提交包

```text
D:\29722\Desktop\GCC\提交相关材料\20260805\S02BI\SquareSumV1.zip
大小：621737 bytes
SHA-256：BB3ECCDFBA6584EBAC4D7D3F5B86C34E5579C935BF775E8C5CA0BC9046B2A565
```

ZIP 根目录是 `SquareSumV1_zip/`，仅含五份 host/kernel 源文件和一个
`.run`，共 6 个文件。独立解压后逐文件核对：五份源码与 S02BI 候选一致，
`.run` 与云端实际安装并完成 `1370/1370` 验证的二进制一致。

## 7. 测评后动作

请只上传上述 S02BI ZIP，并返回 Case1～Case5 原始结果和 `prof_sum`。
只有官方结果能证明该通用 fastPath3 改进是否命中隐藏主耗时；若仍未达到
`1195.599 us`，继续依据真机 profiling 优化未覆盖路径，不反推或硬编码
隐藏 shape。
