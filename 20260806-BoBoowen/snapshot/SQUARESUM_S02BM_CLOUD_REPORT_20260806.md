# SquareSumV1-S02BM 云端验证与待测评交接

更新时间：2026-08-06（Asia/Shanghai）

## 1. 结论

S02BM 是 S02BK 的严格后继候选。S02BK 官方结果为 `3984.7995 us`，
相对正式基线 S02BA `3985.330 us` 只快 `0.5305 us（0.013%）`，属于
平台波动。真机全路径扫描随后发现：fastPath2 窄 inner 在输入量略低于
旧 workspace 门槛 `262144` 时退回少核/单核路径，FP16/BF16 带宽约
`0.16–1.09 GB/s`；跨过门槛后同类布局约 `27–33 us`，形成最高约
16 倍的通用路由断崖。

S02BM 保留 S02BK 的 packed-phase 内核，只按 inner 对 workspace 门槛
分段。它已在 Ascend 910B4、CANN 社区版 8.5.0 上原生构建、正确安装，
通过 `1370/1370` 既有门禁和 `102/102` 新阈值专项门禁，累计
`1472/1472`。正式平台五个 Case 尚未返回，因此不宣称达到榜单目标。

## 2. 路由设计

旧版统一要求：

```text
inputElements >= 262144
reduceElements >= 2048
```

S02BM 改为：

```text
fastPath2 && innerElements < 8  : inputElements >= 4096
fastPath2 && innerElements == 8 : inputElements >= 32768
其他路径                         : inputElements >= 262144
全部仍要求                        : reduceElements >= 2048
```

设计依据：

- `inner=2/4` 的旧路径即使在最小 `reduce=2048` 也受逐行 32B padding
  和低并行度拖累，packed workspace 从最小有效规模即有收益；
- `inner=8` 已经自然满足 FP32 32B 对齐，FP32 小规模旧路径约 24 us，
  因此保留更高的 32768 门槛，避免过早起 40 核；
- `inner>=16` 旧路径已约 18–23 us，不改变；
- fastPath1、fastPath3、fastPath4、`reduce<2048` 和宽 inner 均不改变。

所有条件只依赖 shape、axis、dtype 和布局，不依赖 Case 编号、输入值或
隐藏数据。候选源码：

```text
candidates/squaresum_s02bm_split_narrow_workspace_20260806/SquareSumV1
```

源码提交：`de566c8`。

## 3. 真机性能证据

### 3.1 完整阈值矩阵

54 项相邻版本矩阵：

```text
S02BK median sum: 7080.033 us
S02BM median sum: 1596.376 us
aggregate speedup: 4.435x
best speedup: 15.948x
```

代表性结果：

| 布局 | dtype | S02BK（us） | S02BM（us） | 加速 |
| --- | --- | ---: | ---: | ---: |
| inner=2, input=32768 | FP16 | 396.241 | 27.130 | 14.605x |
| inner=2, input=32768 | BF16 | 403.819 | 27.021 | 14.945x |
| inner=2, input=32768 | FP32 | 323.171 | 26.813 | 12.053x |
| inner=8, input=262136 | FP16 | 481.453 | 30.562 | 15.753x |
| inner=8, input=262136 | BF16 | 497.743 | 31.210 | 15.948x |
| inner=8, input=262136 | FP32 | 201.150 | 29.823 | 6.745x |
| outer=8, inner=8 | BF16 | 187.734 | 29.711 | 6.318x |

### 3.2 小规模交叉矩阵

48 项 `reduce=512…8192` 交叉矩阵：

```text
S02BK median sum: 3125.972 us
S02BM median sum: 1500.735 us
aggregate speedup: 2.083x
best speedup: 7.323x
```

`reduce<2048` 控制项没有进入新路径，变化约 1% 内。代表性收益：

| 布局 | dtype | S02BK（us） | S02BM（us） | 加速 |
| --- | --- | ---: | ---: | ---: |
| inner=2, reduce=8192 | BF16 | 204.860 | 27.974 | 7.323x |
| inner=2, reduce=4096 | FP16 | 102.641 | 28.289 | 3.628x |
| inner=4, reduce=4096 | BF16 | 89.790 | 27.951 | 3.212x |
| inner=8, reduce=4096 | FP16 | 64.627 | 28.209 | 2.291x |

未走新路由的宽 inner/fastPath1 控制项在不同扫描时段有设备调度波动，
源码与对应 kernel 路径均未改变，不把这些差值归因于 S02BM。

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
既有基础门禁                     1295/1295
fastPath3 split-K 专项             39/39
非连续跨路径边界专项               36/36
既有门禁累计                     1370/1370
workspace 完整阈值专项             54/54
workspace 小规模交叉专项           48/48
本轮全部证据                     1472/1472
```

## 5. 构建与发布

云端实验：

```text
/home/ma-user/work/s9/experiments/squaresum_s02bm_cloud_20260806_1216
```

云端发布：

```text
/home/ma-user/work/s9/releases/squaresum_s02bm_20260806_1224
```

实际安装并完成全部门禁的 RUN SHA-256：

```text
62B6F78E9592C44F7832BB81063521F292165B00F95432ED7D10F8ACE637EDDB
```

## 6. 提交包

```text
D:\29722\Desktop\GCC\提交相关材料\20260806\S02BM\SquareSumV1.zip
大小：646317 bytes
SHA-256：3893E200515214E6DA8D1F12693B797E5DD66645915A7C1FC3C0732F5C572678
```

ZIP 只有一个顶层 `SquareSumV1_zip/`，其中包含五份 host/kernel 源文件
和一个 `.run`，共 6 个文件。下载后已逐文件核对：五份源码与 S02BM
候选一致，`.run` 与云端实际安装并通过 `1472/1472` 的二进制一致。

## 7. 下一步

上传本报告指定的 S02BM ZIP，记录官方 Case1–Case5 每项耗时。正式基线
仍为 S02BA `3985.330 us`，目标为 `1195.599 us`。只有官方结果能判断
门槛以下窄 inner 是否命中隐藏主耗时；若仍未改善，继续扫描和优化其他
通用单核/低带宽路径，不反推或硬编码隐藏 shape。
