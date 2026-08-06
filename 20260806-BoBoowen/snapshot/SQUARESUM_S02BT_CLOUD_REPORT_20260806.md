# SquareSumV1-S02BT 云端验证与待测评交接

更新时间：2026-08-06（Asia/Shanghai）

## 1. 结论

S02BT 是 S02BS 的严格后继，继续保留此前已经通过回归的连续归约、
workspace、fastPath3 split-K 和 fastPath4 `inner=1` 路径，并扩展优化
fastPath4 中 `inner=2～16` 的窄内层交错归约。

新实现把多个物理连续输出行合并为一个任务，以一次 `DataCopyPad` 搬运
多行末级归约片段；每个 inner 行在 UB 中补齐到 32 字节，再使用对齐的
树形向量加法归约。归约结果先保存在对齐的 FP32 临时区，最后用标量
读写安全地紧密打包，避免非 32 字节对齐的向量目标地址。

同机 33 点路径矩阵中，S02BS 合计 `11680.390 us`，最终 S02BT 复测合计
`7045.946 us`，加速 `1.658x`。其中 `inner=2/4/8/16` 三种 dtype 合计
分别加速 `1.507x / 1.713x / 2.118x / 1.316x`。最终候选在 Ascend
910B4、CANN 社区版 8.5.0 上构建、安装，并通过累计 `1634/1634`
门禁。

正式平台五个 Case 尚未返回，因此不宣称达到排行榜目标。官方比较基线
仍为 S02BA 的 `3985.330 us`，目标为 `1195.599 us`。

## 2. 通用路由

S02BT 新路径同时满足：

```text
fastPath == 4
reduceMode == 0
2 <= innerElements <= 16
末级归约长度是 2、4、8、16、32 或 64
存在物理步长等于“末级归约长度 × inner”的保留输出维
分组后的 UB 输入不超过 8192 个元素
inner <= 8 时至少形成 32 个任务
inner >= 9 时至少形成 4 个任务
```

Host 按缓冲区容量和任务数从 `8/4/2/1` 中选择最大分组宽度，并复用
构建系统已经包含的 tiling key `5/9/10/11`。分组不会跨越保留输出维
边界，非整除尾部由 `activeRows` 处理。条件只依赖 shape、axis、dtype、
物理 stride 和硬件容量，不依赖 Case 编号、输入值或隐藏数据。

源码：

```text
candidates/squaresum_s02bt_strided_grouped_padded_rows_20260806/SquareSumV1
```

最终调度提交：`91044b4`；对齐输出修复：`1a18e3d`；边界测试：
`88102e6`、`b4979fd`。

## 3. 性能证据

测试布局为通用交错归约，输入元素数均为 1M：

```text
shape=(64, group, 64, inner), axes=(0, 2)
group × inner = 256
```

| 路径（三种 dtype 合计） | S02BS（us） | S02BT（us） | 加速 |
| --- | ---: | ---: | ---: |
| inner=2 | 3789.500 | 2513.948 | 1.507x |
| inner=4 | 2468.344 | 1440.856 | 1.713x |
| inner=8 | 1266.528 | 597.860 | 2.118x |
| inner=16 | 697.888 | 530.192 | 1.316x |
| 33 点完整矩阵 | 11680.390 | 7045.946 | 1.658x |

`inner=1` 源码路径与 S02BS 相同；两次最终运行合计分别为
`141.228 us` 和 `160.096 us`，S02BS 记录为 `140.604 us`，表明该项
受当轮机器波动影响但没有新增算法开销。控制项也出现约 10% 级正负波动，
因此只把 `inner=2/4/8/16` 的大幅、结构性下降视为有效证据。

## 4. 正确性门禁

```text
主矩阵：定向、cloud gate、随机、BF16、扩展       971/971
既有专项：结构、middle8、单例、workspace 等       561/561
S02BT 自适应分组边界                              69/69
S02BT 路径矩阵                                    33/33
累计                                             1634/1634
```

新增 69 项覆盖：

- 末级归约长度 `2/4/8/16/32/64`；
- inner `2/3/4/5/7/8/9/15/16`；
- 分组宽度 `8/4/2/1` 与非整除尾组；
- 多外层输出行和多个外层归约组；
- `keep_dims=True/False`、负轴和乱序轴；
- FP16、BF16、FP32。

## 5. 失败中间版本

- 首个 S02BT 构建可以通过，但直接用向量 `Cast/Adds` 写紧密输出时，
  第二行起目标地址不一定按 32 字节对齐，设备报告
  `UB address accessed by the VEC instruction is not aligned`。该安装立即
  回滚到 S02BS，没有打包。最终版本改用 V→S 同步、标量紧密打包和
  S→MTE3 同步，随后 33/33 与 69/69 全过。
- S02BT3 曾把所有 inner 的最低任务数统一降到 4。`inner=16` 有收益，
  但 `inner=2/4/8` 明显回退，因此废弃。最终版本按 inner 宽度选择最低
  并行任务数：窄 inner 保持 32，较宽 inner 允许 4。

失败中间版本均不作为提交包。云端当前安装的是最终 S02BT4。

## 6. 构建、安装与提交包

云端最终实验：

```text
/home/ma-user/work/s9/experiments/squaresum_s02bt4_cloud_20260806_1416
```

云端发布快照：

```text
/home/ma-user/work/s9/releases/squaresum_s02bt_20260806_1420
```

实际安装并完成最终门禁的 RUN SHA-256：

```text
87D7C68EC6F8C39D133C6BA5B4256C3F0D626A83755A27755EFC170344519D72
```

提交包：

```text
D:\29722\Desktop\GCC\提交相关材料\20260806\S02BT\SquareSumV1.zip
大小：713017 bytes
SHA-256：5E1AAEC27C8465B44695130C6985D873C49ACD807A6AD76CE8FF31C19D3DCCF4
```

ZIP 只有一个顶层 `SquareSumV1_zip/`，包含五份 host/kernel 源文件和
一个 `.run`，共 6 个文件；ZIP 内 `.run` 权限为 `0755`。本地 ZIP 与
云端发布 ZIP 哈希一致，ZIP 中五份源码与候选源码逐字节一致。

## 7. 下一步

上传上述 S02BT ZIP，记录官方 Case1～Case5。只有五项全 Pass 且官方
`prof_sum` 相对 S02BA/S02BK 形成明显、可复现的下降，才把 S02BT 晋级
为新的正式基线。若官方仍接近 `3985 us`，说明隐藏主耗时仍不在这一类
fastPath4 窄 inner 布局，应转向其它通用路径的 profiling，而不是猜测
隐藏 shape。
