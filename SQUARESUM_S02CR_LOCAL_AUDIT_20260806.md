# SquareSumV1-S02CR 通用非连续 Split-K 本地审计

日期：2026-08-06

## 结论

S02CR 是从正式最好 S02CA `3982.569 us` 独立派生的结构候选，目标是
`fastPath=4` 中仍然存在的低并行度区域。它不是 S02CQ 的叠加版本，便于在
真机上分别判断 BF16 中间量化优化和通用非连续 Split-K 的真实收益。

当前只完成本地源码、路由、坐标映射、数值语义和云端入口审计，**尚未在
910B/CANN 8.5.0 构建或运行，禁止打包提交。**

## S02CA 全路径复核

Host 先把归约布局分为四类：

| fastPath | 布局 |
|---:|---|
| 1 | 连续末轴/连续全归约 |
| 2 | 连续中间轴，末尾保留连续 inner |
| 3 | 非连续多轴且包含末轴 |
| 4 | 非连续多轴且末轴保留 |

TilingKey 1～4 对应普通/长 chunk 与顺序/树形 finalizer；5～12 是 grouped
vector、last-axis vector 和不同向量宽度的专用实例；13 处理空归约。

S02CA 已经为 fastPath1/2 的小输出大归约、fastPath3 的部分长尾布局以及
fastPath4 的“小 2 的幂末级归约 + inner<=16”布局实现 workspace Split-K。
仍未覆盖的是：

```text
fastPath4
innerElements > 16（或不满足 grouped-row 专用布局）
outputElements 不大
reduceElements 很大
```

当前 host 对 `innerElements>16` 的普通 fastPath4 使用：

```text
blockDim = ceil(outputElements / 64)，最多 40
```

因此输出 64～256 个元素时可能只启动 1～4 个核。每个核都必须遍历完整
归约维；这类通用布局能够产生毫秒级长尾，符合 Case4 `~2550 us` 所暴露的
“单个路径远慢于其余 Case”现象。这里没有推断 Case4 的具体 shape 或 dtype，
只是消除题面合法域内可证明存在的低并行度区域。

## S02CR 算法

新增 `reduceMode=6`：

1. 扁平归约索引均匀切成 40 个连续区间；
2. 每核处理所有输出连续块，只累加自己的归约区间；
3. 区间内部按最后一个归约轴的自然行边界分段；
4. 每段使用一次多 block `DataCopyPad` 批量搬入多行；
5. UB 内先对多行平方，再做树形行归约；
6. 每核把 FP32 partial 写入独立 workspace 行；
7. `SyncAll` 后 0 核复用现有 40 行树 finalizer 输出结果。

第一版草稿曾逐 reduce index 发 DMA。源码复核发现它会丢掉基线已有的二维
批量搬运能力，因此在提交前已经删除；当前版本保留多行 DMA，并限制每批
最多 4095 行和 `CHUNK` 容量。

## 通用成本门禁

候选只在以下条件满足时选择新模式：

- fastPath4 且未命中已有 grouped-row 路径；
- `inputElements >= 2^18`；
- `reduceElements >= 2048`；
- `0 < outputElements <= 8192`；
- 输出可表示为完整 inner 行；
- 根据输出 chunk 数、旧核数和 40 核归约区间估算，新路径关键工作量至少
  比旧路径低 2 倍。

该模型没有 Case 编号、正式计时、隐藏 shape 或 dtype 分支。本地域内正例的
关键路径估算改善为约 `6.66×–20×`；连续 fastPath2、输入/归约阈值以下、
已有 grouped Split-K 和 workspace 上限外的控制用例均保持原路径。

## 本地证据

- S02CA host/kernel 哈希均与正式基线一致；
- S02CR 只修改 host 和 kernel，CMake 与 tiling 数据结构逐字节不变；
- 路由与源码契约 `12/12` 通过；
- Split-K 舍入模型覆盖 FP16/BF16/FP32、归约长度
  `2048/4097/32768/100000`、三类数值分布；
- 另外逐坐标模拟 rank-5 稀疏轴、无序轴、负轴和 keep_dims；
- 数值与坐标模型合计 `45/45` 通过赛事正式同类误差规则，FP32 最大观测
  相对差约 `1.2e-7`；
- 新增两个题面域性能点，完整矩阵现在为 `25/25`；
- Python 编译、shell 语法、危险输出路径拒绝和 `git diff --check` 均通过。

当前源码哈希：

```text
host   387F0433C2913DCEFA4928C1670B5BFAADC70CF893D629139B38EB1F059EBBB4
kernel 6FDB014A1F989E935C74063548F958E9740FFD9F9678967FB3228595C27C472A
```

这些是本地间接证据，不是 NPU 构建或性能结论。

## 云端门禁

云端开启后，在仓库根目录执行：

```bash
bash diagnostics/run_squaresum_s02cr_cloud_gate_20260806.sh \
  /absolute/path/to/official/SquareSumV1/project \
  /absolute/path/to/new/s02cr_gate_work_20260806
```

一次命令会完成：

1. S02CA/S02CR 从同一官方模板独立构建；
2. 两个不同 `--install-path` 隔离安装；
3. S02CR 六个边界布局 × 三 dtype，共 `18/18` 真机正确性；
4. S02CA/S02CR 两个目标布局 × 三 dtype 的 Event 发现图谱；
5. 所有目标布局和 dtype 的官方兼容 msprof A→B→A；
6. 基线漂移 `<=3%`、任一回退 `<=3%`、总改善 `>=10%` 才通过。

每次运行清理旧 `artifact/`；主结果与日志位于：

```text
artifact/result.json
artifact/run.log
```

构建或正确性提前失败也会原子写入失败阶段和退出码。只有
`artifact/result.json` 的 `passed=true` 才允许考虑正式包；通过目标矩阵
仍不等同于官方隐藏 Case 晋级。
