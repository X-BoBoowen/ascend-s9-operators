# SquareSumV1-S02BF 云端验证与待测评交接

更新时间：2026-08-05（Asia/Shanghai）

## 1. 结论

S02BF 已在 Ascend 910B4、CANN 社区版 8.5.0 上完成独立构建、安装、
正确性门禁、性能回归和提交包审计，可以上传官方平台测评。

这不是线上性能结论。当前官方对比基线仍为 S02BA：

```text
Case1 6.510
Case2 399.088
Case3 138.753
Case4 2550.761
Case5 890.218
prof_sum 3985.330 us
目标 1195.599 us（0.30 × T_baseline）
```

S02BB 官方结果为 `4008.576 us`，未晋级。S02BF 是否击中隐藏慢路，
只能由本次官方 Case1～Case5 结果证明。

## 2. 优化内容

### 2.1 尾部单例连续路由

当 Host 已识别为中间连续归约，并且归约区域之后的所有保留维长度都为
1 时，物理内存中的每个输出仍对应一个连续归约区间。S02BF 将这种通用
布局从 `fastPath=2` 改道到 `fastPath=1`，复用成熟的连续末轴实现。

该条件只依赖公开的 shape/axis 布局属性，不依赖 Case 编号、隐藏 shape、
固定数据值或 dtype。宽度 2/4 等非单例控制布局不改道。

### 2.2 空归约正确性与零热路径开销

定向测试发现历史实现对“归约维为 0、输出非空”的情况没有写入输出，
可能返回未初始化值。S02BE 曾在所有 kernel 入口增加运行时判断并正确
置零，但 60 个非空目标布局的中位耗时相对 S02BD 波动慢约 4.14%。

S02BF 为 `reduceElements == 0` 分配独立 tiling key 13，并用编译期模板
特化执行置零；其余 tiling key 不再包含空归约运行时分支。这样同时保留
空输入正确性和 S02BD 非空热路径。

候选源码：

```text
candidates/squaresum_s02bf_empty_tiling_key_20260805/SquareSumV1
```

## 3. 静态审计

S02BD 路由审计覆盖 72,168 个 rank/shape/axis/keepDims 组合：

```text
PROMOTED=1644
PROMOTED_KEEP_DIMS_EQUIVALENTS=822
ZERO_REDUCE_RETAINED=698
SUMMARY passed
```

S02BE 空归约调度审计：

```text
EMPTY_LAYOUTS=17841
EMPTY_OUTPUT_LAYOUTS_SKIPPED=7564
SCHEDULED_OUTPUTS=44681
LARGE_SCHEDULE_CHECKS=112
SUMMARY passed
```

S02BF 额外确认修改范围、tiling key 13 和非空路径无运行时空归约分支：

```text
SOURCE_SCOPE=PASS
EMPTY_TILING_KEY=13
NONEMPTY_RUNTIME_EMPTY_BRANCHES=0
SUMMARY passed
```

## 4. 云端正确性

S02BF 安装后独立通过：

```text
定向矩阵                         46/46
随机矩阵                        150/150
BF16 严格语义                     4/4
扩展边界矩阵                    726/726
关键路径门禁                      45/45
宽度 8 紧凑搬运专项               60/60
单例间隔语义专项                  72/72
单例中间交叉矩阵                  30/30
单例间隔性能矩阵                  15/15
宽度 8 性能矩阵                   27/27
尾部单例与空归约定向矩阵          120/120
合计门禁调用                    1295/1295
尾部单例性能矩阵（同时验正确性）    66/66
```

空归约覆盖 FP16、BF16、FP32，`keep_dims=true/false`，空输出、非空
输出、完整归约、中间轴归约和带单例间隔的多轴归约。

## 5. 云端性能证据

测量环境为同一台 Ascend 910B4、CANN 8.5.0；每项预热后用 NPU Event
取 7 组样本中位数，并同时与 CPU 参考结果比较。

S02BB 到 S02BF 的 60 个目标布局全部正确：

```text
最小加速 1.03x
中位加速 29.62x
最大加速 234.14x
6 个宽度 2/4 控制项最大绝对波动 2.37%
```

代表性结果：

| 布局 / dtype | S02BB（us） | S02BF（us） | 加速 |
| --- | ---: | ---: | ---: |
| `(6,8193,1)`, axis=1, FP16 | 6097.86 | 32.67 | 186.65x |
| `(1024,8193,1)`, axis=1, FP16 | 12664.96 | 54.09 | 234.14x |
| `(3,131,1,251,1)`, axis=(1,3), FP16 | 2978.62 | 25.11 | 118.61x |
| `(262145,1,1)`, axis=0, FP16 | 344.94 | 29.12 | 11.85x |
| `(131072,2)`, axis=0, FP16 控制 | 175.29 | 175.41 | 1.00x |
| `(65536,4)`, axis=0, FP16 控制 | 90.42 | 89.07 | 1.02x |

S02BF 相对没有入口分支的 S02BD，60 个目标布局中位差为 `-0.19%`；
说明独立 tiling key 已恢复非空热路径。小算例单项仍有明显频率噪声，
不能把任何公开布局等同于隐藏 Case。

## 6. 构建、包与身份

云端实验目录：

```text
/home/ma-user/work/s9/experiments/squaresum_s02bf_cloud_20260805_1842
```

云端发布目录：

```text
/home/ma-user/work/s9/releases/squaresum_s02bf_20260805_1850
```

本地待提交 ZIP：

```text
D:\29722\Desktop\GCC\提交相关材料\20260805\S02BF\SquareSumV1.zip
大小：594583 bytes
SHA-256：B5484C12895CFF3165D865A3E557FCE321221BD5C09FB66DE1C4916A6B35196B
```

ZIP 根目录为 `SquareSumV1_zip/`，只含 `op_host/`、`op_kernel/` 和
`custom_opp_euleros_aarch64.run`。包内五份源码与 S02BF 候选逐文件
SHA-256 一致；包内 `.run` 与云端构建、安装并完成上述验证的文件一致：

```text
RUN SHA-256：A6F489BAA532308882A355F59B864E9621B5386E12830D18F79D52EBCCF284EF
```

对应 Git main 提交：`2710c93`。历史 ZIP 和冻结的 `submission-src/`
均未覆盖，收到官方结果前保留回退基线。

## 7. 官方测评结果

```text
Case1: Pass, Result: 7.908
Case2: Pass, Result: 407.248
Case3: Pass, Result: 139.776
Case4: Pass, Result: 2541.768
Case5: Pass, Result: 904.500
prof_sum: 4001.200
```

相对 S02BA 的五项差值依次为 `+1.398 / +8.160 / +1.023 /
-8.993 / +14.282 us`，合计变慢 `15.870 us`（`+0.398%`）。五项正确性
全部通过，但公开布局的尾部单例收益没有转化为官方总耗时改善；S02BF
不晋级，正式官方基线仍为 S02BA 的 `3985.330 us`。
