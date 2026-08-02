# SquareSumV1 S02AL 本地优化交付审计

日期：2026-08-02

## 1. 结论

本轮已完成当前本机能够可靠完成的源码优化、边界穷举、数值模拟和历史
回归。全程未连接华为云，未消耗 NPU 时长，也未修改正式提交源码和正式
ZIP。

当前应进入云端裁决的两个最新版为：

```text
S02AK  FP32 小输出使用 workspace + tree
candidates/squaresum_s02ak_grouped_short_narrow_20260802/SquareSumV1

S02AL  与 S02AK 内核字节完全相同，仅 FP32 小输出恢复 atomic
candidates/squaresum_s02al_atomic_control_20260802/SquareSumV1
```

S02AK 与 S02AL 不是已经完成的正式提交版本。它们必须先经过 CANN 8.5.0
编译、910B4 正确性和真实耗时验证，才能选择胜者并更新 `submission-src/`
与 ZIP。

## 2. 性能背景与成功标准

最近一次正式测评：

```text
Case1   6.530 us
Case2 394.308 us
Case3 237.885 us
Case4 1716.7145 us
Case5 868.5575 us
sum    3223.995 us
```

当时第 10 名总耗时为 `1268.598 us`。本轮没有用本地模型冒充真实性能；
最终是否改善以及改善幅度，只能由同一云端环境的 A/B 数据裁决。

## 3. 本轮新增候选链

| 候选 | 单一作用 | 本地结论 |
| --- | --- | --- |
| S02AB | 并行末轴每核 1 次连续 workspace 写回，替代逐输出写回 | 布局数值一致；写回模型显著下降 |
| S02AC | 小输出 workspace 全部使用已有树形收尾 | 2,400 个数值样例通过 |
| S02AD | 修复 FP32 长块树形收尾的 UB 容量不匹配 | 194,592 / 196,608 字节；余量 2,016 字节 |
| S02AE | FP32 末轴小输出由 atomic 改为 workspace tree | 独立实验假设；需 S02AL 对照裁决 |
| S02AF | 树形收尾直接从树根输出 | 每个 final tile 删除清零和加零；1,500 组位级一致 |
| S02AG | 中轴、跨轴和 grouped 首个 partial 直接初始化 | 删除每个输出 tile 一次清零；105 组位级一致 |
| S02AH | 连续末轴首个 partial 直接初始化 | 60 组位级一致 |
| S02AI | grouped 长尾/平铺首个 partial 直接初始化 | 135 组位级一致 |
| S02AJ | 顺序 finalizer 首块直接初始化 | 750 组位级一致 |
| S02AK | 补齐 grouped short-tail、物理输出宽 1–7 的 width 1/2/4 路径 | 40,320 组穷举；所选 DMA 比 0.832962 |
| S02AL | S02AK 的 atomic 对照 | 与 S02AK kernel 字节完全相同 |

S02AA 及之前的连续末轴、分段、中轴、长尾和 singleton-axis 优化仍全部
保留在 S02AK/S02AL 中。

## 4. 关键安全修复

S02AC 的 FP32、`LONG_CHUNK=16384`、tree-finalizer 路径原先只给
`floatBuffer_` 分配 32 字节，但 40 核补成 64 行后，256 输出 tile 最多
访问 65,536 字节。S02AD 将 FP32 长树 tile 限为 232，并分配
`64 * 232` 个 FP32 元素：

```text
outputBuffer       4,096 B
inputBuffer       65,536 B
floatBuffer       59,392 B
reduceWorkBuffer  65,536 B
sumBuffer             32 B
total            194,592 B
UB budget        196,608 B
headroom           2,016 B
```

232/233、464/465、1023/1024 等边界已经过 660 组数值和容量模拟。该修复
对 FP32 末轴输出 1–8 不增加 finalizer DMA；对 FP32 中轴恰好 233–256
输出时会从一次变为两次 finalizer DMA，因此真实设备仍需覆盖 232/233。

## 5. S02AK grouped short-narrow 覆盖

旧实现对以下类别退回逐输出 `ReduceSum -> V/S -> GetValue`：

```text
非连续多轴归约
包含最后一轴
trailing reduce 1..64
最内层物理保留维 1..7
```

S02AK 选择能整除物理输出维的最大 width 4/2/1，并只在新路径 DMA 数不高于
旧路径时启用。两种元素字节宽、输出维 1–7、tail 1–64、多种 batch 和外层
输出的 40,320 组穷举结果：

```text
width4 selected   5,545
width2 selected  10,750
width1 selected  18,940
fallback          5,085
old selected DMA  6,490,176
new selected DMA  5,406,072
ratio              0.832962
```

保留 fallback 的 5,085 组没有强行优化，原因是宽路径 DMA 可能增加。

另对 rank 2–5、全部合法轴组合、`keep_dims=true/false` 和 singleton 维做了
524,232 组 Host 路由审计，其中 181,872 组属于 fastPath3。新路径实际选择
width4/2/1 分别 53,952/53,952/72,432 次；90,168 次覆盖 keep_dims，
105,336 次覆盖含 singleton 的输入。所有选中任务均逐地址验证未跨输出行，
batch stride 与 GM 地址映射一致。

## 6. 回归证据

最新 S02AK 已把 S02G–S02V 的所有历史诊断重绑定后完整运行，结果：

```text
SUMMARY: all historical local regressions passed
```

覆盖内容包括：

- 连续末轴 1/2/4/8 输出并行；
- 32/64 分段阈值；
- grouped 短、中、长尾和 singleton 物理轴；
- 中轴与跨轴长块；
- 非对齐 DMA、stride 上限和 partial 容量；
- FP16、BF16、FP32 平方语义和 FP32 累加模型；
- workspace 顺序/树形 finalizer；
- S02AD UB 容量边界；
- S02AF–S02AJ 的位级等价首 partial 初始化；
- S02AK 40,320 组 short-narrow 选择穷举及 224 个数值样例。
- S02AK 524,232 组 rank/axis/keep_dims/singleton 路由与地址审计。

所有 S02AA–S02AL 新增 Python 工具均已通过 `py_compile`，最终候选差异通过
`git diff --no-index --check`。

## 7. 云端最小裁决顺序

云端启动后只执行以下顺序，所有命令继续通过 `ssh_visible.ps1`，保证用户在
JupyterLab 终端看得到：

1. 编译 S02AK；失败则记录完整 CANN 错误并停止性能测试。
2. S02AK 跑 `squaresum_s02al_cloud_gate_20260802.py correctness`，要求
   `45/45`。
3. 编译 S02AL，先只跑两个 `last_workspace_*` 路径的正确性。
4. S02AK/S02AL 只对 `last_workspace_output1,last_workspace_output8`
   做 A/B，裁决 FP32 workspace 与 atomic。
5. 编译 S02AJ；S02AJ/S02AK 只对 `short_width4,short_width2,short_width1,
   short_singleton` 做 A/B，裁决 S02AK 新分流。
6. 将两个局部胜者合并为一个最终候选，再跑完整正确性和完整性能矩阵。
7. 只有最终候选稳定胜过当前正式版本且精度全过，才更新正式源码、打 ZIP
   并做正式平台测评。

日志比较工具：

```text
diagnostics/squaresum_s02al_compare_cloud_logs_20260802.py
```

## 8. 哈希与正式版本隔离

S02AK：

```text
host    7C06E7F351ABA60ADF1B8D88BBA5DBEF53E88AC982BD289B531F700D6027C724
tiling  E669820AB05878A8701FE2E6F33F4B44076EE93E74C1A308C3B5E8FCF54079F6
kernel  49080CD96279CA0FD9B69F086B18770022DC314027273873577E83CFB4D45E52
cmake   3C8EEEE7087102BA8A8A8C4268D5FA6DC5D26B4BD442064E663C747D22F7533C
```

S02AL：

```text
host    BE42485C134BECB0E4298DB690E373466FFE92814D9C6A571C54D3EAAA259948
tiling  E669820AB05878A8701FE2E6F33F4B44076EE93E74C1A308C3B5E8FCF54079F6
kernel  49080CD96279CA0FD9B69F086B18770022DC314027273873577E83CFB4D45E52
cmake   3C8EEEE7087102BA8A8A8C4268D5FA6DC5D26B4BD442064E663C747D22F7533C
```

正式 ZIP 仍为：

```text
D:/29722/Desktop/GCC/提交相关材料/SquareSumV1.zip
EECD9F1FD6B4C0617B6EC2EC632F24BB0F310D5A9D2F2125F03F6EC86ECFAF5B
```

`submission-src/SquareSumV1` 在 Git 状态中没有修改。本轮没有重打 ZIP、
没有推送候选，也没有把任何未经过 910B4 验证的代码标记为正式完成。
