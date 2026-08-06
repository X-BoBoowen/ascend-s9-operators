# SquareSumV1-S02CM 云端验证与待测评交接

更新时间：2026-08-06（Asia/Shanghai）

## 1. 结论

S02CL 正式结果为：

```text
6.470 / 388.257 / 139.113 / 2558.891 / 893.908
prof_sum = 3986.639 us
```

它比 S02BA 基线 `3985.330 us` 慢 `1.309 us`（`+0.033%`），属于噪声
量级且方向为退化，正式淘汰。S02CM 不再调整 Host 门限，而是重写
fastPath4 非 32 字节对齐完整行的数据搬运和归约方式。

Ascend 910B4、CANN 社区版 8.5.0 最终验证：

```text
累计七组正确性回归              315/315
非对齐 inner 扫宽                 30/30
32M 大规模全 fastPath 图谱         26/26
4M 小规模全 fastPath 图谱          42/42
云端发布包审计                      PASS
```

S02CM 尚无正式平台结果。公开真机探针只证明新路径在相应通用布局上有
结构性收益，不能证明已经命中隐藏 Case，也不能据此宣称达到
`1195.599 us` 目标。

## 2. 根因与实现

旧 fastPath4 在完整 inner 行不满足 32 字节对齐时，每个 tile 使用
`currentReduceRows` 次二维 DataCopy，并把每行分别补齐到 32 字节。以
`inner=65` 为例，逐行搬运、逐行 padding 和加宽后的向量归约使 32M 元素
吞吐仅约 104～177 GB/s。

S02CM 在以下通用条件下启用连续完整行路径：

- fastPath4 已命中既有大输入分支；
- tile 从完整 inner 行起点开始并包含完整行；
- inner 不满足当前 dtype 的 32 字节 block 对齐；
- `inner <= 256`，保证最终 8 行 FP32 部分和不超过 8 KiB；
- 条件只依赖 shape、axis、dtype、对齐和 UB 容量。

新路径把连续的 `rows × inner` 输入合并为一次 DataCopy，只在整块末尾做
一次 padding；向量树把任意实际行数分解为若干 2 次幂块并归约到 8 行，
最后每个输出元素只进行一次 8 行标量合并。`inner >= 257`、非完整行及
其他路径全部回退原实现。Host 同时让 16～40 个完整输出行统一采用“一行
一核”，包括 FP32 非对齐 inner。

## 3. 正确性问题与修复

第一次累计回归在 BF16 的 `(40, 32, 768, 2)`、归约轴 `(0, 2)` 上拦截
到错误。原因是 BF16 分支把向上补成 2 次幂的 `reductionRows` 传给新函数，
从而读取了不存在的尾行；应传实际的 `currentReduceRows`。修正后原失败组
所在套件通过 `42/42`，随后七套测试从头重跑并通过：

```text
S02BT  69/69
S02CA  36/36
S02CB  42/42
S02CE  42/42
S02CF  42/42
S02CG  39/39
S02CL  45/45
合计  315/315
```

另外对 FP16/FP32 的 `inner=17,31,33,47,63,65,79,95,127,129,255,
257,511,513,1023` 做了最终扫宽，`30/30` 全部通过；257 及以上验证了
回退路径。

## 4. 性能证据

相同 26 点、约 32M 元素大规模图谱：

```text
S02CL：5991.740 us
S02CM：4996.010 us
变化：-995.730 us（下降 16.62%）
```

关键完整行布局：

| 布局 | dtype | S02CL（us） | S02CM（us） | 加速比 |
| --- | --- | ---: | ---: | ---: |
| 32M，31 行，inner=65 | FP16 | 634.220 | 144.570 | 4.387x |
| 32M，31 行，inner=65 | FP32 | 746.140 | 225.670 | 3.306x |
| 4M，31 行，inner=65 | FP16 | 99.600 | 66.560 | 1.496x |
| 4M，31 行，inner=65 | FP32 | 114.733 | 63.653 | 1.803x |

42 点小规模图谱总和为 `3065.800 -> 2985.540 us`（下降 `2.62%`）；
直接命中的 `inner=65` 两个点分别改善 33.17% 和 44.52%。大规模目标点的
3～4.4 倍改善在多轮运行中重复出现，是本候选的主要证据。

## 5. 路径、提交包与哈希

云端实验：

```text
/home/ma-user/work/s9/experiments/squaresum_s02cm_full_inner_contiguous_20260806_1740
```

关键日志：

```text
build_s02cm_final_surgical.log
correctness_s02cb_bf16_fix.log
cumulative_correctness_315_final_surgical.log
inner_sweep_release.log
large_atlas_release.log
path_atlas_release.log
```

云端发布快照：

```text
/home/ma-user/work/s9/releases/squaresum_s02cm_20260806_1825
```

本地提交包：

```text
D:\29722\Desktop\GCC\提交相关材料\20260806\S02CM\SquareSumV1.zip
大小：871273 bytes
ZIP SHA-256：15CB455648368C37098C871790488497DAE433D1D4FE941B16615BB65117D1E0
RUN SHA-256：E58C9FC7EA08C73E81F50C902C7074AEFD9E74D9641E82F5764776F6E7E151B3
Host SHA-256：71CC81D8E26DC54A56683A79D2E664A4E23F2EFF2B6EF2A3AB38F62056F49AB9
Kernel SHA-256：FBF2FE49A1C9CC84D554475C2831D2EE67ABFE3182C9404F481AEAD633F4BE4B
```

ZIP 仅有顶层 `SquareSumV1_zip/`，共 9 个目录/文件条目。源码权限为
`0640`，`.run` 为 `0755`。本地 ZIP 与云端发布 ZIP、正式源码及经过
315 项回归的构建产物均已逐字节核对一致。

## 6. 正式结果边界

```text
S02BA baseline = 3985.330 us
S02CL           = 3986.639 us（+1.309 us，淘汰）
S02CM           = 尚待正式结果
目标            = 1195.599 us
```

正式评测应上传 S02CM 包，并以五个 Case 全部 Pass 后的各 Case 耗时和
`prof_sum` 判断是否保留。不得把公开探针形状解释为隐藏 Case，也不得根据
Case 编号或猜测数据增加硬编码分支。
