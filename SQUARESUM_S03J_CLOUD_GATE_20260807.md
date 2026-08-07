# SquareSumV1 S03J 云端门禁报告

更新时间：2026-08-07（Asia/Shanghai）

## 结论

S03J 已在 Ascend 910B4、CANN 社区版 8.5.0 环境完成隔离构建、安装、
正确性、边界和 A/B/A 性能验证，当前状态为 **可提交、等待正式平台结果**。
它以正式历史最好 S02F 为唯一基线，只优化 `fastPath2` 的长归约、小 inner
通用路径；S02AY 曾导致 Case4 大幅回退的 `fastPath3` 改动没有合入。

这不是对隐藏 Case 的 shape、dtype 或轴进行猜测。路由只依赖运行时公开
元数据：连续中间轴归约、`reduceElements >= 2048`、`innerElements <= 64`，
达到 workspace tree 门槛时仍使用 S02F 原有 tree key。

## 实现与消融

- S03H：只切换 16K chunk。FP16/BF16 正确且没有稳定收益，FP32 仅
  `120/128` 正确，否决且未打包。
- S03I：加入幂次树归约，FP16/BF16 目标点复现约 24%～41% 改善，但
  FP32 仍为 `120/128`，定位到 long-key FP32 finalizer 缓冲区不足，否决。
- S03J：补足 FP32 long non-tree 的 4096 B finalizer 缓冲区，并让 tree
  finalize 保持 S02F 的 key 3；三个 dtype 均通过。

关键代码改动：

1. `fastPath2 && reduceElements >= 2048 && innerElements <= 64` 的非 tree
   路径使用 long chunk；
2. 每个 tile 的归约行数向下取 2 次幂，尾块补零后做树归约；
3. 第一块直接初始化累计向量，后续块再相加，移除一次无条件清零；
4. 只在仍有后续 tile/输出时执行 V→MTE2 同步；
5. FP32 long non-tree 的 `floatBuffer` 从 32 B 修正为
   `TILE_OUTPUTS * sizeof(float) = 4096 B`。

## 正确性

| 检查 | 结果 |
| --- | ---: |
| 原定向矩阵 | 46/46 Pass |
| BF16 语义探针 | 4/4 Pass |
| 随机矩阵 | 150/150 Pass |
| 扩展矩阵 | 726/726 Pass |
| S03J 路由/边界矩阵 | 135/135 Pass |
| 合计 | **1061/1061 Pass** |

新增边界矩阵覆盖 `reduce=2047/2048/2049`、`8191/8192/8193`，
`inner=1/2/8/31/64/65`，以及 workspace tree 的
`65535/65536/65600` 切换；全部 shape 均在题面公开维度上限内。

## 性能证据

目标路径按 S02F→S03J→S02F 执行，FP16/BF16/FP32 全部通过正确性：

| 场景 | 三 dtype 改善范围 |
| --- | ---: |
| fastPath2，inner=2 | 37.1528%～40.7419% |
| fastPath2，inner=8 | 24.4427%～25.3687% |
| 六个目标点聚合 | **34.1427%** |

未修改路径随后独立复测 15/15 正确；最坏候选回退为 `2.3903%`。最大
基线漂移为 `11.9517%`，来自 FP32 control 点，其候选回退仅 `2.3903%`；
因此这里只把控制测试用于排除明显旁路退化，不把噪声当作收益证明。

S02F→S02AY 的源码与平台结果消融还显示：S02AY 在同类小-inner 路径上
改善 23%～42%，同时其 `fastPath3` rank-4 路径回退 40%～53%。这与正式
S02F→S02BA 的 Case3 改善、Case4 大幅回退方向一致，但只能作为通用路径
归因证据，不能据此断言隐藏 Case 的具体输入。

## 提交物与可追溯信息

- 提交源码：`submission-src/SquareSumV1/`
- 候选源码：
  `candidates/squaresum_s03j_s02f_fast2_small_inner_safe_20260807/`
- 上传包：`提交相关材料/20260807/S03J/SquareSumV1.zip`
- ZIP 大小：`474641 bytes`
- ZIP SHA-256：
  `e775b0889148ae5fe4c1d6dd678db90515bbe5c3eba3da894625d96308d17818`
- 包内 `.run` SHA-256：
  `4215edfd68bdb626f516dc3d6aa662065fe4e4eb24497311841818987722abc1`
- ZIP 条目：9 个，唯一顶层目录 `SquareSumV1_zip/`
- `.run` 权限：`0755`
- 云端构建目录：`/home/ma-user/work/s9/runs/s03j_gate_20260807_1529`
- 最新运行结果：`artifact/result.json`
- 最新运行日志：`artifact/run.log`

## 结论边界

云端门禁只能证明 S03J 的实现正确，并在其通用目标路径上稳定快于 S02F；
它不能证明该路径占正式五个 Case 的多少比例。S03J 尚无正式平台成绩，
不得宣称已经改写历史最好或达到第 10 名。收到正式 Case1～Case5 后，再
按逐 Case 差值决定保留、回滚或继续消融。
