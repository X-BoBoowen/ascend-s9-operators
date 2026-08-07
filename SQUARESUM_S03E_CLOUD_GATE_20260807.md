# SquareSumV1 S03E 云端门禁报告

更新时间：2026-08-07（Asia/Shanghai）

## 结论

S03E 已在 Ascend 910B4、CANN 社区版 8.5.0 环境完成隔离构建、安装、
正确性和 A/B/A 性能门禁，最终状态为 **PASS**。它合并了两个经过独立
验证、触发域互斥的改动：

- S03B：为大规模 `fastPath4` 通用非连续归约增加 40 核 Split-K 与
  FP32 workspace/tree finalize；
- S03D：把仅由 size-one 保留维隔开的物理连续归约路由到现有
  `fastPath1/fastPath2`。

S03A 的 BF16 算术实验没有合入 S03E，因为首轮 A/B/A 的一个基线复测
漂移超过门禁阈值，尚不足以证明其收益稳定。

## 正确性

| 检查 | 结果 |
| --- | ---: |
| 全域 atlas（fp16/bf16/fp32） | 36/36 Pass |
| Split-K 定向正确性 | 18/18 Pass |
| singleton-gap 定向正确性 | 72/72 Pass |
| 合计 | **126/126 Pass** |

所有测试均通过正式 `SquareSumV1` 自定义算子入口执行，不使用 CPU 替代
内核，也没有隐藏 shape、dtype 或轴的硬编码。

## A/B/A 性能结果

每个 profile 执行 30 次正式算子任务，剔除配套 Mul 后取第 10～30 次的
统计值；顺序为 S02F 基线 A、S03E 候选、S02F 基线 B。

| 优化域 | 基线聚合（ns） | S03E 聚合（ns） | 改善 | 结果 |
| --- | ---: | ---: | ---: | --- |
| general-strided Split-K | 8,347,582,250 | 1,107,182,000 | **86.7365%** | Pass |
| singleton-gap | 424,498,250 | 65,480,500 | **84.5746%** | 性能/回退均 Pass；一项基线漂移复测 |

Split-K 的六组 dtype×shape 改善范围为 `81.9060%～90.0072%`，全部通过
10% 最低收益、3% 最大回退和 3% 最大基线漂移门槛。

singleton-gap 的六组改善范围为 `78.7377%～88.7712%`。首轮唯一异常是
`fp32 fast2` 的两次 S02F 基线相差 `3.5813%`，不是候选回退；该组独立
复测结果如下：

| 基线 A（ns） | S03E（ns） | 基线 B（ns） | 基线漂移 | 改善 |
| ---: | ---: | ---: | ---: | ---: |
| 81,331,000 | 17,300,000 | 81,481,500 | **0.1849%** | **78.7486%** |

复测同时通过漂移、回退和收益门槛，因此 S03E 最终门禁判定为 Pass。

## 可追溯文件

- 候选源码：`candidates/squaresum_s03e_s02f_splitk_singleton_20260807/`
- 当前提交源码：`submission-src/SquareSumV1/`
- 最新结果：`artifact/result.json`
- 完整运行日志：`artifact/run.log`
- Split-K 比较：`artifact/SquareSumV1/s03e_splitk_comparison.json`
- singleton-gap 比较：`artifact/SquareSumV1/s03e_singleton_comparison.json`
- 漂移复测：`artifact/SquareSumV1/s03e_singleton_fp32_repeat/comparison.json`
- 云端运行目录：`/home/ma-user/work/s9/runs/s03e_gate_20260807_1348`
- 上传包：`提交相关材料/20260807/S03E/SquareSumV1.zip`
- ZIP SHA-256：`dd4b22bc7eb5000c07357bd9d2355bb6b8e7bdf9e61cb7e8fb4fb3a5196f19b2`
- 包内 `.run` SHA-256：`e0c065ea9cee4b7720e563856ea15292e23d3c59e561b8d4ecb577f5a7ee2719`

## 平台正式结果与最终判定

| Case | S02F（μs） | S03E（μs） | 变化（μs） |
| --- | ---: | ---: | ---: |
| Case1 | 6.530 | 6.540 | +0.010 |
| Case2 | 394.308 | 394.3775 | +0.0695 |
| Case3 | 237.885 | 238.865 | +0.980 |
| Case4 | 1716.7145 | 1718.6145 | +1.900 |
| Case5 | 868.5575 | 902.268 | +33.7105 |
| 合计 | **3223.995** | **3260.665** | **+36.670（+1.1374%）** |

五个 Case 均通过正确性，但性能比 S02F 退化，主要来自 Case5。由此可知，
两个公开定向域的巨大收益没有命中隐藏主路径，且局部 A/B/A 不能单独支持
基线晋升。S03E 最终判定为 **正式否决**，提交源码已恢复为 S02F；该包与
报告仅作为可追溯的失败实验保留。
