# SquareSumV1 评分口径与性能矩阵复核

日期：2026-08-06

## 结论

S02CN 的正式结果为 `4010.864 us`，相对 S02BA 基线 `3985.330 us`
回退 `25.534 us / 0.6407%`，相对当前最好正式结果 S02CA
`3982.569 us` 回退 `28.295 us / 0.7105%`。S02CN 正式淘汰，不能再作为
后续候选基线。

此前实验存在三项会产生错误性能判断的问题：

1. 使用 NPU Event 微基准筛选候选，没有复刻比赛的 `msprof` Task
   Duration、30 轮交错调用和 `[10:30]` 中位数口径。
2. S02CN 主性能矩阵仅覆盖 FP16/FP32；官方样例对 Case2/3 的特殊 Torch
   Tensor 构造方式强烈提示 BF16，需要作为独立性能门禁，而不仅是正确性
   用例。
3. 一部分性能图谱超出了 Excel 规定的维度范围。它们可用于压力测试，不能
   用于证明比赛域内收益。

## 赛题域复核

题面中最后四个命名维度的范围为：

| 维度 | 最大值 |
|---|---:|
| N | 10000 |
| N2 | 10000 |
| N3 | 1000 |
| N4 | 200 |

对关键旧图谱的静态复核结果：

- `squaresum_path_atlas_20260806.py`：21 个 shape 中 7 个越界。
- `squaresum_s02cn_fast2_surface_20260806.py`：24 个 inner 点中 9 个越界。
- S02CN 报告中最醒目的 `inner=15/17` 点分别产生 `reduce=17476/15420`，
  超过 N2 的 10000 上限；这些结果不得作为提交依据。
- `inner=31/64` 等一部分 S02CN 点仍在题目范围内，但正式成绩证明该窄路由
  没有覆盖主要评分耗时。

## 新增验证工具

以下文件构成新的准入链路：

- `diagnostics/squaresum_official_profile_matrix_20260806.json`：只包含题目范围
  内的 fast1/2/3/4、对齐/非对齐、keep_dims 和全归约用例。
- `diagnostics/validate_squaresum_profile_matrix_20260806.py`：运行前验证每个
  shape 的 N/N2/N3/N4 范围，并依据当前 host 逻辑复算 fastPath。任何越界
  或路由不符都会立即失败。
- `diagnostics/squaresum_official_profile_case_20260806.py`：复刻官方 30 轮
  `4096x4096 FP32 Mul -> SquareSumV1` 任务流，并使用官方同类误差判定。
- `diagnostics/parse_squaresum_official_msprof_20260806.py`：仅排除
  `aclnnMul`，随后按官方 `[10:30]` 取 Task Duration 中位数；同时输出完整
  任务名称计数，防止把错误任务当作算子耗时。
- `diagnostics/run_squaresum_official_profile_20260806.sh`：单用例入口。它要求
  profiler 中恰好发现 30 个 `aclnnMul`，否则判定任务流不等价并失败。

本地静态验证命令：

```bash
python3 diagnostics/validate_squaresum_profile_matrix_20260806.py
```

云端单用例命令示例：

```bash
bash diagnostics/run_squaresum_official_profile_20260806.sh \
  fast2_middle_aligned bf16 S02CA
```

## 下次云端执行顺序

1. 安装 S02CA，先运行 `public_case1/fp16`，确认新工具与公开正式 Case1
   处于相同量级，并确认任务计数为 30 个 Mul、30 个 SquareSumV1。
2. S02CA 对 core 层全部 shape 运行 BF16，得到此前缺失的 BF16 路径基线。
3. core 层再运行 FP16/FP32，形成 fast1/2/3/4 的评分等价基线。
4. 只有在相同 shape、dtype、任务流下稳定优于 S02CA 的内核改动，才允许
   新建候选包。
5. 每个候选执行 Baseline-Candidate-Baseline，至少三组；小于 5% 的变化
   视为平台噪声，不提交正式评测。

## 当前不能声称的事项

本地没有 910B/CANN 8.5.0 运行环境，新增工具只完成了语法检查、矩阵域检查
和路由复算。Python `torch.mul(..., out=...)` 是否在该环境中产生与官方包装
器完全相同的 `aclnnMul` 名称，必须由第一次云端 msprof 的 30 条任务计数
确认。在此确认之前，不把新工具产生的任何数值称为正式等价结果。
