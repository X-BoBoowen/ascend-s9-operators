# SquareSumV1-S02CQ 来源与状态

状态：**本地候选，尚未经过 910B/CANN 8.5.0 构建、正确性或性能验证，禁止提交。**

## 基线

本候选从正式提交包 `提交相关材料/20260806/S02CA/SquareSumV1.zip`
逐文件提取。该 ZIP 的 SHA-256 为：

```text
1209A5445514553CC1A0C4D243FB1764DC80A7B66CED427F7DA82699FA10C117
```

原始 kernel SHA-256：

```text
C6EA5927D44DDF905E50B309C08E811E9DF614B7018D47F2BEB6A71115EC4C80
```

原始 host SHA-256：

```text
0B5C6AEE3B01A63192A6CD4A59CF77A19373B6423274DC73968BAA77D84E9203
```

S02CA 的正式平台成绩为 `3982.569 us`，是当前已知最好正式结果。

## 唯一内核改动

BF16 输入仍先通过受支持的 `Cast(bfloat16_t -> float)` 转成 FP32，再用
FP32 `Mul` 做平方。S02CA 会把平方结果重新 Cast 为 BF16，然后再次 Cast
回 FP32 才归约；S02CQ 删除这对中间 Cast，直接以 FP32 平方结果进行 FP32
归约，最终输出仍 Cast 为 BF16。

因此每个 BF16 输入 tile 减少两次全向量 Cast。FP16、FP32 分支、host
tiling、TilingKey、workspace 和输出写回均未修改。

这会放弃 `torch.square` 的逐元素 BF16 中间舍入，但保留最终 BF16 输出。
它只能以比赛官方误差规则判断正确性，不能声称逐 bit 等价。

## 官方 API 依据

- CANN 8.5.0 的基础 `Mul` 在 Atlas A2 上仅支持 half、int16、int32、float，
  不支持 bfloat16_t。
- CANN 8.5.0 的 `MulCast` 在 Atlas A2 上只支持 half 输入及 int8/uint8
  输出，也不能替代 BF16 平方。

因此 S02CQ 没有调用未支持的 BF16 `Mul`，而是减少语义上可由比赛容差
验证的中间量化往返。

## 必须通过的云端门禁

1. 910B/CANN 8.5.0 完整构建。
2. 现有累计正确性矩阵全部通过。
3. 新增 BF16 fast1/2/3/4 题面域矩阵全部通过官方式误差验证。
4. S02CA 与 S02CQ 使用 msprof 官方口径执行 Baseline-Candidate-Baseline。
5. 任一 BF16 正确性失败立即淘汰；稳定收益小于 5% 不进入正式提交。
