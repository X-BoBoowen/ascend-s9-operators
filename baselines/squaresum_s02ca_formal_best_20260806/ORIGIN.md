# SquareSumV1-S02CA 正式最佳基线

状态：**不可修改的正式评测基线。** 该目录仅用于与新候选做同机
Baseline-Candidate-Baseline 对照，不得直接在此目录开发。

## 来源

源码逐文件提取自：

```text
提交相关材料/20260806/S02CA/SquareSumV1.zip
```

原始 ZIP SHA-256：

```text
1209A5445514553CC1A0C4D243FB1764DC80A7B66CED427F7DA82699FA10C117
```

`op_host/square_sum_v1.cpp` SHA-256：

```text
0B5C6AEE3B01A63192A6CD4A59CF77A19373B6423274DC73968BAA77D84E9203
```

`op_kernel/square_sum_v1.cpp` SHA-256：

```text
C6EA5927D44DDF905E50B309C08E811E9DF614B7018D47F2BEB6A71115EC4C80
```

原 ZIP 中的预编译 `.run` 文件没有纳入仓库；云端门禁会在当前 CANN
环境中从以上源码重新构建。

## 正式成绩

S02CA 五个隐藏样例均通过，成绩如下：

```text
Case1: 6.410 us
Case2: 390.238 us
Case3: 138.633 us
Case4: 2557.111 us
Case5: 890.177 us
prof_sum: 3982.569 us
```

这是截至 2026-08-06 已知的最好正式结果，后续候选必须同时通过正确性、
题面域性能矩阵以及同机 A-B-A 稳定性门槛，才能替代它。
