# SquareSumV1 S03 本地封板审计

更新时间：2026-08-06（Asia/Shanghai）

## 1. 结论

本轮不再沿用 S02CA 后继代码继续调阈值，而是从完整正式记录中性能最好的 S02F 源码重新开始。三个单变量候选已经完成本地源码契约、合法域路由和数值模型验证；本地没有 CANN 8.5.0/Ascend 910B，因此下一项能够产生新证据的工作只剩云端原生构建、NPU 正确性和 A/B/A 性能测试。

当前禁止生成正式提交包，也不宣称任何候选已经降低官方耗时。

## 2. 基线纠正

| 版本 | Case1 | Case2 | Case3 | Case4 | Case5 | `prof_sum` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S02F | 6.530 | 394.308 | 237.885 | 1716.7145 | 868.5575 | **3223.995** |
| S02BA | 6.510 | 399.088 | 138.753 | 2550.761 | 890.218 | 3985.330 |
| S02CA | 6.410 | 390.238 | 138.633 | 2557.111 | 890.177 | 3982.569 |

S02CA 只是 S02BA 后继路线中的局部最低值。相对 S02F，S02BA 虽然令 Case3 快约 `99.132 us`，却令 Case4 慢约 `834.047 us`，总计回退 `761.335 us`。因此继续在 S02CA 上微调不能代表对真实最好实现的优化。

用户定义的挑战目标仍按 `T_baseline=3985.330 us` 计算，即 `≤1195.599 us`。若按真实历史最好 S02F 再取 30%，更严格参考线为 `967.1985 us`。

权威基线：

```text
官方 ZIP：D:\29722\Desktop\GCC\提交相关材料\20260730\S02F_1635\SquareSumV1.zip
ZIP SHA-256：EECD9F1FD6B4C0617B6EC2EC632F24BB0F310D5A9D2F2125F03F6EC86ECFAF5B
Host SHA-256：6C4731323E66D3E7A02044FA214386D7C10167A0BA145C6AA8D3902535F5F367
Kernel SHA-256：B668B6A2217130E1878063713B3D03F663C7626B7C175FB6C8E503119002255A
```

仓库基线目录：

```text
baselines/squaresum_s02f_global_best_20260806/SquareSumV1
```

## 3. 独立候选

### 3.1 S03A：BF16 中间转换消除

目录：

```text
candidates/squaresum_s03a_s02f_bf16_fused_20260806/SquareSumV1
```

相对 S02F 只有 Kernel 变化：删除八组 `FP32 square -> BF16 RINT -> FP32` 中间往返，共删除 80 行、增加 0 行。最终写回 BF16 的七处 `CAST_RINT` 和标量回退语义均保留。该候选用于判断 BF16 向量路径上的额外转换是否属于真实瓶颈，不与路由修改混合。

### 3.2 S03B：fastPath4 通用 40 核 Split-K

目录：

```text
candidates/squaresum_s03b_s02f_fast4_splitk_20260806/SquareSumV1
```

S02F 的一般 fastPath4 使用 `ceil(outputElements / 64)` 个核。对于输出较少但归约很大的合法布局，这可能只启动 1～16 核。S03B 仅在下列通用条件成立时启用 40 核 FP32 workspace Split-K：

- `fastPath == 4`；
- 输入元素数至少 `2^18`；
- 归约元素数至少 `2048`；
- 输出元素数为 `1..1024`。

每个核处理互不重叠的扁平归约区间；二维 `DataCopyPad` 只在最后一个归约维的自然行内搬运，遇到边界即切段；40 份 FP32 partial 最后使用 S02F 已有树归约写回。门限只依赖 shape、axis、stride 和规模，不使用 Case 编号或输入值。

### 3.3 S03D：单例维间隙连续路由

目录：

```text
candidates/squaresum_s03d_s02f_singleton_gap_20260806/SquareSumV1
```

该候选只修改 Host 元数据分类，Kernel 和四份其他源码与 S02F 字节一致。当首尾归约轴之间只夹有大小为 1 的保留维时，物理地址仍连续：

- 最后一个归约轴就是末轴时，进入 fastPath1；
- 后面仍有输出后缀时，仅在 `reduceElements >= 8192` 时进入 fastPath2；
- 小于门限或存在大小大于 1 的真实间隙时保持 S02F 原路由。

这项改动单独提取了 S02BA 路线中可能解释 Case3 改善的连续性修正，没有带入其大规模 Kernel 和路由重写，因此不会主动改变普通 fastPath3/4。

## 4. 本地门禁结果

| 门禁 | 结果 |
| --- | ---: |
| 合法 profile 矩阵 | 34/34 |
| S03A 源码契约 | 12/12 |
| S03A BF16 官方容差模型 | 27/27 |
| S03B 路由契约 | 8/8 |
| S03B 多轴/负轴/三 dtype 数值与坐标模型 | 45/45 |
| S03D 路由与物理坐标等价 | 9/9 |
| 新增 Python 文件 AST、JSON | Pass |
| 云端门禁脚本 Bash 语法 | Pass |

完整本地运行日志保存在未纳入 Git 的：

```text
.codex_tmp/s03_local_gate_20260806.log
```

## 5. 云端门禁

统一脚本：

```text
diagnostics/run_squaresum_s03_cloud_gate_20260806.sh
```

脚本会：

1. 清理并原子更新仓库 `artifact/`；
2. 从同一个官方工程模板隔离构建 S02F、S03A、S03B、S03D；
3. 分别安装到互不污染的 OPP 目录；
4. 对 S03A 执行全路径三 dtype 正确性，对 S03B 执行 18 项定向正确性，对 S03D 执行 72 项 singleton-gap 正确性；
5. S03A 使用 BF16 core 矩阵，S03B 使用三 dtype strided-splitK 矩阵，S03D 使用三 dtype singleton-gap 矩阵；每项均执行 S02F-A / candidate / S02F-B；
6. 使用与比赛任务流一致的 30 次 `Mul + SquareSumV1`，过滤 Mul 后取第 10～30 次算子任务中位数；
7. 将候选比较结果写入 `artifact/result.json`，完整日志写入 `artifact/run.log`。

云端启动后的调用形式：

```bash
bash diagnostics/run_squaresum_s03_cloud_gate_20260806.sh \
  /home/ma-user/work/<官方SquareSumV1工程模板> \
  /home/ma-user/work/s03_gate_20260806
```

工作目录必须是一个尚不存在的新目录。实际模板路径应在云端只读确认后代入，不能猜测。

## 6. 晋级规则

- 任一候选只要构建失败或正确性失败，立即淘汰；
- S03A 要求 BF16 矩阵至少改善 2%，任一项回退不超过 3%；
- S03B、S03D 要求目标矩阵至少改善 10%，任一项回退不超过 3%；
- 两次 S02F 基线漂移不得超过 3%；
- 通过自建矩阵只代表有资格形成组合候选，不代表官方隐藏 Case 已改善；
- 只有独立候选通过后，才组合有效改动并再次做全域正确性和 A/B/A；
- 组合候选通过后才生成 `.run` 和比赛 ZIP，之后记录官方五项结果。

截至本文件写入时，本地可验证事项已完成。下一步需要 Ascend 910B 与 CANN 社区版 8.5.0。
