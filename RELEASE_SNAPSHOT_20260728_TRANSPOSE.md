# Transpose 发布与续作快照

时间：2026-07-28 17:10（Asia/Shanghai）

## 发布对象

本快照固化 `Transpose` 的稳定通用实现。CANN 8.5 增强 Transpose API
候选因设备级性能回退被拒绝，没有进入正式源码或发布包。

## 正式包

```text
本地:
D:\29722\Desktop\GCC\提交相关材料\Transpose.zip

云端:
/home/ma-user/work/s9/releases/transpose_20260728_1706/Transpose.zip
```

| 对象 | 大小 | SHA-256 |
| --- | ---: | --- |
| `Transpose.zip` | 410044 B | `b564a3724999cd2f3ef1b2ebf8c6e75c8c72778ffaa659125883130ac1d541cc` |
| 包内 `.run` | 427040 B | `8685eaffd9787893a659c9b1b7424aa88c2881d10921c8246fda7ce4d717709e` |

ZIP 只有一个顶层 `Transpose_zip/`，其中包含：

```text
Transpose_zip/
|-- op_host/
|-- op_kernel/
`-- custom_opp_euleros_aarch64.run
```

`.run` 在 ZIP 内权限为 `0755`。ZIP 完整性测试和 `.run` 自校验通过，
算子注册名为 `Transpose`。

旧本地包已备份为：

```text
D:\29722\Desktop\GCC\提交相关材料\历史版本\
Transpose_before_20260728_1707_0b88b825e7ee.zip
```

旧包 SHA-256：
`0b88b825e7ee6e5cb598e5c3eb4638d4f652a4a0eb32671087a10a4aa3ff57e8`

## 正式源码

| 文件 | SHA-256 |
| --- | --- |
| `op_host/CMakeLists.txt` | `d1b100115b8c34135ccdfc54f91597847a7823ec76cdca995e2b80f5c6092cd2` |
| `op_host/transpose.cpp` | `d68ec597fa68861a04f5da11f17b481674500567328fef57500a387786fdc260` |
| `op_host/transpose_tiling.h` | `71d17bd19c58ada5ee29e6a1b3640e2f3c97ab8451cb9de6611d9a8befd4d5e1` |
| `op_kernel/CMakeLists.txt` | `dc5e6d36cbd092eed6fdc008a40896ede683299a3affeb91d693343bd6f29597` |
| `op_kernel/transpose.cpp` | `2684abf308169407a7cff07c7b82ba2e3c53e10f80a1ae9c26763c410c682293` |

包内源码、`submission-src/Transpose/`、云端 Git 仓库源码和最终构建
目录源码逐文件一致。

## 构建与验证

最终无缓存构建目录：

```text
/home/ma-user/work/s9/experiments/transpose_release_20260728_1703
```

环境：

- Ascend 910B4；
- CANN 社区版 8.5.0；
- GCC/G++ 10.3.0；
- `ascend910b` 编译目标。

安装最终 `.run` 后完成：

| 测试集 | 数量 | 结果 |
| --- | ---: | --- |
| 定向 | 48 | 全部通过 |
| 固定 seed 随机 | 200 | 全部通过 |
| 循环置换、任意排列、特殊 bit pattern 扩展 | 152 | 全部通过 |
| 阈值、分块尾部、3–5 维旋转边界 | 84 | 全部通过 |
| 合计 | 484 | 全部逐位相等 |

同一套 484 例在旧稳定构建和最终无缓存构建上各完整通过一次。

## 已拒绝候选

隔离候选：

```text
本地:
candidates/transpose_enhanced_20260728_1632

云端:
/home/ma-user/work/s9/experiments/transpose_enhanced_20260728_1643
```

候选使用 CANN 8.5 增强 Transpose API 对折叠后的二维矩阵做 UB 大块
转置。正确性通过，但设备级平均 kernel 时间显著回退：

| 场景 | 稳定版 | 增强候选 |
| --- | ---: | ---: |
| float32, 128×256 | 25.756 us | 243.971 us |
| int8, 128×256 | 19.536 us | 337.596 us |
| fp16, 127×257 | 20.913 us | 480.164 us |

profiling 证据：

```text
/home/ma-user/work/s9/profiles/transpose_enhanced_ab_20260728_1657
```

## 下一动作

1. 上传本快照记录的 `Transpose.zip`。
2. 保存 Case1–Case5 的 Pass/Fail、耗时和 `prof_sum`。
3. 不使用旧平台结果评价新包。
4. 若五个 Case 全 Pass，再按真实耗时定位下一轮瓶颈。
5. 平台反馈等待期间按题目顺序继续 `SquareSumV1`。
