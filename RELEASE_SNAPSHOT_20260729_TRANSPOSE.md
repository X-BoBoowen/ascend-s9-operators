# Transpose 发布与续作快照

时间：2026-07-29 20:34（Asia/Shanghai）

## 发布对象

本快照固化 `Transpose` 的向量化通用实现。正式版本优化了 FP16
非整齐尾块、FP32/INT32 8×8 块和 INT8 32×32 块；CANN 8.5
增强 Transpose API 候选因设备级性能回退被拒绝。

## 正式包

```text
本地:
D:\29722\Desktop\GCC\提交相关材料\Transpose.zip

云端:
/home/ma-user/work/s9/releases/transpose_20260729_2032/Transpose.zip
```

| 对象 | 大小 | SHA-256 |
| --- | ---: | --- |
| `Transpose.zip` | 401378 B | `52850235386690dcd89bf0fff039a4e510345a725aa2616d5f2a7c4d4fb5f1d0` |
| 包内 `.run` | 418022 B | `2baf9ecb7e38716b5d4a7ca8e89c890be2392c4b8c9352d8ca67d8d23ca9a9af` |

ZIP 只有一个顶层 `Transpose_zip/`，其中包含：

```text
Transpose_zip/
|-- op_host/
|-- op_kernel/
`-- custom_opp_euleros_aarch64.run
```

`.run` 在 ZIP 内权限为 `0750`，具有可执行权限。ZIP 完整性测试和
`.run` 自校验均通过，
算子注册名为 `Transpose`。

旧本地包已备份为：

```text
D:\29722\Desktop\GCC\提交相关材料\历史版本\
Transpose_pre_vectorized_20260729-2033.zip
```

旧包 SHA-256：
`b564a3724999cd2f3ef1b2ebf8c6e75c8c72778ffaa659125883130ac1d541cc`

## 正式源码

| 文件 | SHA-256 |
| --- | --- |
| `op_host/CMakeLists.txt` | `d1b100115b8c34135ccdfc54f91597847a7823ec76cdca995e2b80f5c6092cd2` |
| `op_host/transpose.cpp` | `d68ec597fa68861a04f5da11f17b481674500567328fef57500a387786fdc260` |
| `op_host/transpose_tiling.h` | `71d17bd19c58ada5ee29e6a1b3640e2f3c97ab8451cb9de6611d9a8befd4d5e1` |
| `op_kernel/CMakeLists.txt` | `dc5e6d36cbd092eed6fdc008a40896ede683299a3affeb91d693343bd6f29597` |
| `op_kernel/transpose.cpp` | `c59c71398d16431b7dd2d6b428dcc7969aae6f01d30ed0819df183f61278b581` |

包内源码、`submission-src/Transpose/`、云端 Git 仓库源码和最终构建
目录源码逐文件一致。

## 构建与验证

最终无缓存构建目录：

```text
/home/ma-user/work/s9/experiments/transpose_release_20260729_2027
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

同一套 484 例在最终实验候选和正式目录无缓存构建上各完整通过一次。
验证扩展也从仓库已跟踪的公共头文件强制重建，并重新通过 48 个定向
用例。

## 正式优化与性能证据

- FP16：`DataCopyPad` 补齐 16×16 边界块，完整块和尾块统一调用硬件
  `Transpose`，只回写有效区域；
- FP32/INT32：为 8×8 块生成 8 组列掩码，单次 `GatherMask` 输出
  64 个转置元素；
- INT8：把相邻字节视为 `uint16` 列对，抽取 32 行后用移位与向量
  转换拆成两条输出行。

NPU Event 同形状中位数：

| 场景 | 优化前 | 正式版本 |
| --- | ---: | ---: |
| FP16, 1024×1024 | 28.934 us | 29.111 us |
| FP16, 1000×1000 尾块 | 176.429 us | 51.082 us |
| FP32, 1024×1024 | 677.332 us | 188.231 us |
| INT32, 1024×1024 | 677.395 us | 188.276 us |
| INT8, 1024×1024 | 479.078 us | 88.978 us |
| FP32, (32,256,512) 末两维交换 | 2745.314 us | 785.701 us |
| INT8, (32,256,512) 末两维交换 | 1909.640 us | 349.423 us |

`msprof` 设备内核中位数：

| 场景 | 优化前 | 正式候选 |
| --- | ---: | ---: |
| FP16, 1024×1024 | 28.411 us | 28.451 us |
| FP16, 1000×1000 尾块 | 176.154 us | 50.321 us |
| FP32, 1024×1024 | 676.153 us | 185.894 us |
| INT8, 1024×1024 | 478.500 us | 88.492 us |

证据目录：

```text
/home/ma-user/work/s9/experiments/transpose_release_20260729_2027
/home/ma-user/work/s9/profiles/transpose_true_baseline_20260729
/home/ma-user/work/s9/profiles/transpose_vector_final_20260729
```

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
5. 平台反馈等待期间不根据隐藏形状猜分支，只按真实 Case 耗时继续优化。
