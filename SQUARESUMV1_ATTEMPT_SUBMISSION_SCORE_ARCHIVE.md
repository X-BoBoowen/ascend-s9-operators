# SquareSumV1 全量尝试、正式提交与成绩归档

更新时间：2026-08-07（Asia/Shanghai）

## 1. 归档范围

本文只记录第五题 **SquareSumV1**，统一管理：

- 比赛平台正式返回的 Case1～Case5 与 `prof_sum`；
- 已经生成但没有完整正式成绩的提交包；
- 本地或云端做过、但没有提交平台的优化候选；
- 每个正式包的 SHA-256，保证成绩能够追溯到实际上传文件。

判定口径：

- 只有平台返回值才叫“正式成绩”；
- 本地 NPU Event、`msprof`、公开 Case 和云端 A/B 都不替代正式成绩；
- 文件夹或 ZIP 存在不等于已经提交；
- 不根据隐藏 Case 猜 shape、dtype 或 TilingKey，不做测试数据硬编码；
- 所有耗时单位均为微秒（μs）。

## 2. 当前结论

- 已归档 **19 次**完整正式成绩，全部为 `5/5 Pass`。
- 历史最好是 **S02F：3223.995 μs**。
- S02BA 以后分支最好是 **S02CA：3982.569 μs**，但仍比 S02F 慢
  `758.574 μs（23.53%）`。
- 用户提供的当时第 10 名成绩为 `1268.598 μs`。S02F 仍慢
  `1955.397 μs`，需要在 S02F 基础上再降低 `60.65%`。
- 本地核验到 **25 个**按日期/版本保存的 SquareSumV1 ZIP：19 个有完整
  正式成绩，6 个没有完整平台成绩。最新的 S03E 已通过真机门禁并打包，
  尚未取得平台隐藏 Case 成绩。

## 3. 全部正式提交成绩

`Δ前次` 为本次 `prof_sum -` 上一次正式结果；负数表示变快。
`Δ最好` 为本次相对历史最低 S02F `3223.995` 的差值。

| # | 版本 | 日期 | Case1 | Case2 | Case3 | Case4 | Case5 | prof_sum | Δ前次 | Δ最好 | ZIP SHA 前缀 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | S01 | 07-29 | 7.720 | 404.592 | 241.660 | 1720.188 | 908.376 | 3282.536 | — | +58.541 | `d5d0407c7f81` |
| 2 | S02E2 | 07-30 | 6.3505 | 400.158 | 239.505 | 1718.344 | 895.168 | 3259.5255 | -23.0105 | +35.5305 | `14b3384ed0a5` |
| 3 | S02F | 08-02 | 6.530 | 394.308 | 237.885 | 1716.7145  | 868.5575 | **3223.995** | -35.5305 | **0** | `eecd9f1fd6b4` |
| 4 | S02BA | 08-05 | 6.510 | 399.088 | 138.753 | 2550.761 | 890.218 | 3985.330 | +761.335 | +761.335 | `23007e299e83` |
| 5 | S02BB | 08-05 | 7.880 | 405.580 | 140.800 | 2551.560 | 902.756 | 4008.576 | +23.246 | +784.581 | `dc7451016f34` |
| 6 | S02BF | 08-05 | 7.908 | 407.248 | 139.776 | 2541.768 | 904.500 | 4001.200 | -7.376 | +777.205 | `b5484c12895c` |
| 7 | S02BG | 08-05 | 6.500 | 399.998 | 138.4125 | 2553.771 | 896.918 | 3995.5995 | -5.6005 | +771.6045 | `d3bc2703620c` |
| 8 | S02BI | 08-06 | 6.470 | 390.5975 | 139.463 | 2552.781 | 893.4775 | 3982.789 | -12.8105 | +758.794 | `bb3eccdfba65` |
| 9 | S02BK | 08-06 | 6.570 | 389.5275 | 138.733 | 2554.151 | 895.818 | 3984.7995 | +2.0105 | +760.8045 | `d283db77bc37` |
| 10 | S02BM | 08-06 | 7.660 | 405.020 | 140.068 | 2555.804 | 904.548 | 4013.100 | +28.3005 | +789.105 | `3893e2005152` |
| 11 | S02BN | 08-06 | 7.584 | 395.736 | 140.568 | 2550.248 | 906.912 | 4001.048 | -12.052 | +777.053 | `0595ab4ebe7c` |
| 12 | S02BS | 08-06 | 7.416 | 407.388 | 139.676 | 2538.380 | 903.168 | 3996.028 | -5.020 | +772.033 | `65ca55c54f3f` |
| 13 | S02BT | 08-06 | 7.448 | 395.728 | 139.240 | 2548.652 | 904.312 | 3995.380 | -0.648 | +771.385 | `5e1aaec27c84` |
| 14 | S02CA | 08-06 | 6.410 | 390.238 | 138.633 | 2557.111 | 890.177 | **3982.569** | -12.811 | +758.574 | `1209a5445514` |
| 15 | S02CD | 08-06 | 6.390 | 397.4975 | 139.603 | 2556.431 | 894.4275 | 3994.349 | +11.780 | +770.354 | `5d847711252a` |
| 16 | S02CF | 08-06 | 6.390 | 397.4975 | 139.603 | 2556.431 | 894.4275 | 3994.349 | 0.000 | +770.354 | `26693f15a8d7` |
| 17 | S02CL | 08-06 | 6.470 | 388.257 | 139.113 | 2558.891 | 893.908 | 3986.639 | -7.710 | +762.644 | `a4c82815ef2f` |
| 18 | S02CM | 08-06 | 6.550 | 400.108 | 139.1025 | 2546.821 | 895.208 | 3987.7895 | +1.1505 | +763.7945 | `15cb45564836` |
| 19 | S02CN | 08-06 | 7.448 | 409.292 | 140.352 | 2550.192 | 903.580 | 4010.864 | +23.0745 | +786.869 | `1d0477757517` |

## 4. 正式成绩分析

### 4.1 真正有效的历史改善

| 版本 | prof_sum | 相对上一正式版本 | 相对 S01 |
| --- | ---: | ---: | ---: |
| S01 | 3282.536 | — | — |
| S02E2 | 3259.5255 | -0.70% | -0.70% |
| S02F | **3223.995** | -1.09% | **-1.78%** |

S01→S02F 的总改善只有 `58.541 μs（1.78%）`。虽然正确性和局部测试
覆盖显著增加，但平台总耗时没有形成数量级下降。

### 4.2 S02BA 以后为什么不能称为进步

S02F 与 S02BA 的 Case 对比：

| Case | S02F | S02BA | 变化 |
| --- | ---: | ---: | ---: |
| Case1 | 6.530 | 6.510 | -0.020 |
| Case2 | 394.308 | 399.088 | +4.780 |
| Case3 | 237.885 | 138.753 | **-99.132** |
| Case4 | 1716.7145 | 2550.761 | **+834.0465** |
| Case5 | 868.5575 | 890.218 | +21.6605 |
| 合计 | **3223.995** | 3985.330 | **+761.335** |

S02BA 路线明显优化了 Case3，却让 Case4 大幅回退。S02BA～S02CN 的
`prof_sum` 始终在 `3982.569～4013.100`，范围只有 `30.531 μs`，说明
后续多次阈值、分核和局部快路修改没有触及决定性瓶颈。

S02CD 与 S02CF 的平台数字完全相同，但 ZIP 哈希不同。本文按用户回传
原样保留，不推断平台是否复用了缓存。

## 5. 已生成 ZIP、但没有完整正式成绩

| 版本/包 | 文件位置 | ZIP SHA-256 | 已知状态 |
| --- | --- | --- | --- |
| 20260727 | `提交相关材料/20260727/SquareSumV1.zip` | `8a5e869946ed33ea970e15d3df5ea746936dd3b69a28b1cd3830b3f7d6405b7f` | 包存在，未找到唯一对应的五 Case 成绩 |
| S02AY | `提交相关材料/20260805/S02AY/SquareSumV1.zip` | `c5a9f237124374d97c78dad97966833ad6a5783ea9b6c8d67cd53ecdebe162ac` | 阶段包/云端实验，无完整平台回传 |
| S02BZ | `提交相关材料/20260806/S02BZ/SquareSumV1.zip` | `7a39c43e37a209d6dcf2477a902c0817f410540fc7c74b1dd41926f7bab69003` | 包存在，无完整平台回传 |
| S02CE | `提交相关材料/20260806/S02CE/SquareSumV1.zip` | `bbe62860c2db8891f5a6699feff649c46cebf9d935bd23d2c81633b4c3eb1d70` | 候选包/云端报告存在，无独立正式成绩 |
| S02CG | `提交相关材料/20260806/S02CG/SquareSumV1.zip` | `c31babbc0802fd89b8bb9ed3b85294876338c84699afa61d8813dd39eb6e1f48` | 候选包/云端报告存在，无独立正式成绩 |

S02BD 完成过云端 A/B，但没有发现同名提交 ZIP，也没有收到完整平台五 Case
成绩，因此只算云端候选，不进入正式成绩表。

## 6. 全部优化尝试索引

本节记录“实际做过什么”，但不把本地候选冒充正式提交。实现位于
`candidates/`，验证脚本主要位于 `diagnostics/`。

### 6.1 2026-07-29：初始归约路线探索

| 尝试 | 方向 | 正式成绩 |
| --- | --- | --- |
| `squaresum_semantics_20260729_1710` | 平方与累加语义 | 无 |
| `squaresum_parallel_20260729_1718` | 并行归约 | 无 |
| `squaresum_tree_20260729_1735` | 树形归约 | 无 |
| `squaresum_tree_split8_20260729_1740` | 8 路树形拆分 | 无 |
| `squaresum_tree_split16_20260729_1743` | 16 路树形拆分 | 无 |
| `squaresum_atomic_fp32_20260729_1755` | FP32 原子累加 | 无 |
| `squaresum_atomic_half_20260729_1814` | Half 原子路线 | 无 |
| `squaresum_chunk16k_20260729_1818` | 16K 分块 | 无 |
| `squaresum_selective16k_20260729_1823` | 条件化 16K 分块 | 无 |
| `squaresum_tilingkey_20260729_1833` | TilingKey 分路 | 无 |
| `squaresum_workspace_probe_20260729_1847` | Workspace 探针 | 无 |
| `squaresum_workspace_best_20260729_1854` | Workspace 阶段最好候选 | 无 |

### 6.2 2026-07-30：S02A～S02G

| 版本 | 方向 | 正式成绩 |
| --- | --- | ---: |
| S02A | atomic workspace | 无 |
| S02B | atomic scalar finalize / packed partial | 无 |
| S02C | finalizer tree / gated finalizer tree | 无 |
| S02H（早期同名） | split tree key | 无 |
| S02E | strided tree | 无 |
| S02E2 | strided padded tree | **3259.5255** |
| S02F | grouped vector8 | **3223.995** |
| S02G | segmented rows | 无 |

同一版本下有两个候选目录时，表示该阶段有先后实现，不等于两次正式提交。

### 6.3 2026-08-02：S02H～S02AL

| 版本 | 方向 | 正式成绩 |
| --- | --- | --- |
| S02H | grouped unaligned | 无 |
| S02I | segmented threshold 成本模型 | 无；仅诊断/模型 |
| S02J | long vector8 静态方案 | 无；仅诊断/模型 |
| S02K | sync clear 静态方案 | 无；仅诊断/模型 |
| S02L | grouped long tail 静态方案 | 无；仅诊断/模型 |
| S02N | middle tail padding 静态方案 | 无；仅诊断/模型 |
| S02O | long middle chunk 静态方案 | 无；仅诊断/模型 |
| S02P | long strided chunk 静态方案 | 无；仅诊断/模型 |
| S02Q | grouped medium tail 静态方案 | 无；仅诊断/模型 |
| S02R | grouped medium width | 无 |
| S02S | last vector2 | 无 |
| S02T | segmented 64 | 无 |
| S02U | segmented 32 | 无 |
| S02V | medium partial capacity | 无 |
| S02W | last vector1 long | 无 |
| S02X | grouped long narrow | 无 |
| S02Y | medium singleton axis | 无 |
| S02Z | small-output last tree | 无 |
| S02AA | small-output single chunk | 无 |
| S02AB | parallel last batch store | 无 |
| S02AC | workspace last tree | 无 |
| S02AD | FP32 tree capacity | 无 |
| S02AE | FP32 workspace tree | 无 |
| S02AF | tree root direct | 无 |
| S02AG | first partial direct | 无 |
| S02AH | last first-partial direct | 无 |
| S02AI | grouped first-partial direct | 无 |
| S02AJ | sequential first-block direct | 无 |
| S02AK | grouped short narrow | 无 |
| S02AL | atomic control | 无 |

### 6.4 2026-08-05：S02AM～S02BI

| 版本 | 方向 | 提交/成绩状态 |
| --- | --- | --- |
| S02AM | parallel last safe batch | 无正式成绩 |
| S02AN | FP32 sequential capacity | 无正式成绩 |
| S02AO | FP32 single-tile tree | 无正式成绩 |
| S02AP | corrected atomic | 无正式成绩 |
| S02AQ | short control | 无正式成绩 |
| S02AR | grouped vector control | 无正式成绩 |
| S02AS | grouped vector barrier | 无正式成绩 |
| S02AT | grouped task boundary | 无正式成绩 |
| S02AU | long stride guard | 无正式成绩 |
| S02AV | middle task boundary | 无正式成绩 |
| S02AW | grouped width-1 aligned | 无正式成绩 |
| S02AX | grouped narrow aligned | 无正式成绩 |
| S02AY | workspace task boundary 综合候选 | 有 ZIP，无完整正式成绩 |
| S02AZ | singleton-gap contiguous | 无正式成绩 |
| S02BA | singleton-gap threshold | **正式 3985.330** |
| S02BB | compact middle-8 | **正式 4008.576** |
| S02BC | compact small middle | 无正式成绩 |
| S02BD | trailing singleton last | 云端 A/B，无正式成绩 |
| S02BE | empty reduce zero | 无正式成绩 |
| S02BF | empty TilingKey 修复 | **正式 4001.200** |
| S02BG | FP32 middle-8 compact | **正式 3995.5995** |
| S02BH | strided full cores | 无正式成绩 |
| S02BI | fastPath3 Split-K | **正式 3982.789** |

### 6.5 2026-08-06：S02BJ～S02CT

| 版本 | 方向 | 提交/成绩状态 |
| --- | --- | --- |
| S02BJ | compact small middle | 无正式成绩 |
| S02BK | packed small middle | **正式 3984.7995** |
| S02BL | lower narrow workspace | 无正式成绩 |
| S02BM | split narrow workspace | **正式 4013.100** |
| S02BN | long-tail Split-K | **正式 4001.048** |
| S02BO | strided narrow full cores | 无正式成绩 |
| S02BP | strided narrow full cores 后继 | 无正式成绩 |
| S02BQ | strided grouped rows | 无正式成绩 |
| S02BR | strided grouped rows key5 | 无正式成绩 |
| S02BS | grouped scalar rows | **正式 3996.028** |
| S02BT | grouped padded rows | **正式 3995.380** |
| S02BU | strided grouped Split-K | 无正式成绩 |
| S02BZ | 阶段候选 | 有 ZIP，无完整正式成绩 |
| S02CA | Split-K 后继正式包 | **正式 3982.569** |
| S02CB | inner=2 compact 正确性模型 | 无；仅诊断/模型 |
| S02CD | 阶段候选 | **正式 3994.349** |
| S02CE | non-contiguous Split-K，output≤32 | 有 ZIP，无独立正式成绩 |
| S02CF | large power-of-two Split-K | **正式 3994.349** |
| S02CG | non-contiguous Split-K，output≤64 | 有 ZIP，无独立正式成绩 |
| S02CH | large-inner long chunk | 无正式成绩 |
| S02CI | fastPath4 full cores | 无正式成绩 |
| S02CJ | fastPath4 aligned rows | 无正式成绩 |
| S02CK | fastPath4 dtype-aligned rows | 无正式成绩 |
| S02CL | fastPath4 aligned rows min16 | **正式 3986.639** |
| S02CM | inner sweep 后继 | **正式 3987.7895** |
| S02CN | middle full rows | **正式 4010.864** |
| S02CO | middle vector finalize | 无正式成绩 |
| S02CP | middle aligned finalize | 无正式成绩 |
| S02CQ | BF16 fused accumulation | 本地候选，未上云/未提交 |
| S02CR | general strided Split-K | 本地候选，未上云/未提交 |
| S02CS | short-tail Split-K | 本地候选，未上云/未提交 |
| S02CT | last output≤16 Split-K | 本地候选，未上云/未提交 |

### 6.6 从历史最好 S02F 重建的 S03 候选

| 版本 | 方向 | 当前状态 |
| --- | --- | --- |
| S03A | 删除 BF16 中间量化往返 | 真机正确性 36/36；覆盖域聚合改善 3.3379%，但一项基线漂移 6.2389% 超过 3% 门槛，暂不合入、未打包 |
| S03B | fastPath4 低并行大归约通用 Split-K | 真机正确性 18/18；A/B/A 聚合改善 86.7105%，门禁通过；已合入 S03E |
| S03D | size-one 间隔下的物理连续归约路由 | 真机正确性 72/72；A/B/A 聚合改善 84.6465%，门禁通过；已合入 S03E |
| S03E | S03B + S03D，触发域互斥的组合候选 | CANN 8.5.0 隔离构建通过，组合正确性 126/126；Split-K 改善 86.7365%，singleton-gap 改善 84.5746%；已打包，未提交平台 |

S03E 首轮 singleton-gap 的 `fp32 fast2` 两次基线漂移 `3.5813%`，候选
本身仍改善 `78.7377%`。该组随后独立 A/B/A 复测，基线漂移降至
`0.1849%`、候选改善 `78.7486%`，因此最终门禁通过。完整证据见
`SQUARESUM_S03E_CLOUD_GATE_20260807.md` 和 `artifact/result.json`。

## 7. 有正式成绩的 ZIP 完整 SHA-256

| 版本 | 文件位置 | SHA-256 |
| --- | --- | --- |
| S01 | `提交相关材料/20260729/SquareSumV1.zip` | `d5d0407c7f81519dce36682d18104d1579bf96642ea898e0706c91550942dbb7` |
| S02E2 | `提交相关材料/20260730/SquareSumV1.zip` | `14b3384ed0a5a740808e008be0d7922bea9d5a27cd837ce8cfcc0dbd3d196ee9` |
| S02F | `提交相关材料/20260730/S02F_1635/SquareSumV1.zip` | `eecd9f1fd6b4c0617b6ec2ec632f24bb0f310d5a9d2f2125f03f6ec86ecfaf5b` |
| S02BA | `提交相关材料/20260805/S02BA/SquareSumV1.zip` | `23007e299e83d1116c783235c924c9cfde729b4fd15ba78a474e5aaf8dc3114b` |
| S02BB | `提交相关材料/20260805/S02BB/SquareSumV1.zip` | `dc7451016f341784e3083653dd9bccbf32b0345a3f9faa462e76162682289f7f` |
| S02BF | `提交相关材料/20260805/S02BF/SquareSumV1.zip` | `b5484c12895cff3165d865a3e557fce321221bd5c09fb66de1c4916a6b35196b` |
| S02BG | `提交相关材料/20260805/S02BG/SquareSumV1.zip` | `d3bc2703620c076ec35a173c558926e9d76868e3f05d1be12bca090bfa6d0247` |
| S02BI | `提交相关材料/20260805/S02BI/SquareSumV1.zip` | `bb3eccdfba6584ebac4d7d3f5b86c34e5579c935bf775e8c5ca0bc9046b2a565` |
| S02BK | `提交相关材料/20260806/S02BK/SquareSumV1.zip` | `d283db77bc37b46b64fd7e9ef0ccfa22ba8dd42397a622378c6674b69a7faca1` |
| S02BM | `提交相关材料/20260806/S02BM/SquareSumV1.zip` | `3893e200515214e6da8d1f12693b797e5dd66645915a7c1fc3c0732f5c572678` |
| S02BN | `提交相关材料/20260806/S02BN/SquareSumV1.zip` | `0595ab4ebe7cc9cb61d14e828d6d2dd5e599d3307341b906734be54f3e9926be` |
| S02BS | `提交相关材料/20260806/S02BS/SquareSumV1.zip` | `65ca55c54f3fe3a388411a109c024a7101ea0bc66e40be4959179af5396409fe` |
| S02BT | `提交相关材料/20260806/S02BT/SquareSumV1.zip` | `5e1aaec27c8465b44695130c6985d873c49acd807a6ad76ce8ff31c19d3dccf4` |
| S02CA | `提交相关材料/20260806/S02CA/SquareSumV1.zip` | `1209a5445514553cc1a0c4d243fb1764dc80a7b66ced427f7da82699fa10c117` |
| S02CD | `提交相关材料/20260806/S02CD/SquareSumV1.zip` | `5d847711252ae49f5ca6ba8b65c14983be703d4ae0493d323b5b61a90ef79d30` |
| S02CF | `提交相关材料/20260806/S02CF/SquareSumV1.zip` | `26693f15a8d7434d840018e948f66cd21e13bcbcf59f9a46617251407e09efc3` |
| S02CL | `提交相关材料/20260806/S02CL/SquareSumV1.zip` | `a4c82815ef2faf8e96fc6d310eac0ee4cc17d41cd0ece3ff1dee38f23891044b` |
| S02CM | `提交相关材料/20260806/S02CM/SquareSumV1.zip` | `15cb455648368c37098c871790488497dae433d1d4fe941b16615bb65117d1e0` |
| S02CN | `提交相关材料/20260806/S02CN/SquareSumV1.zip` | `1d04777575176a168860cda1be96b2f75779c7faaef401e78dfa4f8845d28e16` |

### 7.1 已打包、尚无正式平台成绩

| 版本 | 文件位置 | SHA-256 | 状态 |
| --- | --- | --- | --- |
| S03E | `提交相关材料/20260807/S03E/SquareSumV1.zip` | `dd4b22bc7eb5000c07357bd9d2355bb6b8e7bdf9e61cb7e8fb4fb3a5196f19b2` | 真机门禁通过，待上传 |

## 8. 后续追加规则

每次收到平台结果后，必须追加以下信息：

```text
版本 / 提交日期 / ZIP 路径 / ZIP SHA-256 /
Case1～Case5 原始状态与耗时 / prof_sum /
相对上一正式版本 / 相对历史最好 S02F /
对应候选目录 / 云端报告 / Git 提交 / 是否继续该路线
```

若出现 `Run failed`、`Incorrect op name` 或任一 Case 未通过，也要原样归档，
但不得填写或比较不完整的 `prof_sum`。只有包、本地结果或云端 A/B 时，必须
明确写“无正式平台成绩”。
