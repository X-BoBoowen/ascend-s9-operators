# 昇腾 AI 创新大赛 S9 算子优化

本仓库记录 Ascend 910B、CANN 社区版 8.5.0 环境下五道 S9
算子的通用实现、真机验证、性能实验和正式提交源码。

当前工作原则：

- 先满足题面全部 dtype、rank、轴、广播、空输入和非对齐约束；
- 不根据隐藏样例猜 shape，不写公开 Case 专用分支；
- 每题在独立实验目录中构建、安装、回归和打包；
- 只有平台 Case1–Case5 才能作为最终正确性和性能结论；
- Git 仓库保存源码、测试与文档，不保存 `.run`、ZIP、wheel 和
  profiler 数据库。

> 状态时间：2026-07-28 17:10（Asia/Shanghai）
>
> 当前阶段：`Concat`、`Greater`、`IndexAdd`、`Transpose` 已完成
> 本轮工程闭环并分别生成待测 ZIP；下一题为 `SquareSumV1`。

## 1. 当前状态

| 题目 | 工程状态 | 真机回归 | 当前提交包 | 平台状态 |
| --- | --- | ---: | --- | --- |
| Concat | 本轮完成 | 9 定向 + 100 随机 + 168 扩展 | 已生成并审计 | 新包待上传 |
| Greater | 本轮完成 | 26 定向 + 220 随机 + 9 个 int32 扩展；扩展集连续复跑 3 次 | 已生成并审计 | 新包待上传 |
| IndexAdd | 本轮完成 | 23 定向 + 170 随机 + 345 扩展 = 538 | 已生成并审计 | 新包已交用户评测 |
| Transpose | 本轮完成 | 48 定向 + 200 随机 + 152 扩展 + 84 边界 = 484 | 已生成并审计 | 新包待上传 |
| SquareSumV1 | 已有通用正确版本 | 46 定向 + 150 随机 + 726 扩展 = 922 | 旧候选保留 | 后续继续 |

这里的“本轮完成”表示：

1. 正式算子名称、Host、Kernel、Tiling 和 CMake 契约一致；
2. 在 910B4、CANN 8.5.0 环境重新构建出 `.run`；
3. 安装该 `.run` 后，用自定义扩展直接调用正式算子；
4. 完成定向、随机、扩展和大规模压力回归；
5. ZIP 结构、文件权限、源码哈希和 `.run` 哈希已审计；
6. 本地 ZIP 与云端发布 ZIP 完全一致。

它不表示已经获得新的榜单成绩。四题的新包必须逐题上传，平台
Case1–Case5 全 Pass 后才能确认隐藏用例闭环。

## 2. 本轮四题的提交清单

正式 ZIP 位于本地仓库外：

```text
D:\29722\Desktop\GCC\提交相关材料\
|-- Concat.zip
|-- Greater.zip
|-- IndexAdd.zip
`-- Transpose.zip
```

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `Concat.zip` | 410639 B | `b43e88f82ba5230f9172b1f1f2f4c07381d4f14cd1fb5a4de3cf3871c55d25b2` |
| `Greater.zip` | 398168 B | `316797810d06b57d18c898d1fe449c1b0b51565c6a842845dded75524f7f868d` |
| `IndexAdd.zip` | 417920 B | `cf8c08a60d3b356686a07e136766dd06128afa87dccf67d9c9940330843d990b` |
| `Transpose.zip` | 410044 B | `b564a3724999cd2f3ef1b2ebf8c6e75c8c72778ffaa659125883130ac1d541cc` |

包内 `.run`：

| 题目 | `.run` SHA-256 |
| --- | --- |
| Concat | `9dca80b07e85eacf9b90ac98349f64bf0ba0db19565c440f0c16f12cbf1ca873` |
| Greater | `d8ab6f81aa1eaa775cbe69078db09901037ccad667216697371475a42dbb4298` |
| IndexAdd | `58a5be8bbbc8db62708973fea21bd6630e1d938ae9c8a4ad28132f3014ce679d` |
| Transpose | `8685eaffd9787893a659c9b1b7424aa88c2881d10921c8246fda7ce4d717709e` |

四个 ZIP 均只有一个顶层 `<Operator>_zip/`，包含 `op_host/`、
`op_kernel/` 和一个非空 `custom_opp_euleros_aarch64.run`。
`IndexAdd` 的 `.run` 在 ZIP 中保持 `0750`，其余三个为 `0755`。

## 3. 仓库结构

```text
.
|-- submission-src/
|   |-- Concat/             # 本轮正式源码
|   |-- Greater/            # 本轮正式源码
|   |-- IndexAdd/           # 本轮正式源码
|   |-- Transpose/          # 上一轮稳定源码
|   `-- SquareSumV1/        # 上一轮稳定源码
|-- validation/
|   |-- Concat/             # 定向、随机、profile 与扩展封装
|   |-- Greater/
|   |-- IndexAdd/
|   `-- Transpose/
|-- case_910b/              # 官方公开 case 与运行脚本
|-- operator-descriptors/   # 开发期工程描述
|-- EXPERT_OPINION_SYNTHESIS_20260727.md
|-- EXPERT_REVIEW_TRACKER.md
|-- RELEASE_SNAPSHOT_20260728_FIRST_THREE.md
`-- README.md
```

正式源码入口始终是 `submission-src/<Operator>/`。包内源码已与这里
逐文件计算 SHA-256，四题全部匹配。

## 4. Concat

### 4.1 题面覆盖

- dtype：`float32`、`float16`、`int32`、`int8`；
- 动态输入列表；
- rank 1–8；
- 正轴和负轴；
- 任意合法拼接轴；
- 首个或中间输入为空；
- 非 32B 对齐；
- 当前 Tiling 容量最多 256 个输入。

### 4.2 实现

Host 将任意拼接轴折叠为：

```text
[outer, input_dim * inner] -> [outer, sum(input_dim) * inner]
```

以字节为单位传递每个输入的行宽，避免不同 dtype 下重复维护元素
对齐逻辑。Kernel 通过 `ListTensorDesc` 读取动态输入地址。

当前有两类工作划分：

- `outer` 足够时按行分核；
- `outer` 很小而总字节数很大时按全局 32B 工作块分核，避免超宽单行
  只占用一个核。

大规模多行搬运使用 `DataCopyPad` 二维搬运；行跨度超过硬件 stride
上限时自动回退逐行搬运。Host 根据总字节量和实际 Vector Core 数量
动态选择核数，小任务避免过度起核，大任务可超过旧版固定 16 核。

### 4.3 验证

- 9 个定向用例；
- 100 个随机用例；
- 168 个扩展用例；
- 全部与 `torch.cat` 逐位相等。

覆盖内容包括 33/256 输入、rank 1–8、dim 0/中间/末轴、负轴、
首个空输入、多个空输入、非对齐宽度、宽行分块和大 outer。

本地事件计时只用于版本 A/B，不等价于平台成绩。代表性 p50：

| 场景 | p50 |
| --- | ---: |
| public 非对齐 | 110.09 us |
| 对齐多行 | 107.22 us |
| dim0 大输入 | 101.76 us |
| outer=128 大输入 | 101.04 us |
| 256 输入 | 484.60 us |

## 5. Greater

### 5.1 题面覆盖

- dtype：`float32`、`bfloat16`、`float16`、`int32`、`int8`；
- 输出严格为 bool；
- rank 0–8；
- NumPy/PyTorch 广播；
- 同形、标量、末维广播、交错广播；
- NaN、Inf、`+0/-0`、相等值；
- 空输出和非 32B 对齐。

### 5.2 实现

Host 计算广播后的输出 shape，并为两个输入生成 stride。被广播的维度
stride 为 0。Kernel 识别并优化：

- 连续同形输入；
- 标量输入；
- 可连续搬运的重复 run；
- 通用坐标映射广播。

计算按 4096 元素分块，最多使用 40 个 Vector Core。浮点路径使用
原生 `Compare(GT)`，因此 NaN 参与比较为 false。int8 先无损转 half
再比较。int32 不转 float，避免 24 位有效精度造成全位宽整数误判；
最终版本通过 `max(self, other) == self` 与 `self == other` 两个掩码
组合出严格大于。

### 5.3 int32 修复记录

早期候选曾尝试用高/低 16 位掩码组合比较，在随机全 32 位 bit pattern
中出现 621 个错误。该候选未进入正式包。

最终 int32 路径在以下扩展集全部通过，并连续复跑 3 次：

- 边界值笛卡尔积；
- 257×1021 随机全位宽 bit pattern；
- 200003 个全相等元素；
- 左/右标量极值；
- trailing、interleaved 和 rank-8 广播；
- 空输出。

### 5.4 验证

- 26 个定向用例；
- 220 个随机用例；
- 9 个 int32 扩展用例；
- 扩展集额外连续复跑 3 次；
- 共 255 个不同用例全部通过。

本地扩展调用事件计时约 23–25 us，仅用于回归和 A/B。

## 6. IndexAdd

### 6.1 题面覆盖

- self/source dtype：`float32`、`bfloat16`、`float16`、`int32`、`int8`；
- index dtype：`int32`；
- rank 1–8；
- 任意正/负 dim；
- 重复、乱序和空 index；
- 未命中输出保持 self；
- int8/int32 溢出回绕；
- 题面 index 上限 8000；
- 非 32B 对齐 inner。

### 6.2 实现

Host 将张量折叠为：

```text
outer = product(shape[:dim])
dimSize = shape[dim]
inner = product(shape[dim + 1:])
```

Kernel 自身从 self 构造完整输出，不依赖 Python wrapper 预复制。每个
任务负责一组互不重叠的输出行，因此重复 index 不存在跨核写冲突。

核心优化：

- `inner` 按 256 元素分块；
- `dim` 每组最多 64 行；
- 最多 40 核；
- 行宽与地址满足条件时使用二维 `DataCopyPad` 批量搬入/搬出；
- int8 使用分批累加路径，保持逐位回绕语义；
- float16/bfloat16 在需要时使用 float 累加缓冲；
- index 在核内缓存；
- 当 index 较大且 inner 有多个分块时，为一个 dim 组预构建命中列表，
  在多个 inner chunk 间复用；
- 只有浮点 dimGroups≥8、整数 dimGroups≥16 时才启用命中复用，
  该门限来自通用压力矩阵 A/B，不绑定某个隐藏 shape；
- 连续任务分配减少大 shape 下的重复扫描与负载不均衡。

当前使用 6 个 TilingKey，组合区分普通/批量 int8、命中复用和单
inner chunk 快路。

### 6.3 验证

- 23 个定向用例；
- 170 个随机用例；
- 3 个 seed 共 345 个扩展用例；
- 合计 538 个用例全部通过；
- 另有大 index、大 dim、大 inner、outer>1 的压力和性能矩阵。

代表性本地事件计时：

| 场景 | 中位数 |
| --- | ---: |
| FP16, M=8000, inner=4096 | 721.37 us |
| FP32, outer=2, M=8000, inner=2048 | 845.98 us |
| BF16, outer=2, M=8000, inner=2048 | 756.24 us |
| FP32, M=8000, inner=4096 | 819.30 us |
| int8, dim=4096, M=1024, inner=2048 | 129.68 us |

一个“int8 批量 source 队列 + 无条件命中列表”候选曾在随机 case18
出现 41 个错误，已经丢弃，未进入正式源码和提交包。

## 7. Transpose

### 7.1 题面覆盖

- dtype：`float32`、`float16`、`int32`、`int8`；
- rank 1–6；
- 任意合法 `dims`，支持正轴和负轴；
- 单位维、空输入和非 32B 对齐；
- `N`、`N2` 至 10000，`N3`–`N5` 至 1000；
- 结果与 `torch.permute(...).contiguous()` 逐位一致。

### 7.2 实现

Host 先合并保持不变的前缀和可连续搬运的尾部，再根据排列结构选择
5 个 TilingKey：

- 通用坐标映射、identity 和连续 run 搬运；
- 单矩阵、16×16 对齐的 fp16 硬件 Transpose；
- 循环置换折叠后的二维分块转置；
- 输出末维可聚合时的 gather 快路；
- 多 batch、16×16 对齐的 fp16 硬件 Transpose。

通用路径不依赖隐藏 shape；矩阵行列、batch 和 stride 均来自 Host
Tiling。非 fp16、非对齐尾块和不能安全折叠的任意排列均保留正确回退。

### 7.3 验证与拒绝候选

- 48 个定向用例；
- 200 个固定 seed 随机用例；
- 152 个循环置换、任意排列和特殊 bit pattern 扩展用例；
- 84 个阈值、行列分块和 3–5 维双向旋转边界用例；
- 合计 484 个用例，在旧稳定构建和最终无缓存构建上各完整通过一次。

曾实现 CANN 8.5 增强 Transpose API 的 UB 大块候选。它虽然通过同一
484 例正确性回归，但设备级 profiling 显示明显回退：

| 场景 | 稳定版平均 kernel | 增强候选平均 kernel |
| --- | ---: | ---: |
| float32, 128×256 | 25.756 us | 243.971 us |
| int8, 128×256 | 19.536 us | 337.596 us |
| fp16, 127×257 尾块 | 20.913 us | 480.164 us |

因此增强候选没有进入正式源码、Git 提交或发布 ZIP。

## 8. 源码哈希

### 8.1 Concat

| 文件 | SHA-256 |
| --- | --- |
| `op_host/CMakeLists.txt` | `d1b100115b8c34135ccdfc54f91597847a7823ec76cdca995e2b80f5c6092cd2` |
| `op_host/concat.cpp` | `cd98d736052c05987bb3ebc9fcb32e0a43694e8ed35961cfba45772e384741ce` |
| `op_host/concat_tiling.h` | `5b07ef0615f269c0ea951977916152748bc16fef7d652c276619d287620efb9a` |
| `op_kernel/CMakeLists.txt` | `10b4df9e22540a42e443602357cf8a7bfa71b4c9c7198fa7da6b3f4343b00118` |
| `op_kernel/concat.cpp` | `6ff66e8845a1dd60f148536bd3770677a1627d8a9d9d89a9371b8e0ab86353d8` |

### 8.2 Greater

| 文件 | SHA-256 |
| --- | --- |
| `op_host/CMakeLists.txt` | `d1b100115b8c34135ccdfc54f91597847a7823ec76cdca995e2b80f5c6092cd2` |
| `op_host/greater.cpp` | `1f1b69f6d65128d1c7746a38af2ceb9df501f2f69c95eefa486ae27d60b62a52` |
| `op_host/greater_tiling.h` | `964b0256aef614b62e285afbffd6c967507276aac52493cd0047121b68f93c6b` |
| `op_kernel/CMakeLists.txt` | `de2557844234c8ca7d5952ec124d4c53f0196d9066d5e172f86f62808fb776a6` |
| `op_kernel/greater.cpp` | `0f2aaab2f2a0d804281a177bed09d915bd329d55b402c5fa6a596aef8380c5e0` |

### 8.3 IndexAdd

| 文件 | SHA-256 |
| --- | --- |
| `op_host/CMakeLists.txt` | `d1b100115b8c34135ccdfc54f91597847a7823ec76cdca995e2b80f5c6092cd2` |
| `op_host/index_add.cpp` | `33e4eb4c27f633fcc8031a1ef4e353ff1e98f86ece32d666ff0e424f20831476` |
| `op_host/index_add_tiling.h` | `85d66f140d24855b97d281a8b6f715b81ead115fab7e2aa976a57614616ccb6b` |
| `op_kernel/CMakeLists.txt` | `fb5e4b00af77bb885d29dff3faa1f1c094e79f19f6a366eca0d381a209819627` |
| `op_kernel/index_add.cpp` | `e9ee4fdd157ef6a92ae843eef49eb8179a558885695dc0bdcbfcece82bca7aa8` |

### 8.4 Transpose

| 文件 | SHA-256 |
| --- | --- |
| `op_host/CMakeLists.txt` | `d1b100115b8c34135ccdfc54f91597847a7823ec76cdca995e2b80f5c6092cd2` |
| `op_host/transpose.cpp` | `d68ec597fa68861a04f5da11f17b481674500567328fef57500a387786fdc260` |
| `op_host/transpose_tiling.h` | `71d17bd19c58ada5ee29e6a1b3640e2f3c97ab8451cb9de6611d9a8befd4d5e1` |
| `op_kernel/CMakeLists.txt` | `dc5e6d36cbd092eed6fdc008a40896ede683299a3affeb91d693343bd6f29597` |
| `op_kernel/transpose.cpp` | `2684abf308169407a7cff07c7b82ba2e3c53e10f80a1ae9c26763c410c682293` |

## 9. 云端状态

云环境：

```text
ModelArts: my_env2
NPU: Ascend 910B4
CANN: /home/ma-user/Ascend/cann-8.5.0
工作根目录: /home/ma-user/work/s9
```

最终实验：

```text
/home/ma-user/work/s9/experiments/concat_dynamic_rows_20260728_1417
/home/ma-user/work/s9/experiments/greater_int32_masks_20260728_1439
/home/ma-user/work/s9/experiments/indexadd_hitreuse_20260728_1514
/home/ma-user/work/s9/experiments/transpose_release_20260728_1703
```

云端发布包：

```text
/home/ma-user/work/s9/releases/concat_20260728_1446/Concat.zip
/home/ma-user/work/s9/releases/greater_20260728_1511/Greater.zip
/home/ma-user/work/s9/releases/indexadd_20260728_1622/IndexAdd.zip
/home/ma-user/work/s9/releases/transpose_20260728_1706/Transpose.zip
```

截至 2026-07-28 17:07，四个云端 ZIP 均已重新读取，并与本地
`提交相关材料/` 对比：文件大小、ZIP SHA-256、包内每个源码 SHA-256
和 `.run` SHA-256 全部一致。

所有云端命令通过本地 `ssh_visible.ps1` 执行，并记录在：

```text
/home/ma-user/work/s9/codex-visible-terminal.log
```

## 10. 平台已知结果与解释

用户此前提供的结果：

| 题目 | Case1 | Case2 | Case3 | Case4 | Case5 | 总和 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Concat | 15.7005 | 46.871 | 123.5425 | 116.7625 | 847.587 | 1150.4635 |
| Greater | 5.312 | 39463.772 | 2261.36 | 94.512 | 210.032 | 42034.988 |
| IndexAdd | 9.7805 | 41352.976 | 90.9915 | 2564.231 | 80160.552 | 124178.531 |
| Transpose | 4.5805 | 257.445 | 2466.05 | 2197.384 | 34623.812 | 39549.2715 |

这些结果对应四题本轮最终 ZIP 之前的版本，不能用来评价本 README
记录的新包。新包上传后必须记录：

- 上传时间；
- ZIP SHA-256；
- Case1–Case5 Pass/Fail；
- 每个 Case 的耗时；
- `prof_sum`；
- 排名变化；
- 若失败，保存平台结果文本，不猜隐藏数据。

## 11. 下一次工作如何接续

### 11.1 先收平台反馈

按顺序上传：

1. `Concat.zip`
2. `Greater.zip`
3. `IndexAdd.zip`
4. `Transpose.zip`

每题上传后先确认五个 Case 全 Pass，再比较耗时。不要同时修改多个
算子，否则无法定位性能变化。

### 11.2 复核本地工作区

```bash
git status --short
git log -1 --oneline
git diff --check
```

正式源码只从 `submission-src/` 取。不要从 `candidates/`、
`current-submission-extracted-*` 或历史实验目录覆盖正式版本。

### 11.3 云端执行纪律

显式加载 CANN 8.5.0：

```bash
source /home/ma-user/Ascend/cann-8.5.0/set_env.sh
```

登录欢迎信息可能仍显示镜像预置的旧 CANN 字样，不能据此判断实际构建
版本。构建、安装和运行前都要核对环境变量指向 8.5.0。

每轮实验：

1. 从最终实验复制到带时间戳的新目录；
2. 保存旧源码哈希；
3. 只改一个可检验假设；
4. 重建 `.run` 并安装；
5. 先定向，再随机，再扩展，再压力；
6. 做同形状 A/B；
7. 只有正确且有稳定收益才提升为正式源码；
8. 独立无缓存重建后再打包。

### 11.4 后续题目

下一题为 `SquareSumV1`。先重读题面、Excel 中的官方链接和两轮专家
意见，再核对当前正式源码、旧平台结果和 922 例验证矩阵。每个优化
假设必须在独立实验目录做正确性与同形状 A/B，禁止把未经验证的候选
覆盖 `submission-src/SquareSumV1`。

## 12. 安全与仓库卫生

禁止提交：

- SSH 私钥、云账号、令牌和 `.env`；
- `.run`、`.so`、wheel、目标文件；
- ZIP、profiler 数据库和构建缓存；
- 个人材料、代金券记录和联系方式。

提交前检查：

```bash
git diff --check
git ls-files | grep -Ei '\.(pem|key|run|so|whl|zip)$'
```

## 13. 结论边界

当前能够确认的是：前四题的正式源码、云端构建产物、本地提交包和验证
证据已形成可复现闭环，并已逐哈希对齐。

当前不能确认的是：新包在官方隐藏 Case 上的最终耗时、排名和是否已经
进入奖励区间。所有平台结论必须等待用户回传五个 Case 的实际结果。
