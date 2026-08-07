# SquareSumV1 S03M–S03O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从正式历史最好 S02F 干净构建 S03M、S03N、S03O，依次覆盖相邻保留行分组、任意最内归约长度和不对齐 inner，并以比赛同口径 `msprof` 证明性能，最终把五个正式 Case 的稳定中位耗时降至 `0.30 × 3985.330 = 1195.599 μs` 以内。

**Architecture:** S03M 只从冻结 S02F 移植 S03L 的 key 7 分组路径；S03N 只在 S03M 通过后把幂次归约扩展为任意正长度；S03O 只在 S03N 通过后用 key 8 和二维 DMA 将非 8 对齐 inner 搬入按 32 字节对齐的 UB 行。每阶段建立独立源码快照、静态路由模型、设备正确性矩阵和正式同构 A/B/A 门禁，失败阶段不得晋升。

**Tech Stack:** Ascend C、C++、CANN 社区版 8.5.0、Ascend 910B、Python 3.9、PyTorch NPU、Bash、`msprof`、Git main。

## Global Constraints

- 所有功能、构建和性能验证使用 Ascend 910B 与 CANN 社区版 8.5.0。
- FP16、BF16、FP32 全部正确，保留输入 dtype 的平方语义和题目规定的输出 dtype。
- 不读取或分支于隐藏 Case 编号、隐藏 shape 常量、榜单结果或历史评分。
- 路由只使用 shape、axis、stride、dtype、容量和硬件指令边界。
- 比赛性能口径固定为 30 次调用、排除 30 个 `aclnnMul`、SquareSumV1 Task Duration 取 `[10:30]` 中位数。
- 每阶段执行 Baseline/Candidate/Baseline；目标点逐点改善不少于 50%，控制点回退不超过 3%，两轮基线漂移不超过 3%。
- S03M 基线固定为 S02F；S03N 基线为通过门禁的 S03M；S03O 基线为通过门禁的 S03N。
- 任一正确性失败、构建失败、路由覆盖失败、设备异常、超时或门禁失败均停止该阶段，不更新 `submission-src/SquareSumV1`，不生成正式 ZIP。
- `artifact/` 每次运行前清空，只保存最新运行的 `artifact/result.json`、`artifact/run.log` 及当次明细。
- 本地最终集成和推送落在 `main`；云端通过 `ssh_visible.ps1` 执行，使命令显示在 JupyterLab 终端日志。
- 保留现有大量未跟踪实验资料，不清理、不覆盖、不批量加入 Git。

---

## File Map

- `baselines/squaresum_s02f_global_best_20260806/SquareSumV1/`: 冻结的正式历史最好源码，只读。
- `candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807/`: S03M 源码、来源说明和门禁证据。
- `candidates/squaresum_s03n_arbitrary_grouped_reduce_20260807/`: S03N 源码、来源说明和门禁证据。
- `candidates/squaresum_s03o_unaligned_grouped_rows_20260807/`: S03O 源码、来源说明和门禁证据。
- `submission-src/SquareSumV1/`: 最近一个通过全部内部门禁的候选；未过门禁时保持原样。
- `diagnostics/squaresum_s03m_s03o_static_20260807.py`: 三阶段纯 CPU 路由、容量、任务覆盖和源码注册审计。
- `diagnostics/squaresum_s03m_s03o_correctness_20260807.py`: 三 dtype、轴语义和阶段边界的设备正确性矩阵。
- `diagnostics/squaresum_official_profile_matrix_20260806.json`: 增加 `s03m_target/control`、`s03n_target/control`、`s03o_target/control` 性能层级。
- `diagnostics/run_squaresum_s03m_s03o_gate_20260807.sh`: 构建、隔离安装、正确性、A/B/A、门禁和原子结果写入。
- `diagnostics/pull_squaresum_remote_artifact_20260807.ps1`: 将远端最新阶段产物下载到临时目录，验证阶段后原子替换本地 `artifact/`。
- `diagnostics/compare_squaresum_official_matrices_20260806.py`: 复用现有比较器，门槛明确传入 `50/3/3`。
- `SQUARESUM_S03M_CLOUD_GATE_20260807.md`、`SQUARESUM_S03N_CLOUD_GATE_20260807.md`、`SQUARESUM_S03O_CLOUD_GATE_20260807.md`: 各阶段证据报告；只记录实际结果。
- `SQUARESUMV1_ATTEMPT_SUBMISSION_SCORE_ARCHIVE.md`: 仅在生成正式提交包或收到平台成绩时追加记录。
- `D:/29722/Desktop/GCC/提交相关材料/20260807/S03M/SquareSumV1.zip`、`S03N/SquareSumV1.zip`、`S03O/SquareSumV1.zip`: 只为对应的通过阶段生成。

---

### Task 1: 建立隔离工作区和 S03M 失败测试

**Files:**
- Create: `candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807/ORIGIN.md`
- Create: `candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807/SquareSumV1/`
- Create: `diagnostics/squaresum_s03m_s03o_static_20260807.py`

**Interfaces:**
- Consumes: 冻结目录 `baselines/squaresum_s02f_global_best_20260806/SquareSumV1`。
- Produces: `route(stage, ...) -> RouteDecision`，其中 `RouteDecision` 包含 `enabled`、`key`、`grouped_width`、`scheduled_tasks`、`aligned_inner`；后续三个阶段共享这个审计接口。

- [ ] **Step 1: 使用隔离工作树**

按 `superpowers:using-git-worktrees` 建立分支 `perf/squaresum-s03m-s03o-20260807`。工作树必须从当前 `main` 的 `df8d58c` 或其后继提交创建；执行前确认主工作树只有既有未跟踪材料，没有未提交的 tracked 修改：

```powershell
git -C D:\29722\Desktop\GCC\s9-work status --short
git -C D:\29722\Desktop\GCC\s9-work diff --quiet
git -C D:\29722\Desktop\GCC\s9-work diff --cached --quiet
```

- [ ] **Step 2: 从 S02F 建立逐字节相同的 S03M 快照**

机械复制五个比赛文件，不复制构建产物；随后用 SHA-256 证明副本等于 S02F：

```powershell
$src = 'baselines\squaresum_s02f_global_best_20260806\SquareSumV1'
$dst = 'candidates\squaresum_s03m_s02f_strided_grouped_clean_20260807\SquareSumV1'
New-Item -ItemType Directory -Path $dst -Force | Out-Null
Copy-Item -LiteralPath "$src\op_host" -Destination $dst -Recurse
Copy-Item -LiteralPath "$src\op_kernel" -Destination $dst -Recurse
git diff --no-index --exit-code -- $src $dst
```

`ORIGIN.md` 固定写明：源是 S02F，正式成绩 `3223.995 μs`，S03J/S03K 不进入源码，S03L 仅作为 key 7 增量参考。

- [ ] **Step 3: 编写 S03M 先失败的静态测试**

在 `squaresum_s03m_s03o_static_20260807.py` 中定义：

```python
@dataclass(frozen=True)
class RouteDecision:
    enabled: bool
    key: int
    grouped_width: int
    scheduled_tasks: int
    aligned_inner: int


def route(stage, *, input_elements, output_elements, reduce_elements,
          reduce_rank, last_reduce, inner, grouped_dim, outer_rows,
          input_type_bytes):
    elements_per_block = 32 // input_type_bytes
    aligned_inner = (
        (inner + elements_per_block - 1) // elements_per_block
        * elements_per_block
    )
    power_of_two = last_reduce > 1 and not (
        last_reduce & (last_reduce - 1)
    )
    allow_arbitrary = stage in {"s03n", "s03o"}
    padded = stage == "s03o" and inner % 8 != 0
    row_stride = aligned_inner if padded else inner
    base_ok = (
        input_elements >= (1 << 20)
        and reduce_elements >= 2048
        and reduce_rank > 1
        and last_reduce > 1
        and reduce_elements % last_reduce == 0
        and output_elements % inner == 0
        and (allow_arbitrary or power_of_two)
        and (padded or inner % 8 == 0)
    )
    for width in (8, 4, 2):
        tasks = outer_rows * ((grouped_dim + width - 1) // width)
        buffer_elements = width * last_reduce * row_stride
        if (
            base_ok
            and grouped_dim >= width
            and buffer_elements <= 8192
            and width * inner <= 1024
            and width * row_stride <= 8192
            and tasks >= 32
            and width * last_reduce <= 65535
        ):
            return RouteDecision(
                True, 8 if padded else 7, width, tasks, aligned_inner
            )
    return RouteDecision(False, 0, 0, 0, aligned_inner)
```

测试必须断言 S03M 对 `last_reduce=128, inner=16` 选择 key 7，对 `last_reduce=127`、`inner=15`、任务 31、UB 超限和输入门槛以下回退。脚本提供 `--model-only` 和 `--source PATH` 两种模式；本任务只运行前者，下一任务先用后者证明 key 7 源码检查失败。

- [ ] **Step 4: 运行纯路由模型并确认通过**

Run:

```powershell
python diagnostics\squaresum_s03m_s03o_static_20260807.py --stage s03m --model-only
```

Expected: 零退出，S03M 路由正例、回退边界、宽度和任务数断言全部通过。

- [ ] **Step 5: 提交测试和干净来源快照**

```powershell
git add candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807 diagnostics/squaresum_s03m_s03o_static_20260807.py
git commit -m "test: define clean S03M grouped-row gate"
```

---

### Task 2: 仅向 S02F 移植 S03M key 7

**Files:**
- Modify: `candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807/SquareSumV1/op_host/square_sum_v1.cpp`
- Modify: `candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807/SquareSumV1/op_kernel/square_sum_v1.cpp`
- Modify: `candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807/SquareSumV1/op_kernel/CMakeLists.txt`

**Interfaces:**
- Consumes: S02F 现有 tiling 数据字段，不新增 tiling struct 字段。
- Produces: Host 选择 key 7；`KernelSquareSumV1<..., STRIDED_GROUPED_ROWS=true>` 调用 `ProcessStridedGroupedRows()`；不满足条件时与 S02F 完全同路由。

- [ ] **Step 1: 加入源码注册断言并确认测试先失败**

`--source` 模式读取 Host、Kernel 和 Kernel CMake，要求包含 `stridedGroupedRows`、`ProcessStridedGroupedRows`、`TILING_KEY_IS(7)` 和 `--tiling_key=1,2,3,4,5,7`，同时禁止 `middleFullRows`、`ProcessMiddleFullRows`、`TILING_KEY_IS(6)`。先运行：

```powershell
python diagnostics\squaresum_s03m_s03o_static_20260807.py --stage s03m --source candidates\squaresum_s03m_s02f_strided_grouped_clean_20260807\SquareSumV1
```

Expected: 非零退出，错误明确指出 key 7 源码标志不存在；纯路由模型仍通过。

- [ ] **Step 2: 加入 Host 路由和任务调度**

在 S02F 的 `groupedVector8` 计算之后加入 `stridedGroupedRows`。条件逐项固定为：

```cpp
fastPath == 4U && reduceMode == 0U &&
inputElements >= (1U << 20U) && reduceElements >= 2048U &&
reduceRank > 1U && innerElements > 0U &&
innerElements % 8U == 0U &&
outputElements % innerElements == 0U &&
reduceInputStrides[reduceRank - 1U] == innerElements &&
lastReduceDim > 1U &&
(lastReduceDim & (lastReduceDim - 1U)) == 0U &&
reduceElements % lastReduceDim == 0U
```

用 `SafeMultiply(lastReduceDim, innerElements, rowElements)`、从后向前查找 `outputInputStrides[axis] == rowElements` 的非单例保留轴，按宽度 `8,4,2` 选择第一个同时满足以下条件的配置：

```cpp
width * rowElements <= 8192U;
width * innerElements <= 1024U;
outerOutputRows * ceil(groupedOutputDim / width) >= 32U;
```

`scheduledOutputs` 对 key 7 等于 `stridedGroupedTasks`，`blockDim=min(tasks, 48)`，TilingKey 优先级为 `stridedGroupedRows ? 7U : S02F原有选择`。

- [ ] **Step 3: 加入 Kernel 模板开关和幂次行归约辅助函数**

模板签名固定为：

```cpp
template <typename T, uint32_t CHUNK, bool TREE_FINALIZE,
          bool GROUPED_VECTOR8, bool STRIDED_GROUPED_ROWS = false>
class KernelSquareSumV1;
```

`Process()` 在 reduceMode 和旧 fastPath 之前执行：

```cpp
if constexpr (STRIDED_GROUPED_ROWS) {
    ProcessStridedGroupedRows();
    return;
}
```

将 S02F 的幂次树循环抽为：

```cpp
__aicore__ inline void ReduceRowsInPlace(
    AscendC::LocalTensor<float> values,
    const uint32_t rows,
    const uint32_t rowStride)
{
    for (uint32_t activeRows = rows;
         activeRows > 1U;
         activeRows >>= 1U) {
        const uint32_t halfRows = activeRows >> 1U;
        AscendC::Add(
            values,
            values,
            values[halfRows * rowStride],
            halfRows * rowStride);
    }
}
```

原 `ReduceRowsInto` 调用该函数后再累加，保持旧路径数值顺序不变。

- [ ] **Step 4: 加入 `ProcessStridedGroupedRows`**

从 `candidates/squaresum_s03l_strided_grouped_rows_20260807/SquareSumV1/op_kernel/square_sum_v1.cpp` 的 `ProcessStridedGroupedRows` 提取完整实现，但不带入 `MIDDLE_FULL_ROWS`、`ProcessMiddleFullRows`、S03J/S03K buffer 修改或 fast2 路由。

每个任务执行：计算 `activeRows`；一次连续 GM→UB 搬入 `activeRows * lastReduceDim * inner`；按 dtype 保持原平方语义；对每个分组输出行执行 `ReduceRowsInPlace(..., lastReduceDim, inner)`；累加到 FP32 `accumulateLocal[row * inner]`；最后一次连续写回 `activeRows * inner`。Kernel 内重算出的 `groupedWidth` 和任务数必须与 Host 的 8/4/2、32-task、8192/1024 容量规则逐项相同；不能出现 Kernel 静默返回的 Host 可达状态。

- [ ] **Step 5: 注册 key 7 并运行静态测试**

`op_kernel/CMakeLists.txt` 固定为：

```cmake
add_ops_compile_options(SquareSumV1 OPTIONS --tiling_key=1,2,3,4,5,7)
```

入口新增：

```cpp
} else if (TILING_KEY_IS(7)) {
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.outputElements == 0) return;
    KernelSquareSumV1<DTYPE_INPUT, NORMAL_CHUNK, false, false, true> op;
    op.Init(input, output, workspace, tilingData);
    op.Process();
}
```

Run:

```powershell
python diagnostics\squaresum_s03m_s03o_static_20260807.py --stage s03m --source candidates\squaresum_s03m_s02f_strided_grouped_clean_20260807\SquareSumV1
git diff --no-index --stat -- baselines\squaresum_s02f_global_best_20260806\SquareSumV1 candidates\squaresum_s03m_s02f_strided_grouped_clean_20260807\SquareSumV1
```

Expected: 静态测试通过；只有 Host、Kernel、Kernel CMake 三个文件有差异；无 key 6、`middleFullRows`、`ProcessMiddleFullRows`。

- [ ] **Step 6: 提交 S03M 实现**

```powershell
git add candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807
git commit -m "perf: add clean S03M grouped-row kernel"
```

---

### Task 3: S03M 设备正确性和正式同构门禁

**Files:**
- Create: `diagnostics/squaresum_s03m_s03o_correctness_20260807.py`
- Modify: `diagnostics/squaresum_official_profile_matrix_20260806.json`
- Create: `diagnostics/run_squaresum_s03m_s03o_gate_20260807.sh`
- Create: `diagnostics/pull_squaresum_remote_artifact_20260807.ps1`
- Create after real run: `SQUARESUM_S03M_CLOUD_GATE_20260807.md`
- Create after real run: `candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807/evidence/*.json`

**Interfaces:**
- Consumes: S03M candidate, S02F baseline, existing `build_squaresum_source_20260806.sh`, profile runner and comparison script。
- Produces: 原子写入的 `artifact/result.json`，其中 `passed` 只有在构建、正确性、目标与控制 A/B/A 全部通过时为 `true`。

- [ ] **Step 1: 编写正确性矩阵**

脚本参数固定为 `--stage {s03m,s03n,s03o} --label LABEL`。S03M 覆盖三 dtype，并至少包含：分组维 `7/8/9/15/16/17/31/32/33`，last reduce `64/128/256/512`，inner `8/16/24/32/64`，任务 `31/32/33`，输入门槛前/等于/后，负轴、无序轴、`keep_dims`、singleton gap；回退点包含 last reduce `63/65/127/129` 和 inner `7/15/17`。参考值固定为：

```python
expected = torch.sum(torch.square(input_cpu), dim=axes, keepdim=keep_dims)
torch.testing.assert_close(
    actual, expected,
    rtol={torch.float16: 4e-3, torch.bfloat16: 4e-2,
          torch.float32: 4e-4}[dtype],
    atol={torch.float16: 4e-3, torch.bfloat16: 4e-2,
          torch.float32: 4e-4}[dtype],
    equal_nan=True,
)
```

单阶段门禁不得只运行新增矩阵；S03M 候选必须依次执行下列七套：

```bash
python3 validation/SquareSumV1/test_matrix.py
python3 validation/SquareSumV1/bf16_semantic_probe.py
python3 validation/SquareSumV1/random_matrix.py
python3 validation/SquareSumV1/extended_matrix.py
python3 diagnostics/squaresum_s03j_boundary_correctness_20260807.py
python3 diagnostics/squaresum_s03k_middle_full_rows_correctness_20260807.py S03M
python3 diagnostics/squaresum_s03m_s03o_correctness_20260807.py --stage s03m --label S03M
```

前六套已有期望总数为 `46+4+150+726+135+69=1130`；新增套件的实际用例数由脚本打印并写入结果，所有套件零失败才进入 profile。它们共同覆盖 rank 2–5、单轴/多轴/负轴/无序轴、单例维和 `keep_dims`。

- [ ] **Step 2: 增加 S03M 性能层级**

向统一矩阵添加 `s03m_target`：`[8,64,32,128,16]/[1,3]/false`、`[4,64,64,64,16]/[1,3]/true`、`[8,32,32,128,8]/[1,3]/false`、`[16,32,16,128,16]/[3,1]/false`、`[7,64,33,64,16]/[-4,-2]/true`，三元组依次为 shape/axes/keep_dims。`s03m_control` 固定包含 `[8,64,32,127,16]/[1,3]`、`[8,64,32,128,15]/[1,3]`、`[4,64,8,128,16]/[1,3]`，以及统一矩阵现有 `atlas_fast4_rank4_inner64`、`atlas_fast3_rank4_large`、`atlas_fast2_inner8`、`public_case1`。所有 case 保留 `shape/axes/keep_dims/route/tier`，不含评分 Case 编号。

- [ ] **Step 3: 编写单阶段云端门禁**

脚本参数固定为：

```bash
bash diagnostics/run_squaresum_s03m_s03o_gate_20260807.sh \
  s03m \
  /home/ma-user/work/s9/submission_official_names_20260724_1948/projects/SquareSumV1 \
  /home/ma-user/work/s9/runs/s03m_gate_20260807
```

脚本必须：安全拒绝 `/`、仓库根和已存在 work root；断言 `ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME`、`ASCEND_OPP_PATH` 均位于 `/home/ma-user/Ascend/cann-8.5.0`；清空 `artifact/`；构建 baseline/candidate；隔离安装；导入验证库；运行专项正确性；依次安装/执行 baseline-A、candidate、baseline-B 的 target 与 control 三 dtype 矩阵；把唯一 `.run` 复制到 `artifact/SquareSumV1/s03m`、`s03n` 或 `s03o` 对应目录中的 `custom_opp_euleros_aarch64.run`；在最终 JSON 中写入 `stage`、`passed`、`baseline_source`、`candidate_source`、仓库相对 `run_package`、`correctness`、`target_comparison`、`control_comparison` 和 SHA-256；调用比较器：

```bash
python3 diagnostics/compare_squaresum_official_matrices_20260806.py \
  --baseline-a "$baseline_a" --candidate "$candidate" \
  --baseline-b "$baseline_b" \
  --minimum-improvement-percent 50 \
  --maximum-regression-percent 3 \
  --maximum-baseline-drift-percent 3 \
  --output "$comparison"
```

目标和控制分开比较：目标使用最小改善 50%，控制使用最小改善 `-100`、最大回退 3%；两者基线漂移均为 3%。任何子步骤失败时，trap 用临时文件加 `os.replace` 写入 `passed:false`。

- [ ] **Step 4: 在 main 同步后通过 JupyterLab 可见入口运行**

隔离分支提交后，在主工作树执行 `git merge --ff-only perf/squaresum-s03m-s03o-20260807` 和 `git push origin main`。随后：

```powershell
& D:\29722\Desktop\GCC\s9-work\ssh_visible.ps1 -Command "cd /home/ma-user/work/s9/repository/ascend-s9-operators && git pull --ff-only origin main && bash diagnostics/run_squaresum_s03m_s03o_gate_20260807.sh s03m /home/ma-user/work/s9/submission_official_names_20260724_1948/projects/SquareSumV1 /home/ma-user/work/s9/runs/s03m_gate_20260807"
```

远端门禁结束后，创建 `diagnostics/pull_squaresum_remote_artifact_20260807.ps1`，将远端最新 artifact 下载到临时目录，验证 `result.json` 后再替换本地 artifact。传输和清理均在 PowerShell 内完成：

```powershell
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('s03m', 's03n', 's03o')]
    [string]$Stage
)
$repoRoot = [IO.Path]::GetFullPath('D:\29722\Desktop\GCC\s9-work')
$artifactRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'artifact'))
$downloadRoot = [IO.Path]::GetFullPath(
    (Join-Path $repoRoot ".codex_tmp\artifact-$Stage"))
if (-not $artifactRoot.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar)) {
    throw 'artifact path escaped the repository'
}
if (Test-Path -LiteralPath $downloadRoot) {
    Remove-Item -LiteralPath $downloadRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null
& scp.exe -B -P 32375 -i D:\29722\Desktop\GCC\GCC.pem -r `
    'ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com:/home/ma-user/work/s9/repository/ascend-s9-operators/artifact/.' `
    $downloadRoot
if ($LASTEXITCODE -ne 0) { throw 'artifact download failed' }
$gate = Get-Content -LiteralPath (Join-Path $downloadRoot 'result.json') -Raw |
    ConvertFrom-Json
if ($gate.stage -ne $Stage) { throw 'downloaded artifact belongs to another stage' }
Get-ChildItem -LiteralPath $artifactRoot -Force | Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $downloadRoot -Force |
    Move-Item -Destination $artifactRoot -Force
```

保存脚本后运行：

```powershell
& diagnostics\pull_squaresum_remote_artifact_20260807.ps1 -Stage s03m
```

Expected: 全量正确性通过；每个 profile 有 30 个 Mul 和至少 30 个非 Mul 任务；target 每点改善 ≥50%；control 回退 ≤3%；baseline drift ≤3%；`artifact/result.json` 为 `passed:true`。任一条件不满足，停在 S03M，先按 `superpowers:systematic-debugging` 定位，不能开始 S03N。

- [ ] **Step 5: 归档实际证据并提交**

把实际 JSON、日志摘要、SHA-256 和门禁结论写入候选 evidence 与 `SQUARESUM_S03M_CLOUD_GATE_20260807.md`。报告不得把 Event 值称为晋级证据。

```powershell
git add diagnostics candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807/evidence SQUARESUM_S03M_CLOUD_GATE_20260807.md artifact
git commit -m "test: record S03M formal-equivalent gate"
```

---

### Task 4: S03N 任意最内归约长度

**Files:**
- Create: `candidates/squaresum_s03n_arbitrary_grouped_reduce_20260807/ORIGIN.md`
- Create: `candidates/squaresum_s03n_arbitrary_grouped_reduce_20260807/SquareSumV1/`
- Modify: `diagnostics/squaresum_s03m_s03o_static_20260807.py`
- Modify: `diagnostics/squaresum_s03m_s03o_correctness_20260807.py`
- Modify: `diagnostics/squaresum_official_profile_matrix_20260806.json`
- Create after real run: `SQUARESUM_S03N_CLOUD_GATE_20260807.md`

**Interfaces:**
- Consumes: 通过 S03M 门禁的 key 7 路径。
- Produces: `ReduceArbitraryRowsInto(values, accumulate, rows, rowStride, valueCount)`；key 7 接受所有合法 `lastReduceDim > 1`。

- [ ] **Step 1: 克隆已通过的 S03M 并写先失败测试**

复制 S03M 五个比赛文件为 S03N；`ORIGIN.md` 写入 S03M commit、门禁结果 SHA。静态测试新增：S03N 对 `3/7/31/33/63/65/127/129` 选择 key 7，S03M 对同一输入仍回退。运行测试，Expected: 因 Host 仍要求 2 的幂而失败。

- [ ] **Step 2: 实现任意行数 FP32 分块树归约**

新增具体函数：

```cpp
__aicore__ inline void ReduceArbitraryRowsInto(
    AscendC::LocalTensor<float> values,
    AscendC::LocalTensor<float> accumulate,
    const uint32_t rows,
    const uint32_t rowStride,
    const uint32_t valueCount)
{
    uint32_t firstRow = 0U;
    uint32_t remainingRows = rows;
    while (remainingRows > 0U) {
        const uint32_t chunkRows =
            HighestPowerOfTwo(remainingRows);
        AscendC::LocalTensor<float> chunk =
            values[firstRow * rowStride];
        ReduceRowsInPlace(chunk, chunkRows, rowStride);
        AscendC::Add(
            accumulate, accumulate, chunk, valueCount);
        firstRow += chunkRows;
        remainingRows -= chunkRows;
    }
}
```

key 7 的三条 dtype 分支都改为直接调用该函数；`accumulateLocal` 继续在外层归约组之前清零一次。Host 只删除幂次条件，保留 `lastReduceDim > 1`、整除、`SafeMultiply`、8192/1024/32-task 和 inner `%8==0`。

- [ ] **Step 3: 运行静态与设备正确性边界**

Correctness 必含 `1/2/3/7/8/9/31/32/33/63/64/65/127/128/129`；`1` 回退，其他合法长度按容量选择 key 7。额外包含外层归约组大于 1，防止 chunk 累加覆盖而不是累加。

```powershell
python diagnostics\squaresum_s03m_s03o_static_20260807.py --stage s03n --source candidates\squaresum_s03n_arbitrary_grouped_reduce_20260807\SquareSumV1
```

云端运行 `--stage s03n` 专项正确性；Expected: 三 dtype 全部通过且没有越界、超时或设备异常。

S03N 门禁重新运行 Task 3 的前六套既有正确性脚本，再运行新增脚本的 `--stage s03m` 与 `--stage s03n` 两档；不能用 S03M 已保存结果替代本次 S03N 安装上的回归验证。

- [ ] **Step 4: 执行 S03N A/B/A 门禁**

`s03n_target` 固定使用 `[8,128,32,63,16]`、`[8,128,32,65,16]`、`[8,64,32,127,16]`、`[8,64,32,129,16]`，axes 为 `[1,3]`，三 dtype 全部测量。`s03n_control` 固定包含 S03M 的 `[8,64,32,128,16]/[1,3]`，以及 `atlas_fast3_rank4_large`、`atlas_fast2_inner8`、`public_case1` 和任务不足 fast4。基线安装通过门禁的 S03M，门槛仍为 `50/3/3`。若通过，归档真实结果并提交：

```powershell
& D:\29722\Desktop\GCC\s9-work\ssh_visible.ps1 -Command "cd /home/ma-user/work/s9/repository/ascend-s9-operators && git pull --ff-only origin main && bash diagnostics/run_squaresum_s03m_s03o_gate_20260807.sh s03n /home/ma-user/work/s9/submission_official_names_20260724_1948/projects/SquareSumV1 /home/ma-user/work/s9/runs/s03n_gate_20260807"
& diagnostics\pull_squaresum_remote_artifact_20260807.ps1 -Stage s03n
```

```powershell
git add candidates/squaresum_s03n_arbitrary_grouped_reduce_20260807 diagnostics SQUARESUM_S03N_CLOUD_GATE_20260807.md artifact
git commit -m "perf: support arbitrary grouped reductions in S03N"
```

若失败，保留 S03M 为最近通过版本，S03N 不进入 `submission-src`，不开始 S03O。

---

### Task 5: S03O 非 8 对齐 inner 的二维 DMA

**Files:**
- Create: `candidates/squaresum_s03o_unaligned_grouped_rows_20260807/ORIGIN.md`
- Create: `candidates/squaresum_s03o_unaligned_grouped_rows_20260807/SquareSumV1/`
- Modify: `diagnostics/squaresum_s03m_s03o_static_20260807.py`
- Modify: `diagnostics/squaresum_s03m_s03o_correctness_20260807.py`
- Modify: `diagnostics/squaresum_official_profile_matrix_20260806.json`
- Create after real run: `SQUARESUM_S03O_CLOUD_GATE_20260807.md`

**Interfaces:**
- Consumes: 通过 S03N 的任意长度 key 7。
- Produces: key 8 `ProcessStridedGroupedPaddedRows()`；aligned key 7 和所有旧 S02F 路径保持不变。

- [ ] **Step 1: 克隆 S03N 并写先失败的 key 8 测试**

静态测试断言 inner `1/2/7/9/15/17/31/33/63/65/127/129/199` 在容量和任务合法时选择 key 8；inner `8/16/24/32/64` 保持 key 7；`blockCount > 65535`、padded UB 超过 8192、输出超过 1024、任务少于 32 时回退。运行测试，Expected: 因 key 8 尚未注册而失败。

- [ ] **Step 2: Host 计算对齐容量并选择 key 8**

Host 计算：

```cpp
const uint64_t elementsPerBlock = 32U / inputTypeBytes;
const uint64_t alignedInner =
    (innerElements + elementsPerBlock - 1U) /
    elementsPerBlock * elementsPerBlock;
```

key 8 仅在 `innerElements % 8U != 0U` 时考虑，复用 S03N 所有条件，并要求：

```cpp
SafeMultiply(width, lastReduceDim, blockCount) &&
blockCount <= std::numeric_limits<uint16_t>::max() &&
SafeMultiply(blockCount, alignedInner, bufferElements) &&
bufferElements <= 8192U &&
width * innerElements <= 1024U &&
width * alignedInner <= 8192U;
```

调度优先级固定为 key 8 → key 7 → S02F 路径。Host 与 Kernel 共享同样的 8/4/2 宽度选择。

- [ ] **Step 3: Kernel 使用二维 DataCopyPad 和 padded stride**

模板增加 `bool STRIDED_GROUPED_PADDED_ROWS=false`，`Process()` 优先调用 `ProcessStridedGroupedPaddedRows()`。每个任务设置：

```cpp
const uint32_t elementsPerBlock = 32U / sizeof(T);
const uint32_t alignedInner =
    (inner + elementsPerBlock - 1U) /
    elementsPerBlock * elementsPerBlock;
const uint32_t blockCount = activeRows * lastReduceDim;

AscendC::DataCopyExtParams copyParams;
copyParams.blockCount = static_cast<uint16_t>(blockCount);
copyParams.blockLen = inner * sizeof(T);       // bytes
copyParams.srcStride = 0U;                    // bytes
copyParams.dstStride = 0U;                    // 32-byte blocks

AscendC::DataCopyPadExtParams<T> padParams;
padParams.isPad = alignedInner != inner;
padParams.leftPadding = 0U;
padParams.rightPadding = static_cast<uint8_t>(
    alignedInner - inner);                    // elements
padParams.paddingValue = T{};
```

平方和 Cast 的处理长度是 `blockCount * alignedInner`。每个分组输出行从 `row * lastReduceDim * alignedInner` 开始，调用：

```cpp
ReduceArbitraryRowsInto(
    rowValues,
    accumulateLocal[row * alignedInner],
    lastReduceDim,
    alignedInner,
    inner);
```

写回前逐行把真实 `inner` 个结果转换到紧凑的 `outputLocal[row * inner]`，最终一次写回 `activeRows * inner`；padding 不得写回 GM。

- [ ] **Step 4: 注册 key 8 并验证构建边界**

Kernel CMake 固定注册 `1,2,3,4,5,7,8`；入口 key 8 实例化 padded 模板。静态测试检查 Host 可达 key 在 CMake 和入口中一一存在，并检查 `blockCount`、`blockLen`、right padding、UB 乘法均有显式上界。

- [ ] **Step 5: 执行 S03O 正确性与 A/B/A 门禁**

`s03o_target` 固定使用 `[8,64,32,128,15]`、`[8,64,32,127,15]`、`[8,64,32,128,17]`、`[8,32,32,128,31]` 和 `[8,16,32,128,63]`，axes 为 `[1,3]`，三 dtype 全部测量；其余 `7/9/33/65/127/129/199` 在正确性矩阵验证。`s03o_control` 固定包含 S03N 的 `[8,64,32,127,16]/[1,3]`、S03M 的 `[8,64,32,128,16]/[1,3]`、`atlas_fast3_rank4_large`、`atlas_fast2_inner8`、`public_case1` 和 UB 容量回退。基线为通过门禁的 S03N，门槛为 `50/3/3`。

S03O 候选必须重新运行 Task 3 的前六套既有正确性脚本，并依次运行新增脚本的 `--stage s03m`、`--stage s03n`、`--stage s03o`；三档累积零失败后才能执行 A/B/A。

```powershell
& D:\29722\Desktop\GCC\s9-work\ssh_visible.ps1 -Command "cd /home/ma-user/work/s9/repository/ascend-s9-operators && git pull --ff-only origin main && bash diagnostics/run_squaresum_s03m_s03o_gate_20260807.sh s03o /home/ma-user/work/s9/submission_official_names_20260724_1948/projects/SquareSumV1 /home/ma-user/work/s9/runs/s03o_gate_20260807"
& diagnostics\pull_squaresum_remote_artifact_20260807.ps1 -Stage s03o
```

Expected: 三 dtype 正确；target 逐点 ≥50%；controls ≤3%；baseline drift ≤3%。通过后提交实际证据；失败则保留 S03N，不生成 S03O ZIP。

---

### Task 6: 晋升、打包、归档与正式结果闭环

**Files:**
- Modify conditionally: `submission-src/SquareSumV1/`
- Modify: `SQUARESUMV1_ATTEMPT_SUBMISSION_SCORE_ARCHIVE.md`
- Modify: `README.md`
- Modify: `artifact/result.json`
- Modify: `artifact/run.log`
- Create conditionally: `D:/29722/Desktop/GCC/提交相关材料/20260807/S03M/SquareSumV1.zip`、`S03N/SquareSumV1.zip` 或 `S03O/SquareSumV1.zip`

**Interfaces:**
- Consumes: 最近一个 `artifact/result.json` 明确 `passed:true` 的阶段和该阶段独立构建的 `.run`。
- Produces: main 上可追溯的源码、报告和唯一正式 ZIP；平台五 Case 结果回填后决定正式基线。

- [ ] **Step 1: 只晋升通过内部门禁的最高阶段**

先读取 `artifact/result.json`，同时核对 target/control comparison 的 `passed` 与正确性总数。只有全部为真才把对应候选五个比赛文件同步到 `submission-src/SquareSumV1`。同步后执行逐文件 SHA-256 比较，Expected: 五个文件完全一致；其他题目录无变化。

- [ ] **Step 2: 从已验证 RUN 原子生成 ZIP**

```powershell
$gate = Get-Content -LiteralPath artifact\result.json -Raw |
    ConvertFrom-Json
if (-not $gate.passed) { throw 'Latest stage did not pass its gate' }
$stageName = $gate.stage.ToUpperInvariant()
$repoRoot = [IO.Path]::GetFullPath((Get-Location).Path)
$runPath = [IO.Path]::GetFullPath(
    (Join-Path $repoRoot ([string]$gate.run_package)))
if (-not $runPath.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar)) {
    throw 'RUN path escaped the repository'
}
$zipPath = "D:\29722\Desktop\GCC\提交相关材料\20260807\$stageName\SquareSumV1.zip"
python diagnostics\package_operator_submission_20260807.py `
    --source submission-src\SquareSumV1 `
    --run $runPath `
    --output $zipPath
```

执行 ZIP 审计：唯一顶层目录 `SquareSumV1_zip/`；五个源码文件与 `submission-src` 逐字节相同；`.run` 非空且 Unix mode 为 `0755`；记录 Host、Kernel、CMake、RUN、ZIP SHA-256。这里的 RUN 路径和阶段名必须从当次 `artifact/result.json` 读取，不能人工猜写。

- [ ] **Step 3: 更新归档和 README**

归档记录阶段来源、commit、内部门禁、ZIP 路径和 SHA；没有平台成绩时明确写“待正式评测”，不得记为正式最好。README 写明当前正式最好仍是 S02F `3223.995 μs`，以及本轮目标 `1195.599 μs`。

- [ ] **Step 4: 完整验证后提交并推送 main**

按 `superpowers:verification-before-completion` 重新运行静态测试、检查 Git diff、ZIP 结构和 SHA。然后：

```powershell
git add submission-src/SquareSumV1 SQUARESUMV1_ATTEMPT_SUBMISSION_SCORE_ARCHIVE.md README.md artifact
git commit -m "release: package verified SquareSumV1 stage"
git push origin main
git status --short
```

Expected: tracked worktree clean；既有未跟踪实验材料仍保留；GitHub main 包含候选源码、证据、README 和 artifact 最新结果。

- [ ] **Step 5: 回填正式平台五 Case 并判断总目标**

收到平台原始结果后逐 Case 写入归档。五个 Case 全部 Pass 且 `prof_sum <= 1195.599 μs` 才完成总目标；否则把正式较优者设为新正式基线，保留失败方向证据，继续分析未覆盖的通用执行路径，不降低目标。
