# S9 五算子逐题独立评分优化方案

> 起草日期：2026-07-30
> 适用环境：Ascend 910B、CANN 社区版 8.5.0
> 当前正式源码：`s9-work/submission-src/<Op>/`
> 历史事实来源：`check/20260728优化迭代跟踪.md`
> 本方案用途：确定下一轮可执行的单变量实验、正确性门禁、性能判据和打包流程

---

## 1. 评分口径

五道题分别使用各自五个 Case 的 `prof_sum` 排名，再把题内名次换算为积分：

| 题内名次 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 10 名以后 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 积分 | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |

必须严格区分：

1. 五个 Case 的耗时只在同一道题内部求和。
2. 不同算子的 `prof_sum` 不相加形成赛事成绩。
3. 赛季总分相加的是五道题各自的名次积分。
4. 每题第 1～3 名及第 4～10 名的单项奖金独立评定。
5. 跨题投入顺序应比较“预计能跨越多少题内名次、增加多少积分”，不能比较不同算子的绝对耗时。

当前已取得五道题第 10 名的 2026-07-30 门槛。五题当前成绩均高于对应门槛，因此都在第 10 名以后，当前积分均为 0。现阶段先以“至少一道题进入前十、获得首个 10 分和单项优秀奖区间”为目标。

---

## 2. 当前正式平台结果

本轮五个正式 ZIP 均为 `5/5 Pass`：

| 题目 | 当前版本 | Case1 | Case2 | Case3 | Case4 | Case5 | 题内 `prof_sum` | 相对本题 B0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Concat | Concat-C01 同包最低观测 | 14.13 | 38.9005 | 19.07 | 105.102 | 511.44 | 688.6425 | 40.14% |
| Greater | Greater-G01 | 5.24 | 16026.332 | 2262.108 | 92.344 | 209.348 | 18595.372 | 55.76% |
| IndexAdd | IndexAdd-I01 | 9.03 | 37491.429 | 88.1215 | 2568.781 | 79270.514 | 119427.8755 | 3.83% |
| Transpose | Transpose-T02 | 4.62 | 51.161 | 2467.94 | 2195.774 | 11583.732 | 16303.227 | 58.78% |
| SquareSumV1 | SquareSumV1-S02E2 | 6.3505 | 400.158 | 239.505 | 1718.344 | 895.168 | 3259.5255 | 71.28% |

Concat 的 `688.6425` 与此前 `693.2035` 来自完全相同的 ZIP，差值只有 `0.66%`。它属于同包平台波动，不是新版本收益。

---

## 3. 跨题投入决策模型

### 3.1 榜单数据表

取得五张实时榜单后填写：

| 题目 | 当前名次 | 当前积分 | 当前 `prof_sum` | 第10名 | 第3名 | 第1名 | 最近可跨档目标 | 预计新增积分 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Concat | 10 名以后 | 0 | 688.6425 | 225.096 | 待填 | 待填 | 第10名 | +10 |
| Greater | 10 名以后 | 0 | 18595.372 | 572.743 | 待填 | 待填 | 第10名 | +10 |
| IndexAdd | 10 名以后 | 0 | 119427.8755 | 841.584 | 待填 | 待填 | 第10名 | +10 |
| Transpose | 10 名以后 | 0 | 16303.227 | 1063.036 | 待填 | 待填 | 第10名 | +10 |
| SquareSumV1 | 10 名以后 | 0 | 3259.5255 | 1268.598 | 待填 | 待填 | 第10名 | +10 |

### 3.2 第 10 名差距

| 题目 | 当前值 | 第10名 | 至少需下降 | 至少需下降比例 | 当前/门槛 |
|---|---:|---:|---:|---:|---:|
| SquareSumV1 | 3259.5255 | 1268.598 | 1990.9275 | 61.08% | 2.57× |
| Concat | 688.6425 | 225.096 | 463.5465 | 67.31% | 3.06× |
| Transpose | 16303.227 | 1063.036 | 15240.191 | 93.48% | 15.34× |
| Greater | 18595.372 | 572.743 | 18022.629 | 96.92% | 32.47× |
| IndexAdd | 119427.8755 | 841.584 | 118586.2915 | 99.30% | 141.91× |

这张表只表示距离，不等于工程难度。IndexAdd 虽然差距最大，但存在数量级复杂度问题，理论收益上限也可能最大；因此远距离题仍需做低成本结构诊断，不能仅凭倍数永久放弃。

### 3.3 慢 Case 的硬预算

若暂时假设非重点 Case 保持当前耗时，进入第 10 名要求：

| 题目 | 必须共同优化的 Case | 当前合计 | 可用预算 | 所需加速 |
|---|---|---:|---:|---:|
| SquareSumV1 | Case4 + Case5 | 2613.512 | 622.5845 | 4.20× |
| Concat | Case5 | 511.44 | 47.8935 | 10.68× |
| Transpose | Case3 + Case4 + Case5 | 16247.446 | 1007.255 | 16.13× |
| Greater | Case2 + Case3 | 18288.44 | 265.811 | 68.80× |
| IndexAdd | Case2 + Case4 + Case5 | 119330.7245 | 744.4325 | 160.29× |

结论：

- SquareSumV1 不能只修安全或只优化 Case4；Case4、Case5 必须共同压到约 `615` 以下。
- Concat 几乎必须消除 Case5 的整个当前慢路径。
- Transpose 必须同时重做 Case3/4/5，继续只优化 Case5 无法进入前十。
- Greater 即使 Case2 变为 0，其他 Case 仍超过门槛；Case2、Case3 必须共同重构。
- IndexAdd 即使 Case2、Case5 变为 0，Case4 等残余仍超过门槛；需要完整算法级重构。

### 3.4 量化优先值

对每个候选实验计算：

```text
预计优先值
= 预计新增积分 × 命中置信度 × 正确性成功率
  /（预计工程小时 × 平台评测周转成本）
```

同时维护一个独立的奖金目标：

```text
单项奖优先值
= 跨入第10/第3/第1名的概率 × 对应奖金额
  / 预计工程成本
```

规则：

- 已证实的越界、竞态和错误结果不参与普通排序，必须先修。
- 同一题内优先处理占该题 `prof_sum` 高、且源码证据充分的路径。
- 两个候选预计积分相同时，优先低风险、易回退、一次实验可归因的版本。
- 没有实时榜单时，不得声称某道题“对总分影响小”。

---

## 4. 证据分级

| 等级 | 含义 | 可支持的结论 |
|---|---|---|
| A | 当前正式源码直接确认 | 可以实施针对该结构的单变量修改 |
| B | 正式平台前后结果支持 | 可以判断某次修改影响了部分隐藏 Case，但不能反推出隐藏 shape |
| C | 真机日志、Event 或 Profiler 确认 | 可以判断瓶颈单元、核利用率和流水状态 |
| D | 由耗时分布或代码结构做出的高置信推断 | 只能先做诊断或小范围 A/B |
| E | 尚待验证 | 不允许直接大改或描述为事实 |

隐藏 Case 的 shape、dtype 和精确 TilingKey 不得猜测或硬编码。

---

## 5. 当前决策表

| 题目/路径 | 主要问题 | 证据 | 下一单变量实验 | 主要风险 | 验收条件 |
|---|---|---|---|---|---|
| SquareSumV1 atomic | 对齐到 8 个 float 后直接写真实输出，1～7 输出可能越界 | A | S02A：atomic 目标改为对齐 workspace，最终按真实长度写回 | 同步或 final copy 回退 | guard/canary 完整；全部回归 Pass |
| Concat flat/row 分核 | useful task 与 blockDim 可能不匹配，产生空核或小任务启动开销 | A，Case 映射为 D | C02A：只修改分核成本模型 | 小 Case 固定开销上升 | Case5 明显下降；其余题内 Case 不异常回退 |
| Greater int32/general broadcast | int32 中间链长；一般广播仍有逐元素寻址/GM 访问 | A，Case2 映射为 B/D | G02A 诊断后选择 G02B 或 G02C | bool 语义、广播尾块错误 | Case2 目标路径 microbench 至少下降 10% |
| IndexAdd hit-list | hit-list 随 chunkGroup/outer 重复扫描；现有复用与并行度存在取舍 | A，Case2/5 映射为 B/D | I02A 统计扫描次数，再做 I02B owner A/B | 降低扫描但严重损失核并行 | 扫描次数实降且 Case2/5 至少一项稳定下降 |
| Transpose Key3/其他 Key | Key3 部分标量路径已消除，但无真实流水；Case3/4 路径未知 | A/B/E | T03A 通用路径矩阵，再选择单 Key 实验 | 修改未命中目标 Case | 本地对应路径明确下降；平台结果可归因 |
| SquareSumV1 fastPath3 | 逐输出二维 DMA、归约和标量同步；S02E2 平台只改善 0.70% | A/B/C | S02F：连续 8 输出批处理 | 指令掩码上限、长尾错误 | 966 例 Pass；命中形状 A/B 至少 10%；不命中路径无回退 |

---

## 6. SquareSumV1 路线

### 6.1 S02A：atomic 安全 workspace

这是当前唯一不依赖榜单即可立即实施的修改。

唯一假设：

> atomic 仍保留多核归约结构，但不再写真实 output 的对齐尾部；改写到足够大的用户 workspace 后，仅由 core 0 按真实输出字节数回写。

Host：

1. `reduceMode == 1` 也申请用户 workspace。
2. `partialStride = AlignUp(outputElements, 8)`。
3. atomic 模式用户区为 `partialStride * sizeof(float)`。
4. workspace reduce 模式仍为 `partialStride * blockDim * sizeof(float)`。
5. 加上系统 workspace，并检查乘法、加法和 `size_t` 转换溢出。

Kernel：

1. `reduceMode == 1 || reduceMode == 2` 均使用 `GetUserWorkspace`。
2. atomic 模式把 `workspaceGm_` 绑定为 `partialStride` 个 float。
3. core 0 清零对齐 workspace，第一次 `SyncAll`。
4. 各核只写真实输出 lane，padding lane 保持 0。
5. atomic add 的目标改为 workspace。
6. 等待 atomic MTE3 完成并恢复 atomic none，第二次 `SyncAll`。
7. core 0 从 workspace 取最终值，使用真实 `outputElements * sizeof(float)` 的 `DataCopyPad` 写 output。

专项矩阵：

```text
outputElements = 1..9
reduceElements = 2047, 2048, 2049, 8191, 8192, 8193
blockDim = 8, 16, 32, 40
output 前后 guard = 64B
同输入重复 = 100 次
```

决定：

- 任一 guard 被改写：撤回实现。
- 数值错误或不稳定：撤回实现。
- 安全正确但性能回退：保留安全语义，另开 S02A1 优化同步/final copy；不恢复越界写法。

### 6.2 S02B 以后

S02A 通过后根据通用路径 microbenchmark 选择一个：

| 编号 | 唯一修改 | 适用证据 |
|---|---|---|
| S02B | 每核多个 partial 合并为一次对齐写 | MTE3 指令数过高 |
| S02C | finalizer 以批量 DMA + UB 树归约替代逐核小搬运 | Profiler 确认 finalizer 占比高 |
| S02D | 只 A/B blockDim 成本模型 | 核同步或尾核成本明显 |
| S02E | strided-inner 顺序 Add 改树归约 | 本地 fastPath4 明显慢 |
| S02F | grouped-suffix 批量计算多个输出 | 本地 fastPath3 明显慢 |
| S02G | ReduceContiguous 三种实现 microbenchmark | 连续长归约仍为主成本 |
| S02H | 按路径拆分 TilingKey/UB | 不同路径 UB 需求冲突 |
| S02I | 真正 CopyIn/Compute/CopyOut 流水 | 时间轴确认当前串行 |

---

## 7. Concat 路线

当前题内瓶颈：Case5 占 `74.27%`。第 10 名门槛为 `225.096`；若其他 Case 不变，Case5 必须低于 `47.8935`。

### C02A：active-core 分核模型

唯一修改：Host 的 blockDim 只由可执行 task 数和字节成本决定，Kernel 搬运算法保持不变。

需要记录：

```text
outer
inputCount
row tasks
flat tasks
total bytes
usefulTaskCount
blockDim
active core count
每核 min/max bytes
```

成功线分两级：

- 本地大 flat/row 矩阵 active core 与 useful task 匹配。
- 第一轮归因线：平台 Case5 至少下降约 10%，低于约 `460`。
- 进入前十硬目标：题内 `prof_sum < 225.096`，在其他 Case 不变时 Case5 `< 47.8935`。
- Case1/2 的固定开销不出现数量级回退。

### C02B～C02E

按证据依次选择，不叠加：

1. C02B：消除 `CopyFlat` 对 input descriptor 的重复前缀扫描。
2. C02C：32B 对齐中段走普通 `DataCopy`，只有头尾使用 `DataCopyPad`。
3. C02D：在时间轴证明串行后引入双缓冲。
4. C02E：最后才调整 tile bytes 和每核目标字节。

Concat 的最终门槛必须由 Concat 第 10、第 3、第 1 名目标线重设。

---

## 8. Greater 路线

当前题内瓶颈：

- Case2：`16026.332`，占 `86.18%`。
- Case3：`2262.108`，占 `12.16%`。

### G02A：通用路径诊断

不能尝试提取隐藏 shape。应在本地构建覆盖所有合法广播模式和 dtype 的矩阵，记录：

```text
dtype
broadcastMode
partitionUnitElements
runLength / totalRuns
blockDim
tileElements
scalar/vector element count
MTE2 / Vector / MTE3 时间
```

### G02B：int32 packed mask

仅当本地和平台响应支持 Case2 命中 int32 路径时实施。

唯一修改：减少当前 Max/Equal/Sub/Cast 等中间链和完整 half Tensor 展开；广播寻址、tile 和 blockDim 不变。

风险：

- 有符号 int32 极值；
- 相等、大小关系和 bool 输出必须逐元素一致；
- 尾块 padding 不得污染输出；
- 不允许用浮点转换代替精确 int32 比较。

### G02C～G02F

1. scalar broadcast：标量只读一次，整 tile 复用。
2. constant suffix：周期模板一次构造，多 repeat 复用。
3. periodic suffix：按周期批量搬入/展开，不逐元素更新坐标。
4. 大 run 拆成 `(run, chunk)`，避免少量 run 限制核数。
5. 最后按时间轴证据做双缓冲。

工程成功线：

- 单变量归因线：目标路径至少下降 10%。
- 进入前十硬目标：题内 `prof_sum < 572.743`。
- 当前非 Case2 的 Case 合计 `2569.04`，已超过门槛；因此 G02 不能只优化 Case2。
- 若 Case1/4/5 暂时不变，Case2 + Case3 必须低于 `265.811`。

---

## 9. IndexAdd 路线

当前题内瓶颈：Case2 + Case5 占本题 `97.77%`。

### 9.1 对专家建议的修正

`BuildHitList(groupStart, groupCount)` 的逻辑结果不随 `outer` 和 inner chunk 改变，这是源码事实。

但 hit-list 当前保存在单核 UB 中：

- 不同 `(outer, dimGroup, chunkGroup)` task 不能直接共享 UB；
- 把 owner 简单改成 `dimGroup` 会减少扫描，但也可能把 40 核工作压到少数核；
- 若要达到 `G × M + O × updates`，必须有 workspace 中的共享压缩表、可靠的阶段同步，或分离预处理；
- 因此不能只修改 task 编码就宣称跨 outer 复用完成。

### I02A：扫描/并行度诊断

对通用本地矩阵记录：

```text
outer / dimSize / inner / indexCount
dimGroups / chunkGroups
useHitReuse
BuildHitList 调用次数
总 index 扫描元素数
有效 update 数
active cores
每核 chunk 数
MTE2 / scalar / vector / MTE3 时间
```

### I02B-owner：无 workspace 的 owner A/B

唯一修改：一个 `(outer, dimGroup)` owner 构建一次 hit-list，并消费该 outer 下的所有 inner chunks。

只在以下任一条件满足时启用：

1. `outer × dimGroups` 已足以占满目标核数；或
2. 诊断证明重复扫描成本远大于并行度损失。

必须与当前 chunkGroup 方案做同 shape A/B。若扫描下降但总时间上升，立即回退，不继续调 tile。

### I02C-workspace：跨 outer 共享 hit-list

只有 I02B 证明扫描是主成本、但并行度不足时才进入。

方案：

1. 为每个 dim group 统计 hit count。
2. 生成 prefix offset。
3. 把 `(indexPosition, rowInGroup)` 压缩到 workspace。
4. 所有 outer/chunk task 只消费压缩表。

这是高风险结构版本，必须单独验证：

- workspace 大小溢出；
- 阶段间全核同步；
- 重复 index 的稳定累加语义；
- 多核写同一 output 的数据竞争；
- 稀疏场景压缩开销是否超过收益。

### I02D/I02E

- I02D：仅对对齐、支持 dtype、密集更新启用 atomic 快路径。
- I02E：atomic 不稳定或 dtype 不支持时使用稳定 bucket workspace。
- 数量级问题解决前，不做双缓冲或 tile 微调。

工程成功线：

- I02B 若 Case2/5 都不足 10% 改善，停止微调并重新判断主成本。
- 结构路线第一目标：Case2、Case5 各下降 20%。
- 进入前十硬目标：题内 `prof_sum < 841.584`。
- Case1 + Case3 当前为 `97.1515`；若它们不变，Case2 + Case4 + Case5 必须低于 `744.4325`。
- 这要求主要三 Case 合计加速约 `160.29×`，必须以算法级复杂度改变为目标，参数微调无意义。

---

## 10. Transpose 路线

当前题内瓶颈：

- Case5：`11583.732`，占 `71.05%`。
- Case3 + Case4：`4663.714`，占 `28.61%`。

### 10.1 对隐藏 Case 诊断的边界

不得要求或尝试提取隐藏 Case 的 shape/dtype。

可用方法：

1. 本地覆盖所有 rank、dtype、合法 permutation 和边界。
2. 为每种 TilingKey 建立独立 microbenchmark。
3. 一次只修改一个 TilingKey。
4. 用平台哪个 Case 响应来做归因，但不反推出或硬编码隐藏 shape。

### T03A：通用路径矩阵

记录：

```text
dtype / rank / shape / permutation
normalized permutation
TilingKey / blockDim
rows / cols / totalTiles / tilesPerCore
contiguousElements / totalRuns / chunksPerRun
gather stride / tail tiles
MTE2 / Vector / MTE3 / scalar 时间
```

### 后续单变量分支

| 本地确认路径 | 下一实验 |
|---|---|
| Key3 float/int32 | T03B 真双缓冲；通过后再测 supertile |
| Key3 int8 | T03B 真双缓冲；随后独立比较 pair-gather、half 中转、批量 gather |
| Key1 | 把完整 run 拆成 `(run, chunk)` 提高并行度 |
| Key4 int8 | uint16 GatherMask 取槽首元素，再向量压回 int8 |
| Key2/5 | 修正伪流水，避免每次 EnQue 后立即 DeQue |
| adjacent swap | 新增通用 Key6，不写死 shape |

工程门槛：

```text
Case5 第一目标 < 8109
Case3 < 1974
Case4 < 1757
题内 prof_sum 第一目标 < 12000
进入第10名硬目标 < 1063.036
```

现有 Case1 + Case2 为 `55.781`；若保持不变，Case3 + Case4 + Case5 必须低于 `1007.255`，相当于三项合计加速约 `16.13×`。

---

## 11. 正确性门禁

### 11.1 公共矩阵

| 类别 | 必测内容 |
|---|---|
| 空/退化 | 0 元素、extent 0/1、标量、单行、单列 |
| 对齐边界 | 7/8/9、15/16/17、31/32/33、63/64/65 |
| tile 边界 | tile-1、tile、tile+1、2×tile±1 |
| 核间边界 | 工作量不能整除 8/16/32/40 核 |
| dtype | 题目声明支持的全部 dtype |
| 特殊值 | `+0/-0`、Inf、NaN、subnormal、整数极值 |
| 地址安全 | 输入、输出、workspace 前后 guard/canary |
| 稳定性 | 相同输入重复 100 次；atomic 路径重点检查 |
| 大地址 | 64 位 offset、乘法和 workspace size 溢出 |

### 11.2 题目专项

- Concat：空 input、负 dim、非对齐拼接、多个空 input、所有输入均空。
- Greater：完整广播规则、标量广播、int32 极值、NaN 比较语义、尾块。
- IndexAdd：重复 index、负/越界 index 契约、非对齐 inner、同一 row 多次命中。
- Transpose：所有 permutation、extent 0/1、四种 dtype 逐 bit、int8 256 种 bit pattern。
- SquareSumV1：负 dims、重复 dims 契约、bf16/fp16 累加语义、atomic/workspace guard。

任何正确性失败、越界、竞态或未定义行为都直接否决版本，不进入性能比较。

---

## 12. 性能测量规则

```text
预热：2 次
正式测量：至少 3 次
记录：median / min / max
离散度 > 2%：扩展为 7 次
环境：同设备、同 CANN、同 .run
变量：一次只改一个主要假设
```

必须分开记录：

- 官方平台 Case Result；
- 官方平台 `prof_sum`；
- 本地 Event；
- msProf task time；
- runner 总时间。

小于约 2% 的平台变化先按波动处理；同包复测不能描述成代码收益。

---

## 13. 版本、打包和回退

### 13.1 编号

```text
Concat-C02A...
Greater-G02A...
IndexAdd-I02A...
Transpose-T03A...
SquareSumV1-S02A...
```

诊断版、失败版和回退版都保留记录，不覆盖 B0 或上一最佳版本。

### 13.2 正式包清单

```text
[ ] diff 只包含目标修改
[ ] Host/Tiling/Kernel 编译成功
[ ] 全部随机、边界和专项回归通过
[ ] guard/canary 未改变
[ ] 本地 A/B 达到预期，或属于必须提交的安全修复
[ ] .run 确由当前源码构建
[ ] ZIP 内源码与构建源码完全一致
[ ] ZIP 层级和 op name 符合题目要求
[ ] 记录 source / .run / ZIP SHA-256
[ ] 平台 5/5 Pass
[ ] 更新主跟踪文档
```

### 13.3 回退原则

- Incorrect、guard 失败、数据竞争：立即撤回。
- 安全修复正确但性能回退：保留安全语义，另开性能子版本。
- 目标 Case 改善而另一大 Case 回退：不直接合并，收紧 Host 条件或拆分 TilingKey。
- 所有变化小于 2%：重复测量，不宣称收益。
- 修改未命中预期路径：结果不能证明假设，回到诊断。

---

## 14. 当前冻结的执行顺序

根据第 10 名门槛，冻结以下步骤：

1. 保持五个当前正式 ZIP 和源码为可回退基线。
2. 第一争分题为 SquareSumV1：先完成 S02A 安全修复，再连续攻 Case4/5，目标 `prof_sum < 1268.598`。
3. 第二争分题为 Concat：Case5 必须进入约 `48` 的量级，目标 `prof_sum < 225.096`。
4. 第三题暂定 Transpose：必须按通用路径覆盖 Case3/4/5，目标 `< 1063.036`。
5. Greater 与 IndexAdd 先各做一次低成本结构诊断；根据实测复杂度收益决定第四、第五顺序。
6. 一次只执行一个算子的一个 A/B 编号，不把两个题的修改混入同一评测包。
7. 每次平台结果归档后重新计算距离第 10 名的题内预算。

第 3、第 1 名门槛可在进入前十后再补；它们不阻塞当前获得首个积分的目标。

---

## 15. 开始实施前需要的外部信息

第 10 名门槛已经齐全，足以制定第一阶段争分顺序。第 3 名和第 1 名门槛可在任一题进入前十后再提供，用于第二阶段单项奖目标。

云端开始实施时还需要：

1. ModelArts Notebook 已启动。
2. SSH 远程开发可连接。
3. CANN 8.5.0 环境仍可用。
4. 用户确认可以消耗本轮云端额度。

在这些条件满足前，可以继续完善本地测试和候选 diff，但不得声称已经编译、真机验证或获得平台收益。

---

## 16. 2026-07-30 实施进展：SquareSumV1-S02E2

已完成并晋级：

1. S02A：atomic 对齐 workspace 安全修复。
2. S02C/S02H：大 workspace finalizer 使用 2D DMA + UB 树归约，并以
   TilingKey 1/2/3/4 隔离顺序与树形 finalizer。
3. S02E2：fastPath4 一次搬入真实归约行，UB 补零到下一 2 次幂后做
   二叉树归约。

结果：

- 全量正确性 `950/950 Pass`；
- fastPath4 大矩形覆盖集三 dtype 稳定改善约 `35%～38%`；
- 24/48/96 行 fp16 扫点分别由 `62.035/98.626/178.216 μs` 降至
  `46.302/65.879/115.766 μs`；
- 非目标控制路径未观察到稳定回退；
- 正式 ZIP SHA-256：
  `14B3384ED0A5A740808E008BE0D7922BEA9D5A27CD837CE8CFCC0DBD3D196EE9`。

官方结果：

```text
Case1 6.3505
Case2 400.158
Case3 239.505
Case4 1718.344
Case5 895.168
prof_sum 3259.5255
```

相比 S01 的 3282.536 只改善 23.0105 μs（0.70%），其中 Case4 只改善
1.844 μs（0.11%）。结论是 fastPath4 没有命中主要隐藏瓶颈；该结论不
允许用于反推隐藏 shape。

---

## 17. 2026-07-30 实施进展：SquareSumV1-S02F

唯一主要修改：为满足严格安全条件的 fastPath3 新增 TilingKey 5，一次
处理 8 个最内层连续输出，合并二维 DMA、向量平方、尾部归约和输出写回。

安全门控包括：

- fastPath3 且不使用 atomic/workspace 模式；
- 尾部归约按输入 dtype 的 32B block 对齐；
- 尾长不超过 64；
- 最内层保留输出维连续、长度可被 8 整除；
- 8 个输出的输入片段不超过 UB，二维搬运 stride 不溢出。

开发中曾发现尾长 1024 错误。CANN 8.5.0 源码证明 vector repeat 为
256B，`WholeReduceSum<float>` 一次最多覆盖 64 个 float；因此门控上限
收紧为 64，长尾回退 S02E2。该错误未进入正式源码或提交包。

正确性结果：

```text
46 directed
4 bf16 semantic
4 atomic
10 workspace
10 tiling-key
150 random
726 extended
16 grouped-vector8 专项
合计 966/966 Pass
```

S02E2/S02F 相邻时间窗口 A/B：

| 场景 | S02E2 | S02F | 改善 |
|---|---:|---:|---:|
| small aligned fp16 | 43.614 | 27.567 | 36.79% |
| medium aligned fp16 | 92.608 | 68.257 | 26.30% |
| large aligned fp16 | 98.771 | 77.377 | 21.66% |
| wide output fp16 | 187.409 | 125.721 | 32.92% |
| medium aligned bf16 | 100.449 | 78.185 | 22.16% |
| medium aligned fp32 | 88.014 | 67.717 | 23.06% |
| 未命中尾长 33 fp16 | 101.707 | 101.552 | 0.15% 波动 |

命中形状合计改善 27.18%，未命中路径无稳定回退。正式源码已在独立
release 目录重建，包内源码逐文件与 `submission-src/SquareSumV1`
一致。

```text
云端 release:
/home/ma-user/work/s9/release/squaresum_s02f_20260730_1629

.run SHA-256:
AAA8096582170BAB688607E40651E8C1D8B9B86E7F889F61948C8D1CE7320563

ZIP SHA-256:
EECD9F1FD6B4C0617B6EC2EC632F24BB0F310D5A9D2F2125F03F6EC86ECFAF5B
```

当前状态：S02F 等待官方平台评测。平台结果返回前不更新最低
`prof_sum`，不声称已进入前十。下一候选为 S02G（`ReduceContiguous`
分层归约 microbenchmark）或基于新平台结果选择的其他通用路径；继续
禁止猜测或硬编码隐藏 shape。
