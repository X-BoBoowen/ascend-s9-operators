# SquareSumV1-S02CQ 本地审计与云端准入方案

日期：2026-08-06

## 当前结论

S02CN 的正式结果为 `4010.864 us`，相对正式最好 S02CA 的
`3982.569 us` 回退 `28.295 us / 0.7105%`，已经淘汰。此前连续微调没有
突破的主要原因不是单个阈值选错，而是测量闭环存在系统性缺口：Event
微基准不等于官方 `msprof` 口径、图谱中含题面维度范围外的数据、且 BF16
没有进入正式性能门禁。

S02CQ 是完成测量复核后的第一个新候选。它目前只通过本地静态与数值模型
审计，**尚未在 910B/CANN 8.5.0 构建或运行，不得提交。**

## 可追溯基线

仓库新增不可修改基线：

```text
baselines/squaresum_s02ca_formal_best_20260806/SquareSumV1
```

它逐文件来自正式提交包 `20260806/S02CA/SquareSumV1.zip`。正式 ZIP、host
和 kernel 的 SHA-256 记录在同目录 `ORIGIN.md`。云端对照不再复用不明来源
的安装状态，而是从该源码重新构建 S02CA。

## S02CQ 唯一改动

S02CA 的 BF16 路径执行：

```text
BF16 input -> FP32 -> FP32 square -> BF16(RINT) -> FP32 -> FP32 reduce
```

S02CQ 删除平方之后的 BF16→FP32 中间量化往返，改为：

```text
BF16 input -> FP32 -> FP32 square -> FP32 reduce -> final BF16(RINT)
```

与 S02CA 比较，host 完全相同，kernel 只有 `160` 行删除、`0` 行新增；
FP16/FP32、tiling、workspace 和最终 BF16 写回均未修改。CANN 8.5 的
Atlas A2 `Mul` 不支持直接 BF16 输入，因此候选仍使用受支持的 BF16→FP32
Cast，不调用未支持接口。

## 已完成的本地验证

- S02CA host SHA-256：
  `0B5C6AEE3B01A63192A6CD4A59CF77A19373B6423274DC73968BAA77D84E9203`
- S02CA kernel SHA-256：
  `C6EA5927D44DDF905E50B309C08E811E9DF614B7018D47F2BEB6A71115EC4C80`
- S02CQ host 与 S02CA 相同；
- S02CQ kernel SHA-256：
  `5E7160EC078FB1DF96F37E8FA3762FD692862D413BF872F12C0BA57E7F8D80FD`
- 静态检查确认 16 处中间 BF16 往返全部删除，13 处最终 BF16 RINT 保留；
- 纯 NumPy BF16-RNE 模型覆盖 reduce 长度
  `1,2,3,7,31,127,513,4096,10000` 和三类数值分布，共 `27/27` 通过官方式
  `1%` 容差/`0.1%` 错误比例规则；最大相对误差约 `0.775%`；
- 23 个性能矩阵 shape 全部通过题面维度范围和 host 路由复算；
- 三个新增 shell 入口均通过语法检查；危险的 `.`/`..` 输出路径会在清理
  artifact 前被拒绝。

数值模型只能证明候选语义具有通过官方容差的合理性，不能替代 NPU 真机
正确性。

## 云端一次性门禁

云端重启后，先找到比赛提供的完整 SquareSumV1 工程模板目录。该模板必须
包含 `build.sh`、`CMakeLists.txt`、`cmake/`、`framework/` 和 `scripts/`。
在仓库根目录执行：

```bash
bash diagnostics/run_squaresum_s02cq_cloud_gate_20260806.sh \
  /absolute/path/to/official/SquareSumV1/project \
  /absolute/path/to/new/s02cq_gate_work_20260806
```

第二个路径必须尚不存在。脚本在一条命令内完成：

1. 本地矩阵、静态和 BF16 数值模型复核；
2. 从同一模板分别构建 S02CA 与 S02CQ；
3. 用两个独立 `--install-path` 安装包，子进程分别加载对应
   `set_env.bash`，避免 OPP 环境串版；
4. S02CQ 对题面域 atlas 的 FP16/BF16/FP32 全路由正确性验证；
5. BF16 core 矩阵按官方任务流执行 S02CA-A → S02CQ → S02CA-B；
6. 基线漂移 `<=3%`、任一用例回退 `<=3%`、总耗时改善 `>=5%` 才通过。

最新结果统一位于：

```text
artifact/result.json
artifact/run.log
artifact/SquareSumV1/candidate_domain_atlas.log
```

每次门禁开始会清理整个 `artifact/`，因此只保留最新一次服务器运行数据。
`result.json` 采用 UTF-8 缩进 JSON 和临时文件替换的原子写入。门禁在构建
或正确性阶段提前失败时也会写入失败阶段与退出码，并保留完整日志，但不会
生成或发布提交包。

## 性能预期与边界

从公开构造方式推断，Case2/Case3 很可能是 BF16；S02CQ 的改动只可能直接
改善这两类 BF16 路径。它不会解释或解决约 `2550 us` 的 Case4，因此即使
S02CQ 真机通过，也不可能单独达到 `0.30 × 3985.330 = 1195.599 us` 的总
目标。它的价值是先验证遗漏的 BF16 融合机会，同时用域内 atlas 获取下一步
定位 Case4 所需的可靠结构证据。该 dtype 判断是公开代码证据支持的推断，
不是隐藏 shape 猜测。

只有 `artifact/result.json` 中 `passed=true`，并且真机正确性日志
完整通过，S02CQ 才能进入正式打包候选；否则立即淘汰，继续从 S02CA 开发。
