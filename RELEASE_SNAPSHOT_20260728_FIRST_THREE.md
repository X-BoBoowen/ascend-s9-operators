# Concat、Greater、IndexAdd 发布与续作快照

时间：2026-07-28 16:35（Asia/Shanghai）

## 发布对象

本快照只固化本轮完成的三题：

- `Concat`
- `Greater`
- `IndexAdd`

`Transpose` 的增强 API 候选尚未修改正式源码；`SquareSumV1` 不在本次
提交范围。

## Git 范围

应提交：

```text
README.md
RELEASE_SNAPSHOT_20260728_FIRST_THREE.md
submission-src/Concat/
submission-src/Greater/
submission-src/IndexAdd/
validation/Concat/
validation/Greater/
validation/IndexAdd/
validation/greater_int32_extended.py
validation/indexadd_stress_benchmark.py
```

不应提交：

```text
candidates/
artifact-inspect-*/
remote-baseline-*/
submission-ready-*/
*.run
*.zip
*.so
*.whl
PROF*/
```

## 最终包

| 题目 | 本地文件 | 云端文件 | ZIP SHA-256 |
| --- | --- | --- | --- |
| Concat | `D:\29722\Desktop\GCC\提交相关材料\Concat.zip` | `/home/ma-user/work/s9/releases/concat_20260728_1446/Concat.zip` | `b43e88f82ba5230f9172b1f1f2f4c07381d4f14cd1fb5a4de3cf3871c55d25b2` |
| Greater | `D:\29722\Desktop\GCC\提交相关材料\Greater.zip` | `/home/ma-user/work/s9/releases/greater_20260728_1511/Greater.zip` | `316797810d06b57d18c898d1fe449c1b0b51565c6a842845dded75524f7f868d` |
| IndexAdd | `D:\29722\Desktop\GCC\提交相关材料\IndexAdd.zip` | `/home/ma-user/work/s9/releases/indexadd_20260728_1622/IndexAdd.zip` | `cf8c08a60d3b356686a07e136766dd06128afa87dccf67d9c9940330843d990b` |

本地和云端 ZIP 已逐文件核对一致。

## 最终实验

```text
/home/ma-user/work/s9/experiments/concat_dynamic_rows_20260728_1417
/home/ma-user/work/s9/experiments/greater_int32_masks_20260728_1439
/home/ma-user/work/s9/experiments/indexadd_hitreuse_20260728_1514
```

## 验证摘要

| 题目 | 定向 | 随机 | 扩展 | 额外说明 |
| --- | ---: | ---: | ---: | --- |
| Concat | 9 | 100 | 168 | 全部逐位相等 |
| Greater | 26 | 220 | 9 | int32 扩展集连续复跑 3 次 |
| IndexAdd | 23 | 170 | 345 | 合计 538；另有大规模压力矩阵 |

## 已拒绝候选

- Greater：高/低 16 位掩码 int32 比较在全位宽随机值中错误，未发布。
- IndexAdd：int8 source 队列批量化与无条件 hit-list 组合在随机测试中
  出错，未发布。
- Transpose：增强 API 方案只有审查结论，没有进入正式源码。

## 下一动作

1. 推送本快照到 GitHub `main`。
2. 在云端建立或更新同一 GitHub 仓库副本，并核对同一 commit。
3. 用户依次上传三个 ZIP。
4. 回填 Case1–Case5、`prof_sum` 和排名。
5. 若五 Case 全 Pass，继续按总耗时排序优化；若失败，先保存原始平台
   输出并定位契约/正确性，不猜隐藏 shape。
6. 平台反馈等待期间继续 `Transpose`。
