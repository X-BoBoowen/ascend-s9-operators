# 昇腾AI创新算子挑战赛（S4赛季）【基础性能命题】

本项目是“昇腾AI创新算子挑战赛（S4赛季）【基础性能命题】”的开源代码。

## 赛题列表

以下是本次挑战赛的五个赛题：

- [ ] Reshape
- [ ] RmsNorm
- [x] SelectV2
- [x] Pows
- [ ] Gather

## TODO

针对已完成的算子，后续的优化方向包括：

*   **SelectV2 & Pows**:
    *   细分广播与非广播场景的实现，进行针对性优化。
    *   细分各种数据类型的case，写到tilling key中。

---

# Ascend AI Innovation Operator Challenge (S4 Season) [Basic Performance Proposition]

This project contains the open-source code for the "Ascend AI Innovation Operator Challenge (S4 Season) [Basic Performance Proposition]".

## Contest Problems

Below are the five contest problems for this challenge:

- [ ] Reshape
- [ ] RmsNorm
- [x] SelectV2
- [x] Pows
- [ ] Gather

## TODO

Future optimization directions for the completed operators include:

*   **SelectV2 & Pows**:
    *   Refine implementations for broadcasting and non-broadcasting scenarios for targeted optimization.
    *  Subdivide cases for various data types and write them into the tiling key.