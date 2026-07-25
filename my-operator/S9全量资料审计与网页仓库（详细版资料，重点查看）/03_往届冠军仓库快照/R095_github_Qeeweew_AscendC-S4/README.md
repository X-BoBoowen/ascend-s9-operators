# 昇腾AI创新算子挑战赛S4赛季 - 复数矩阵乘法实现

## 项目概述
本项目实现了复数矩阵乘法算子，用于昇腾AI处理器。通过将复数矩阵乘法转换为实数矩阵乘法，充分利用了昇腾AI处理器的矩阵计算能
力，实现高性能复数矩阵运算。

## 实现原理
复数矩阵乘法可分解为多个实数矩阵乘法。对于两个复数矩阵A和B：
- A = A_real + i*A_imag
- B = B_real + i*B_imag

结果矩阵C的计算公式为：


C_real = A_real * B_real - A_imag * B_imag C_imag = A_real * B_imag + A_imag * B_real



在实现中，我们：
1. 将输入复数矩阵拆分为实部和虚部
2. 通过3次实数矩阵乘法计算中间结果：
   - P1 = A_real * B_real
   - P2 = A_imag * B_imag
   - P3 = (A_real + A_imag) * (B_real + B_imag)
3. 组合中间结果得到最终输出：
   - C_real = P1 - P2
   - C_imag = P3 - P1 - P2

## 关键优化
1. **内存布局优化**：将复数矩阵的实部和虚部交错存储，便于向量化操作
2. **分块计算**：将大矩阵划分为小块(TILE_M x TILE_N)，适配昇腾AI处理器的本地缓存
3. **并行计算**：利用多核并行处理不同矩阵块
4. **指令级优化**：使用昇腾AI处理器的专用矩阵计算指令

## 文件结构
- `op_host/mat_mul.cpp`：主机端代码，包含Tiling函数实现
- `op_kernel/mat_mul.cpp`：设备端核函数实现
