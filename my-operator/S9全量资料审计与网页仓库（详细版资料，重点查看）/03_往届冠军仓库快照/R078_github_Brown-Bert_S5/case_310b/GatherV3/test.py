import tensorflow as tf
import numpy as np

# 定义三维张量 (2深度, 3行, 4列)
params = np.random.uniform(-100, 100, [2, 3, 1, 5, 1]).astype(np.float32)  # shape=(2, 3, 4)
print(params)

# 二维 indices: 指定从每个深度切片中选择的行
indices = np.random.randint(0, 5, size=[2, 3, 3]).astype(np.int32)  # shape=(2, 2)
print(indices)

result = tf.gather(params, indices, axis=3, batch_dims=2)  # 沿行收集
print("沿行收集结果形状:", result.shape)  # (2, 2, 2, 4)
print("输出:\n", result.numpy())