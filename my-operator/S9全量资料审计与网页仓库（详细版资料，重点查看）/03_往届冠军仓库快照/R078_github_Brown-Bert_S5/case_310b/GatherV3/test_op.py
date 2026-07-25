import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import tensorflow as tf
import sys  

case_data = {
    'case1': {
        'input': np.random.uniform(-100, 100, [2, 3, 1, 5, 1]).astype(np.float32),
        'indices': np.random.randint(0, 5, size=[2, 3, 3]).astype(np.int32),
        'axis': np.array([3]).astype(np.int32),
        'batch_dims': 2,
        'negative_index_support':False
    }
}

def verify_result(real_result, golden):
      # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    if golden.dtype == np.float16:
        loss = 1e-3
    else:
        loss = 1e-4

    real_result_1d = real_result.reshape(-1)
    golden_1d = golden.reshape(-1)
    for i in range(len(real_result_1d)):
        if real_result_1d[i] != golden_1d[i]:
            print(f'Index {i} not equal: real_result={real_result_1d[i]}, golden={golden_1d[i]}', end="")
    
    minimum = 10e-10
    result = np.abs(real_result - golden)  # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(real_result), np.abs(golden))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, loss)  # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss)  # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss:  # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

    
class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseNmae='case'+str(num)
    
        input_x = torch.from_numpy(case_data[caseNmae]["input"])
        print(input_x)
        input_indices = torch.from_numpy(case_data[caseNmae]["indices"])
        print(input_indices)
        input_axis = torch.from_numpy(case_data[caseNmae]["axis"])
        batch_dims = case_data[caseNmae]["batch_dims"]
        negative_index_support = case_data[caseNmae]["negative_index_support"]
        
        output = tf.gather(input_x.numpy(), input_indices.numpy(), axis=input_axis.numpy()[0],batch_dims=batch_dims)
        # 修改输入
        output_npu = custom_ops_lib.custom_op(input_x.npu(), input_indices.npu(), input_axis.npu(), batch_dims,negative_index_support, input_axis.numpy()[0], int(num))
        if output_npu is None:
            print(f"{caseNmae} execution timed out!")
        else:

            if verify_result(output_npu.cpu().numpy(), output.numpy()):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
