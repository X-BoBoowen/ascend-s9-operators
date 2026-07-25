import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import tensorflow as tf
import sys  
import threading
from typing import Optional, Tuple
case_data = {
    'case1': {
        'A_shape': [32, 32],
        'data_type': np.float32,
        'B_shape': [32, 32],
        'out_shape': [32, 32]
    },
    'case2': {
        'A_shape': [128, 256,512],
        'data_type': np.float32,
        'B_shape': [128, 512,256],
        'out_shape': [128, 256,256]
    },
    'case3': {
        'A_shape': [128,128, 63],
        'data_type': np.float32,
        'B_shape': [128,63, 128],
        'out_shape': [128, 128,128]
    },
    'case4': {
        'A_shape': [1,512, 512],
        'data_type': np.float32,
        'B_shape': [1,512, 512],
        'out_shape': [1,512, 512]
    },
    'case5': {
        'A_shape': [2048, 2048],
        'data_type': np.float32,
        'B_shape': [2048, 2048],
        'out_shape': [2048, 2048]
    }
}
def run_with_timeout(func, args=(), kwargs={}, timeout=30):
    result = []
    def target():
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            result.append(e)
            print("函数执行异常:",e)
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None
    if isinstance(result[0], Exception):
        raise result[0]
    return result[0]

def verify_result(real_result, golden):
      # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    if golden.dtype == np.float16:
        loss = 1e-3
    else:
        loss = 1e-4
    
    minimum= 1e-10
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
        if num == '5':
            np.random.seed(42)
        
        caseNmae='case'+num
        tensor_input_real = np.random.uniform(1, 100,case_data[caseNmae]['A_shape']).astype(case_data[caseNmae]['data_type'])
        tensor_input_imag = np.random.uniform(1, 100,case_data[caseNmae]['A_shape']).astype(case_data[caseNmae]['data_type'])

       
        tensor_values_real = np.random.uniform(1, 100,case_data[caseNmae]['B_shape']).astype(case_data[caseNmae]['data_type'])
        tensor_values_imag = np.random.uniform(1, 100,case_data[caseNmae]['B_shape']).astype(case_data[caseNmae]['data_type'])
        
        complex_tensor_A = torch.complex(torch.from_numpy(tensor_input_real), torch.from_numpy(tensor_input_imag))
        complex_tensor_B = torch.complex(torch.from_numpy(tensor_values_real), torch.from_numpy(tensor_values_imag))

        golden = torch.matmul(complex_tensor_A, complex_tensor_B)
        if num == '2':
            tensor_bias_real = np.random.uniform(1, 100,case_data[caseNmae]['out_shape']).astype(case_data[caseNmae]['data_type'])
            tensor_bias_imag = np.random.uniform(1, 100,case_data[caseNmae]['out_shape']).astype(case_data[caseNmae]['data_type'])
            complex_tensor_Bias = torch.complex(torch.from_numpy(tensor_bias_real), torch.from_numpy(tensor_bias_imag))
            golden=golden+complex_tensor_Bias
            tensor_bias_npu = complex_tensor_Bias.npu()
        else:
            tensor_bias_npu = None
        golden=golden.numpy()
        tensor_input_npu = complex_tensor_A.npu()
        tensor_values_npu = complex_tensor_B.npu()
        #tensor_bias_npu = tuple(tensor_bias_npu.tolist())
        if num == '5':
            tensor_input_npu_tmp = complex_tensor_A.npu()
            tensor_values_npu_tmp = complex_tensor_B.npu()
            output = custom_ops_lib.custom_op(tensor_input_npu_tmp, tensor_values_npu_tmp,tensor_bias_npu)
        
        # 修改输入
        output = custom_ops_lib.custom_op(tensor_input_npu, tensor_values_npu,tensor_bias_npu)
        if output is None:
            print(f"{caseNmae} execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, golden):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
