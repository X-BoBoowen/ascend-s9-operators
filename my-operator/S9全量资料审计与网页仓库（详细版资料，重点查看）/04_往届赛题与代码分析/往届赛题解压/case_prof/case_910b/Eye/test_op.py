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
import copy

case_data = {
     'case3': {
        'shape': [3, 4, 18, 10],
        'data_type': np.float64,
        'numRows': 18,
        'numColumns': 10,
        'batchShape': [3, 4],
        'dtype': 2,
    },
    'case2': {
        'shape': [8, 8, 512, 512],
        'data_type': np.float32,
        'numRows': 512,
        'numColumns': 512,
        'batchShape': [8, 8],
        'dtype': 0,
    },
    'case1': {
        'shape': [32, 32, 32, 32],
        'data_type': np.float16,
        'numRows': 32,
        'numColumns': 32,
        'batchShape': [32, 32],
        'dtype': 1,
    },
    'case5': {
        'shape': [32, 32, 256, 16],
        'data_type': np.float64,
        'numRows': 256,
        'numColumns': 16,
        'batchShape': [32, 32],
        'dtype': 2,
    },
    'case4': {
        'shape': [16, 128, 512],
        'data_type': np.float32,
        'numRows': 128,
        'numColumns': 512,
        'batchShape': [16],
        'dtype': 0,
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
    loss = 1e-10
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
        caseNmae='case'+num
        tensor_self = np.zeros( case_data[caseNmae]['shape']).astype(case_data[caseNmae]['data_type'])
        
        
        numRows=np.array(case_data[caseNmae]['numRows']).astype(np.int64)
        dtype=np.array(case_data[caseNmae]['dtype']).astype(np.int64)
        numColumns=np.array(case_data[caseNmae]['numColumns']).astype(np.int64)
        batchShape=np.array(case_data[caseNmae]['batchShape']).astype(np.int64)
        
        res = torch.eye(n=case_data[caseNmae]['numRows'],m=case_data[caseNmae]['numColumns'])
        res = torch.broadcast_to(res, case_data[caseNmae]['shape'])
        golden=res.numpy().astype(case_data[caseNmae]['data_type'])

        tensor_self_npu = torch.from_numpy(tensor_self).npu()
        numColumns_npu = torch.from_numpy(numColumns).npu()
        numRows_npu = torch.from_numpy(numRows).npu()
        batchShape_npu=torch.from_numpy(batchShape).npu()
        batchShape_npu = tuple(batchShape_npu.tolist())
        
        dtype_npu=torch.from_numpy(dtype).npu()
        
        # 修改输入
        if num == '5':
            tensor_self_temp = copy.deepcopy(tensor_self)
            tensor_self_npu_temp = torch.from_numpy(tensor_self_temp).npu()
            numColumns_npu_temp = torch.from_numpy(numColumns).npu()
            numRows_npu_temp = torch.from_numpy(numRows).npu()
            batchShape_npu_temp=torch.from_numpy(batchShape).npu()
            batchShape_npu_temp = tuple(batchShape_npu_temp.tolist())
            dtype_npu_temp=torch.from_numpy(dtype).npu()
            output = custom_ops_lib.custom_op(tensor_self_npu_temp, numRows_npu_temp, numColumns_npu_temp,batchShape_npu_temp,dtype_npu_temp)
            
        output = custom_ops_lib.custom_op(tensor_self_npu,  numRows_npu,numColumns_npu,batchShape_npu,dtype_npu)
        if output is None:
            print(f"{caseNmae} execution timed out!")
        else:
            print(type(output))
            output = output.cpu().numpy()
            if verify_result(output, golden):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
