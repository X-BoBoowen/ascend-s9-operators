import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  
from typing import Optional, Tuple
case_data = {
    'case1': {
        'x':(np.random.uniform(-1, 1, [1024]) * np.random.randint(0, 10, [1024])).astype(np.float16),
        'transpose':False,
        'dtype':3
    }
}

def verify_result(real_result, golden):
      # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    if golden.dtype == np.float16:
        loss = 1e-3
    else:
        loss = 1e-4
    
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
        x = torch.from_numpy(case_data[caseNmae]["x"])
        transpose = case_data[caseNmae]["transpose"]
        dtype = case_data[caseNmae]["dtype"]

        golden = torch.nonzero(x).numpy()
        if transpose:
            golden = golden.transpose()
        if dtype == 3:
            golden = golden.astype(np.int32)
        else:
            golden = golden.astype(np.int64)
        # 修改输入
        output = custom_ops_lib.custom_op(x.npu(), transpose, dtype, golden.shape[0], golden.shape[1], int(num))
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
    
