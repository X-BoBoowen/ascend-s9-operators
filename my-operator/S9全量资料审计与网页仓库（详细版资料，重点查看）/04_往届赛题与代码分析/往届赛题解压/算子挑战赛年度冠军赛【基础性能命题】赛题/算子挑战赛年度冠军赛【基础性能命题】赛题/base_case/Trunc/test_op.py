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
        'x1':np.random.uniform(-100, 100, [512]).astype(np.float16)
    },

    # 'case{num}': {
    #     'x1':(torch.rand([j, k, l, m],dtype=torch.bfloat16) * scale) - scale / 2
    # },

}

def verify_result(real_result, golden):
    if golden.dtype == torch.float32:
        loss = 1e-4  # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    else:
        loss = 1e-3  # 容忍偏差，一般fp32要求绝对误差和相对误差均不超过百万分之一
    minimum = 10e-10

    result = torch.abs(real_result - golden)  # 计算运算结果和预期结果偏差
    deno = torch.maximum(torch.abs(real_result), torch.abs(golden))  # 获取最大值并组成新数组
    result_atol = torch.less_equal(result, loss)  # 计算绝对误差
    result_rtol = torch.less_equal(result / torch.add(deno, minimum), loss)  # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if torch.sum(result_rtol == False) > real_result.element_size() * loss and torch.sum(result_atol == False) > real_result.element_size() * loss:  # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseNmae='case'+str(num)
        x1 = None
        
        # if int(num) == :
        #     x1 = case_data[caseNmae]["x1"]
        # else:
        #     x1 = torch.from_numpy(case_data[caseNmae]["x1"])
        x1 = torch.from_numpy(case_data[caseNmae]["x1"])
        golden = torch.trunc(x1)
        # 修改输入
        output = custom_ops_lib.custom_op(x1.npu(),int(num))
        if output is None:
            print(f"{caseNmae} execution timed out!")
        else:
            output = output.cpu()
            if verify_result(output, golden):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
