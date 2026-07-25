import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

case_data = {
    'case1': {
        'x':np.random.uniform(-1, 1, [512, 512]).astype(np.float16),
        'num_groups': 2,
        'num_channels': 512,
        'eps': 1e-5,
        'affine': True
    }
}

def verify_result(real_result, golden):
    if golden.dtype == torch.float32:
        loss = 1e-4  # 容忍偏差，一般fp32要求绝对误差和相对误差均不超过万分之一
    else:
        loss = 1e-3  # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    minimum = 10e-10

    a = real_result - golden  # 计算运算结果和预期结果偏差
    rtol_diff = torch.abs(a)   # 计算运算结果和预期结果偏差绝对值
    golden = torch.where(golden == 0, minimum, golden) # 替换0值为10e-10，防止除零错误
    atol_diff = torch.abs(torch.div(a, golden))  # 计算运算结果和预期结果偏差相对误差
    error_result = (rtol_diff > loss) & (atol_diff > loss)  # 计算运算结果和预期结果偏差是否同时超出误差范围
    err_num = torch.sum(error_result == True) # 相对偏差和绝对偏差均超出预期的元素个数
    if real_result.numel() * loss < err_num:  # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

    
class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseNmae='case'+str(num) 
        input_x = None

        if int(num) == 3:
            input_x = case_data[caseNmae]["x"]

        else:
            input_x = torch.from_numpy(case_data[caseNmae]["x"])
        
        num_groups = case_data[caseNmae]["num_groups"]
        num_channels = case_data[caseNmae]["num_channels"]
        eps = case_data[caseNmae]["eps"]
        affine = case_data[caseNmae]["affine"]
        
        torch_func = torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)
        output = torch_func(input_x)
        # 修改输入
        output_npu = custom_ops_lib.custom_op(input_x.npu(), num_groups, num_channels, eps, affine)
        if output_npu is None:
            print(f"{caseNmae} execution timed out!")
        else:

            if verify_result(output_npu.cpu(), output):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
