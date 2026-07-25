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
        'input_x': np.random.uniform(-5, 5, [32]).astype(np.float16),
        'v': np.random.uniform(-5, 5, [32]).astype(np.float16),
        'dim': 0
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
        input_v = None
        input_dim = case_data[caseNmae]["dim"]
        if int(num) == 3:
            input_x = case_data[caseNmae]["input_x"]
            input_v = case_data[caseNmae]["v"]
        else:
            input_x = torch.from_numpy(case_data[caseNmae]["input_x"])
            input_v = torch.from_numpy(case_data[caseNmae]["v"])
        input_x.requires_grad = True
        need_cast = (input_x.dtype is not torch.float32)
        obj_type = input_x.dtype
        if need_cast:
            input_x = input_x.to(torch.float32)
            input_v = input_v.to(torch.float32)
        glu = lambda x: torch.nn.functional.glu(x, dim=input_dim)
        glu_result = glu(input_x)
        jvp_result_golden = torch.ops.aten.glu_jvp(glu_result, input_x, input_v, input_dim)
        if need_cast:
            input_x = input_x.to(obj_type)
            input_v = input_v.to(obj_type)
            jvp_result_golden = jvp_result_golden.to(obj_type)
            glu_result = glu_result.to(obj_type)
        # 修改输入
        jvp_result = custom_ops_lib.custom_op(glu_result.npu(), input_x.npu(), input_v.npu(), input_dim, int(num))
        if  jvp_result is None:
            print(f"{caseNmae} execution timed out!")
        else:
            jvp_result = jvp_result.cpu()
            if verify_result(jvp_result, jvp_result_golden):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
