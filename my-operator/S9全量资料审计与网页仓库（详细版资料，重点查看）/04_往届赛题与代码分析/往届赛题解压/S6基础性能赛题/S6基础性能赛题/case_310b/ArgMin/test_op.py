import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

case_data = {
    'case1': {
        'x':np.random.uniform(-1, 1, [1000]).astype(np.float16),
        'dim': 0,
        'keep_dim': False
    }
}

def verify_result(real_result, golden):
    if  golden.dtype == torch.int64:
        loss = 1e-5  # 容忍偏差，一般int要求无误差
    else:
        loss = 1e-3  
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
        
        dim = case_data[caseNmae]["dim"]
        keep_dim = case_data[caseNmae]["keep_dim"]
        
        output = torch.argmin(input_x, dim=dim, keepdim=keep_dim)
        # 修改输入
        output_npu = custom_ops_lib.custom_op(input_x.npu(), dim, keep_dim, output)
        if output_npu is None:
            print(f"{caseNmae} execution timed out!")
        else:

            if verify_result(output_npu.cpu(), output):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
