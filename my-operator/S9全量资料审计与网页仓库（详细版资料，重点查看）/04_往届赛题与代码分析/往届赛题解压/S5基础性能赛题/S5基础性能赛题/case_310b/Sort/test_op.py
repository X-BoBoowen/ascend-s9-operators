import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

case_data = {
    'case1': {
        'input_x':np.random.uniform(-100, 100, [128]).astype(np.float16),
        'axis':0,
        'descending':False,
        'stable':False
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

def verify_sort_indices(input_x, output_golden, indices_golden, dim):
    indices_golden = indices_golden.to(torch.int64)
    reconstructed_output = torch.gather(input_x, dim=dim, index=indices_golden)
    is_correct = torch.all(reconstructed_output == output_golden)
    if not is_correct:  # 误差超出预期时返回打印错误，返回对比失败
        print("[ERROR] result error")
        return False
    print("test pass")
    return is_correct


class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseNmae='case'+str(num)
        input_x = None
        input_dim = case_data[caseNmae]["axis"]
        input_descending = case_data[caseNmae]["descending"]
        input_stable = case_data[caseNmae]["stable"]
        if int(num) == 3:
            input_x = case_data[caseNmae]["input_x"]
        else:
            input_x = torch.from_numpy(case_data[caseNmae]["input_x"])
        
        output_golden, indices_golden = torch.sort(input_x, dim=input_dim, descending=input_descending,stable=input_stable)

        # 修改输入
        output, indices  = custom_ops_lib.custom_op(input_x.npu(), input_dim,input_descending, input_stable, int(num))
        if output is None or indices is None:
            print(f"{caseNmae} execution timed out!")
        else:
            output = output.cpu()
            indices = indices.cpu()
            if input_stable:
                if verify_result(output, output_golden) and verify_result(indices, indices_golden):
                    print(f"{caseNmae} verify result pass!")
                else:
                    print(f"{caseNmae} verify result failed!")
            else:
                if verify_result(output, output_golden) and verify_sort_indices(input_x, output, indices, input_dim):
                    print(f"{caseNmae} verify result pass!")
                else:
                    print(f"{caseNmae} verify result failed!")


if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
