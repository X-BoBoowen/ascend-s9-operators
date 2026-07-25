import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

import torch.nn.functional as F
case_data = {
    # 案例1：float16类型，4D输入（batch, channel, h, w）
    'case1': {
        'grad_output': torch.rand((3, 16, 512, 512), dtype=torch.float16) * 2 - 1,  # 范围(-1,1)
        'input': torch.rand((3, 32, 1024, 1024), dtype=torch.float16, requires_grad=True) * 2 - 1,  # 范围(-1,1)
        'indices': torch.randint(low=-1, high=2, size=(3, 16, 512, 512), dtype=torch.int32),  # 范围(-1,1)
        'kernel_size': torch.tensor([3], dtype=torch.int32),
        'output_size': torch.tensor([16, 512, 512], dtype=torch.int32)
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
        dim = 0
        grad_output = case_data[caseNmae]["grad_output"]
        input_x = case_data[caseNmae]["input"]
        indices = case_data[caseNmae]["indices"]
        
        batch_size=0
        channels=0
        if input_x.dim() == 4:
          batch_size = 1
          channels = input_x.size(0) 
        else:
          batch_size = input_x.size(0)
          channels = input_x.size(1)
        input_dtype = input_x.dtype
        input_device = input_x.device
        
        _random_sample = torch.rand(
        batch_size, channels, 3,  # 第1维为 channels，匹配输入通道数
        dtype=input_dtype,
        device=input_device
        )
        print(_random_sample.size())
        # 直接访问 shape 属性（无需括号）
        print("_random_sample 形状:", _random_sample.shape)  # 输出类似: (2, 3, 3)
        #output = backward_operator_fast(grad_output, x, indices)
        #output=torch.rand((16, 512,512), dtype=torch.float16) * 2 - 1
        
        kernel_size = case_data[caseNmae]["kernel_size"]
        output_size = case_data[caseNmae]["output_size"]
        kernel_size_value=0
        if len(kernel_size.tolist()) == 1:
          kernel_size_value=kernel_size.tolist()[0]
        else:
          kernel_size_value=kernel_size.tolist()
        output, indices_temp = F.fractional_max_pool3d(
                          input_x,
                          kernel_size=kernel_size_value,
                          output_size=output_size.tolist(),
                          return_indices=True,  # 必须返回indices，用于反向传播
                          _random_samples=_random_sample
                          )
        grad_x = torch.autograd.grad(
                outputs=output,       # 正向输出
                inputs=input_x,             # 输入x
                grad_outputs=grad_output  # 上游梯度
                )[0] 
        # 修改输入
        
        kernel_size_npu = kernel_size.npu().tolist()
        output_size_npu = output_size.npu().tolist()
        indices_temp_int32 = indices_temp.to(dtype=torch.int32)
        output_npu = custom_ops_lib.custom_op(grad_output.npu(),input_x.npu(),indices_temp_int32.npu(),_random_sample.npu(),kernel_size_npu, output_size_npu, int(num))
        if output_npu is None:
            print(f"{caseNmae} execution timed out!")
        else:

            if verify_result(output_npu.cpu(), grad_x):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")
        

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
