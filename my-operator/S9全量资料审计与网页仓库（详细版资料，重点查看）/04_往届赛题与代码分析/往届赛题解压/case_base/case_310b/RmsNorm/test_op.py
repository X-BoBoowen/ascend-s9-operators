import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import tensorflow as tf
import sys  
import threading



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
    if golden.dtype == torch.float32:
        loss = 1e-4  # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    elif golden.dtype == torch.bfloat16:
        loss = 5e-3
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
    def test_custom_op_case1(self):
        input_x = np.random.uniform(1, 10, [2,512]).astype(np.float16)
        input_gamma = np.random.uniform(0,1, [512]).astype(np.float16)
        epsilon = 1e-6
        x = torch.from_numpy(input_x)
        gamma = torch.from_numpy(input_gamma)
        mean = torch.mean(x ** 2, dim=1, keepdim=True)
        rstd = torch.rsqrt(mean + epsilon)
        golden = gamma * (x * rstd)

        x_npu = x.npu()
        gamma_npu = gamma.npu()

        # 修改输入
        output = custom_ops_lib.custom_op(x_npu, gamma_npu, epsilon)
        if output is None:
            print("case1 execution timed out!")
        else:
            output_y = output[0].cpu()
            rstd_output = output[1].cpu()
            if verify_result(output_y, golden) and verify_result(rstd_output, rstd):
                print("case1 verify result pass!")
            else:
                print("case1 verify result failed!")

    def test_custom_op_case2(self):

        input_x = np.random.uniform(1, 10, [8,512,512]).astype(np.float32)
        input_gamma = np.random.uniform(0,1, [512,512]).astype(np.float32)
        epsilon = 1e-6
        x = torch.from_numpy(input_x)
        gamma = torch.from_numpy(input_gamma)
        mean = torch.mean(x ** 2, dim=(1,2), keepdim=True)
        rstd = torch.rsqrt(mean + epsilon)
        golden = gamma * (x * rstd)

        x_npu = x.npu()
        gamma_npu = gamma.npu()

        # 修改输入
        output = custom_ops_lib.custom_op(x_npu, gamma_npu, epsilon)
        if output is None:
            print("case2 execution timed out!")
        else:
            output_y = output[0].cpu()
            rstd_output = output[1].cpu()
            if verify_result(output_y, golden) and verify_result(rstd_output, rstd):
                print("case2 verify result pass!")
            else:
                print("case2 verify result failed!")

    def test_custom_op_case3(self):
        input_x = torch.linspace(1, 10, 7* 1025)
        input_gamma= torch.linspace(0, 1, 1025)
        x = input_x.view(7,1025).to(torch.bfloat16)
        gamma = input_gamma.view(1025).to(torch.bfloat16)
        epsilon = 1e-6
        mean = torch.mean(x ** 2, dim=1, keepdim=True)
        rstd = torch.rsqrt(mean + epsilon)
        golden = gamma * (x * rstd)

        x_npu = x.npu()
        gamma_npu = gamma.npu()

        # 修改输入
        output = custom_ops_lib.custom_op(x_npu, gamma_npu, epsilon)
        if output is None:
            print("case3 execution timed out!")
        else:
            output_y = output[0].cpu()
            rstd_output = output[1].cpu()
            if verify_result(output_y, golden) and verify_result(rstd_output, rstd):
                print("case3 verify result pass!")
            else:
                print("case3 verify result failed!")

    def test_custom_op_case4(self):
        input_x = np.random.uniform(1, 10, [11,1023,1023]).astype(np.float32)
        input_gamma = np.random.uniform(0,1, [1023,1023]).astype(np.float32)
        epsilon = 1e-6
        x = torch.from_numpy(input_x)
        gamma = torch.from_numpy(input_gamma)
        mean = torch.mean(x ** 2, dim=(1,2), keepdim=True)
        rstd = torch.rsqrt(mean + epsilon)
        golden = gamma * (x * rstd)

        x_npu = x.npu()
        gamma_npu = gamma.npu()

        # 修改输入
        output = custom_ops_lib.custom_op(x_npu, gamma_npu, epsilon)
        if output is None:
            print("case4 execution timed out!")
        else:
            output_y = output[0].cpu()
            rstd_output = output[1].cpu()
            if verify_result(output_y, golden) and verify_result(rstd_output, rstd):
                print("case4 verify result pass!")
            else:
                print("case4 verify result failed!")

    def test_custom_op_case5(self):
        np.random.seed(48)
        input_x = np.random.uniform(1, 10, [48,1024,1024]).astype(np.float32)
        input_gamma = np.random.uniform(0,1, [1024,1024]).astype(np.float32)
        epsilon = 1e-6
        x = torch.from_numpy(input_x)
        gamma = torch.from_numpy(input_gamma)
        mean = torch.mean(x ** 2, dim=(1,2), keepdim=True)
        rstd = torch.rsqrt(mean + epsilon)
        golden = gamma * (x * rstd)

        x_2 = x.clone()
        gamma_2 = gamma.clone()

        x_npu = x.npu()
        gamma_npu = gamma.npu()

        x_npu2 = x_2.npu()
        gamma_npu2 = gamma_2.npu()
        # 多次执行测试性能，第一次为预热，统计性能时选第二次的耗时。
        output = custom_ops_lib.custom_op(x_npu, gamma_npu, epsilon)
        # 修改输入
        output = custom_ops_lib.custom_op(x_npu2, gamma_npu2, epsilon)

        if output is None:
            print("case5 execution timed out!")
        else:
            output_y = output[0].cpu()
            rstd_output = output[1].cpu()
            if verify_result(output_y, golden) and verify_result(rstd_output, rstd):
                print("case5 verify result pass!")
            else:
                print("case5 verify result failed!")

if __name__ == "__main__":
    print(sys.argv)
    if sys.argv[1] == '1':
        TestCustomOP().test_custom_op_case1()
    elif sys.argv[1] == '2':
        TestCustomOP().test_custom_op_case2()
    elif sys.argv[1] == '3':
        TestCustomOP().test_custom_op_case3()
    elif sys.argv[1] == '4':
        TestCustomOP().test_custom_op_case4()
    elif sys.argv[1] == '5':
        TestCustomOP().test_custom_op_case5() 
