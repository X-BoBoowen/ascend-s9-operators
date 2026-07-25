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
        x1 = torch.linspace(1, 10, 8 * 1024)
        x2= torch.linspace(0, 4, 8 * 1024)
        x1 = x1.view(8,1024).to(torch.float16)
        x2 = x2.view(8,1024).to(torch.float16)
        cpu_result = torch.pow(x1, x2)

        x1_npu = x1.npu()
        x2_npu = x2.npu()

        # 修改输入
        output = custom_ops_lib.custom_op(x1_npu, x2_npu)
        if output is None:
            print("case1 execution timed out!")
        else:
            output = output.cpu()
            if verify_result(output, cpu_result):
                print("case1 verify result pass!")
            else:
                print("case1 verify result failed!")

    def test_custom_op_case2(self):
        x1 = torch.linspace(1, 10, 9 * 1023)
        x2= torch.linspace(0, 4, 9 * 1023)
        x1 = x1.view(9,1023).to(torch.float32)
        x2 = x2.view(9,1023).to(torch.float32)
        cpu_result = torch.pow(x1, x2)

        x1_npu = x1.npu()
        x2_npu = x2.npu()

        # 修改输入
        output = custom_ops_lib.custom_op(x1_npu, x2_npu)
        if output is None:
            print("case2 execution timed out!")
        else:
            output = output.cpu()
            if verify_result(output, cpu_result):
                print("case2 verify result pass!")
            else:
                print("case2 verify result failed!")

    def test_custom_op_case3(self):
        x1 = torch.linspace(1, 10, 8 * 6 * 1024)
        x2= torch.linspace(0, 4,  8 * 6 * 1024)
        x1 = x1.view(8,6, 1024).to(torch.bfloat16)
        x2 = x2.view(8,6, 1024).to(torch.bfloat16)
        cpu_result = torch.pow(x1, x2)

        x1_npu = x1.npu()
        x2_npu = x2.npu()

        # 修改输入
        output = custom_ops_lib.custom_op(x1_npu, x2_npu)
        if output is None:
            print("case3 execution timed out!")
        else:
            output = output.cpu()
            if verify_result(output, cpu_result):
                print("case3 verify result pass!")
            else:
                print("case3 verify result failed!")

    def test_custom_op_case4(self):
        x1 = torch.linspace(1, 10, 7*7 * 1023)
        x2= torch.linspace(0, 4, 1*7 * 1023)
        x1 = x1.view(7,7,1023).to(torch.float32)
        x2 = x2.view(1,7,1023).to(torch.float32)
        cpu_result = torch.pow(x1, x2)

        x1_npu = x1.npu()
        x2_npu = x2.npu()

        # 修改输入
        output = custom_ops_lib.custom_op(x1_npu, x2_npu)
        if output is None:
            print("case4 execution timed out!")
        else:
            output = output.cpu()
            if verify_result(output, cpu_result):
                print("case4 verify result pass!")
            else:
                print("case4 verify result failed!")

    def test_custom_op_case5(self):
        np.random.seed(42)
        x1 = np.random.uniform(1, 10, [8,8,1024,1024]).astype(np.float32)
        x1_tmp = x1.copy()
        x2 = np.random.uniform(1, 4, [8,8,1024,1024]).astype(np.float32)
        x2_tmp = x2.copy()
        torch_x1 = torch.from_numpy(x1)
        torch_x2 = torch.from_numpy(x2)
        cpu_result = torch.pow(torch_x1, torch_x2)
        x1_npu = torch.from_numpy(x1).npu()
        x2_npu = torch.from_numpy(x2).npu()
       

        x1_tmp_npu = torch.from_numpy(x1_tmp).npu()
        x2_tmp_npu = torch.from_numpy(x2_tmp).npu()

        # 多次执行测试性能，第一次为预热，统计性能时选第二次的耗时。
        output = custom_ops_lib.custom_op(x1_npu, x2_npu)
        # 修改输入
        output = custom_ops_lib.custom_op(x1_tmp_npu, x2_tmp_npu)

        if output is None:
            print("case5 execution timed out!")
        else:
            output = output.cpu()
            if verify_result(output, cpu_result):
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
