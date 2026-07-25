import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
from copy import deepcopy
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
    if golden.dtype == np.float16:
        loss = 1e-2  # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一 
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
    def test_custom_op_case1(self):
        input_x = np.random.uniform(1, 100, [32, 32]).astype(np.float16)
        input_src = np.random.uniform(1,100, [32, 32]).astype(np.float16)
        input_index = np.random.uniform(0,31, [32, 32]).astype(np.int64)
        reduce = "sum"
        dim = 1
        include_self = False
        input_x_cpu = torch.from_numpy(input_x)
        input_src_cpu = torch.from_numpy(input_src)
        input_index_cpu = torch.from_numpy(input_index)
        input_x_npu = input_x_cpu.npu()
        input_src_npu = input_src_cpu.npu()
        input_index_npu = torch.from_numpy(input_index.astype(np.int32)).npu()
        cpu_result = torch.scatter_reduce(input=input_x_cpu,dim=dim,index=input_index_cpu,src=input_src_cpu,reduce=reduce,include_self=include_self)
        # 修改输入
        output = custom_ops_lib.custom_op(input_x_npu, input_index_npu,input_src_npu, dim, reduce,include_self)
        
        if output is None:
            print("case1 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, cpu_result.numpy()):
                print("case1 verify result pass!")
            else:
                print("case1 verify result failed!")

    def test_custom_op_case2(self):

        input_x = np.random.uniform(1, 100, [128, 128]).astype(np.float32)
        input_src = np.random.uniform(1,100, [128, 128]).astype(np.float32)
        input_index = np.random.uniform(0,127, [128, 128]).astype(np.int64)
        reduce = "sum"
        dim = 1
        include_self = False
        input_x_cpu = torch.from_numpy(input_x)
        input_src_cpu = torch.from_numpy(input_src)
        input_index_cpu = torch.from_numpy(input_index)
        input_x_npu = input_x_cpu.npu()
        input_src_npu = input_src_cpu.npu()
        input_index_npu = torch.from_numpy(input_index.astype(np.int32)).npu()
        cpu_result = torch.scatter_reduce(input=input_x_cpu,dim=dim,index=input_index_cpu,src=input_src_cpu,reduce=reduce,include_self=include_self)
        # 修改输入
        output = custom_ops_lib.custom_op(input_x_npu, input_index_npu,input_src_npu, dim, reduce,include_self)
        
        if output is None:
            print("case2 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, cpu_result.numpy()):
                print("case2 verify result pass!")
            else:
                print("case2 verify result failed!")

    def test_custom_op_case3(self):
        input_x = np.random.uniform(1, 100, [32,32, 63]).astype(np.float32)
        input_src = np.random.uniform(1,100, [32,32, 63]).astype(np.float32)
        input_index = np.random.uniform(0,32, [32,32, 63]).astype(np.int64)
        reduce = "sum"
        dim = 2
        include_self = True
        input_x_cpu = torch.from_numpy(input_x)
        input_src_cpu = torch.from_numpy(input_src)
        input_index_cpu = torch.from_numpy(input_index)
        input_x_npu = input_x_cpu.npu()
        input_src_npu = input_src_cpu.npu()
        input_index_npu = torch.from_numpy(input_index.astype(np.int32)).npu()
        cpu_result = torch.scatter_reduce(input=input_x_cpu,dim=dim,index=input_index_cpu,src=input_src_cpu,reduce=reduce,include_self=include_self)
        # 修改输入
        output = custom_ops_lib.custom_op(input_x_npu, input_index_npu,input_src_npu, dim, reduce,include_self)
        if output is None:
            print("case3 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, cpu_result.numpy()):
                print("case3 verify result pass!")
            else:
                print("case3 verify result failed!")

    def test_custom_op_case4(self):
        input_x = np.random.uniform(1, 100, [512,32, 16]).astype(np.float32)
        input_src = np.random.uniform(1,100, [512,32, 16]).astype(np.float32)
        input_index = np.random.uniform(0,15, [512,32, 16]).astype(np.int64)
        reduce = "amax"
        dim = 1
        include_self = False
        input_x_cpu = torch.from_numpy(input_x)
        input_src_cpu = torch.from_numpy(input_src)
        input_index_cpu = torch.from_numpy(input_index)

        input_x_npu = input_x_cpu.npu()
        input_src_npu = input_src_cpu.npu()
        input_index_npu = torch.from_numpy(input_index.astype(np.int32)).npu()
        

        cpu_result = torch.scatter_reduce(input=input_x_cpu,dim=dim,index=input_index_cpu,src=input_src_cpu,reduce=reduce,include_self=include_self)
        # 修改输入
        output = custom_ops_lib.custom_op(input_x_npu, input_index_npu,input_src_npu, dim, reduce,include_self)
        
        if output is None:
            print("case4 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, cpu_result.numpy()):
                print("case4 verify result pass!")
            else:
                print("case4 verify result failed!")

    def test_custom_op_case5(self):
        np.random.seed(34)
        input_x = np.random.uniform(1, 100, [512,128, 32]).astype(np.float32)
        input_src = np.random.uniform(1,100, [512,128, 32]).astype(np.float32)
        input_index = np.random.uniform(0,31, [512,128, 32]).astype(np.int64)
        reduce = "amin"
        dim = 0
        include_self = False
        input_x_cpu = torch.from_numpy(input_x)
        input_src_cpu = torch.from_numpy(input_src)
        input_index_cpu = torch.from_numpy(input_index)
        input_x_npu = input_x_cpu.npu()
        input_src_npu = input_src_cpu.npu()
        input_index_npu = torch.from_numpy(input_index.astype(np.int32)).npu()

        input_x_cpu_2 = deepcopy(input_x_cpu)
        input_src_cpu_2 = deepcopy(input_src_cpu)
        input_index_cpu_2 = deepcopy(input_index_cpu)

        input_x_npu_2 = input_x_cpu_2.npu()
        input_src_npu_2 = input_src_cpu_2.npu()
        input_index_npu_2 = torch.from_numpy(input_index.astype(np.int32)).npu()

        cpu_result = torch.scatter_reduce(input=input_x_cpu,dim=dim,index=input_index_cpu,src=input_src_cpu,reduce=reduce,include_self=include_self)
        # 修改输入
        output = custom_ops_lib.custom_op(input_x_npu, input_index_npu,input_src_npu, dim, reduce,include_self)
        output = custom_ops_lib.custom_op(input_x_npu_2, input_index_npu_2,input_src_npu_2, dim, reduce,include_self)
        
        if output is None:
            print("case5 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, cpu_result.numpy()):
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
