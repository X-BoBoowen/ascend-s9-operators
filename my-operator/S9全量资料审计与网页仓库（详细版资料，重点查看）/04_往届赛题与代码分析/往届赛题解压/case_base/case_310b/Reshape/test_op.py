import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
# import tensorflow as tf
import sys  
import threading
import copy

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
    if golden.dtype == np.float32:
        loss = 1e-4  # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    else:
        loss = 1e-3  # 容忍偏差，一般fp32要求绝对误差和相对误差均不超过百万分之一
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

def reshape_tensor(x, shape_tensor, axis=0, num_axes=-1):
    original_dims = len(x.shape)
    if num_axes == -1:
        num_axes = original_dims - axis
    left_shape = list(x.shape[:axis])
    right_shape = list(x.shape[axis + num_axes:])
    new_middle_shape = shape_tensor.tolist()
    new_shape = left_shape + new_middle_shape + right_shape
    reshaped_x = x.reshape(new_shape)
    return reshaped_x

# def reshape_tensor(x, shape, axis=0, num_axes=-1):
#     original_shape = list(x.shape)
#     num_dims = len(original_shape)
#     if num_axes == -1:
#         num_axes = num_dims - axis
#     left_shape = original_shape[:axis]
#     right_shape = original_shape[axis + num_axes:]
#     product_original = np.prod(original_shape[axis:axis + num_axes])
#     product_new = np.prod(shape)
#     if -1 in shape:
#         index = shape.index(-1)
#         new_shape = shape.copy()
#         new_shape[index] = product_original // (np.prod(new_shape[:index]) * np.prod(new_shape[index + 1:]))
#     else:
#         new_shape = shape
#     final_shape = left_shape + new_shape + right_shape
#     reshaped_x = x.reshape(final_shape)
#     return reshaped_x

class TestCustomOP(TestCase):
    def test_custom_op_case1(self):
        length_x = [8, 1024]
        x = (torch.rand(length_x, device='cpu') * 10 - 5).to(torch.float16)
        shape = torch.tensor([4, 2, 1024], dtype=torch.int32)
        axis = 0
        num_axes = -1
        cpu_result = reshape_tensor(x, shape, axis, num_axes).numpy()

        # 修改输入
        output = custom_ops_lib.custom_op(x.npu(), shape.npu(), axis, num_axes)
        output_cpu = output.cpu()
        output_np = output_cpu.numpy()
        if output is None:
            print("case1 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output_np, cpu_result):
                print("case1 verify result pass!")
            else:
                print("case1 verify result failed!")

    def test_custom_op_case2(self):
        length_x = [7, 8, 2048]
        x = (torch.randint(low=-5, high=5, size=length_x, device='cpu')).to(torch.int8)
        shape = torch.tensor([8, 7, 2048], dtype=torch.int32)
        axis = 0
        num_axes = -1
        cpu_result = reshape_tensor(x, shape, axis, num_axes).numpy()

        # 修改输入
        output = custom_ops_lib.custom_op(x.npu(), shape.npu(), axis, num_axes)
        output_cpu = output.cpu()
        output_np = output_cpu.numpy()
        if output is None:
            print("case2 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output_np, cpu_result):
                print("case2 verify result pass!")
            else:
                print("case2 verify result failed!")

    def test_custom_op_case3(self):
        length_x = [3,4,512,1024]
        x = (torch.rand(length_x, device='cpu') * 10 - 5).to(torch.int64)
        shape = torch.tensor([4,3,512,1024], dtype=torch.int32)
        axis = 0
        num_axes = -1
        cpu_result = reshape_tensor(x, shape, axis, num_axes).numpy()

        # 修改输入
        output = custom_ops_lib.custom_op(x.npu(), shape.npu(), axis, num_axes)
        output_cpu = output.cpu()
        output_np = output_cpu.numpy()
        if output is None:
            print("case3 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output_np, cpu_result):
                print("case3 verify result pass!")
            else:
                print("case3 verify result failed!")

    def test_custom_op_case4(self):
        length_x = [8,47,123,1023]
        x = (torch.rand(length_x, device='cpu') * 10 - 5).to(torch.float32)
        shape = torch.tensor([123,47,8,1023], dtype=torch.int32)
        axis = 0
        num_axes = -1
        cpu_result = reshape_tensor(x, shape, axis, num_axes).numpy()

        # 修改输入
        output = custom_ops_lib.custom_op(x.npu(), shape.npu(), axis, num_axes)
        output_cpu = output.cpu()
        output_np = output_cpu.numpy()
        if output is None:
            print("case4 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output_np, cpu_result):
                print("case4 verify result pass!")
            else:
                print("case4 verify result failed!")

    def test_custom_op_case5(self):
        length_x = [4,4,512,2048]
        x = (torch.rand(length_x, device='cpu') * 10 - 5).to(torch.float32)
        x_2 = copy.deepcopy(x)
        shape = torch.tensor([2,2,2048,2048], dtype=torch.int32)
        shape_2 = copy.deepcopy(shape)
        axis = 0
        num_axes = -1
        cpu_result = reshape_tensor(x, shape, axis, num_axes).numpy()

        # 修改输入
        output = custom_ops_lib.custom_op(x.npu(), shape.npu(), axis, num_axes)
        output = custom_ops_lib.custom_op(x_2.npu(), shape_2.npu(), axis, num_axes)
        output_cpu = output.cpu()
        output_np = output_cpu.numpy()
        if output is None:
            print("case5 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output_np, cpu_result):
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
