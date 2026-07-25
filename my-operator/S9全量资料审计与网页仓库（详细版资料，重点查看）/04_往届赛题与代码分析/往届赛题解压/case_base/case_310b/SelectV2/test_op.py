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

class TestCustomOP(TestCase):
    def test_custom_op_case1(self):
        condition = np.random.rand(8, 2048) > 0.5
        condition = condition.astype(np.bool_)
        then = np.random.uniform(1, 100, [8, 2048]).astype(np.float16)
        elsedo = np.random.uniform(1, 100, [8, 2048]).astype(np.float16)
        cpu_result = tf.raw_ops.SelectV2(condition=condition, t=then, e=elsedo).numpy()

        condition_npu = torch.from_numpy(condition).npu()
        then_npu = torch.from_numpy(then).npu()
        elsedo_npu = torch.from_numpy(elsedo).npu()

        # 修改输入
        output = custom_ops_lib.custom_op(condition_npu, then_npu, elsedo_npu)
        if output is None:
            print("case1 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, cpu_result):
                print("case1 verify result pass!")
            else:
                print("case1 verify result failed!")

    def test_custom_op_case2(self):
        condition = np.random.rand(7, 2051) > 0.5
        condition = condition.astype(np.bool_)
        then = np.random.uniform(1, 100, [7, 2051]).astype(np.int8)
        elsedo = np.random.uniform(1, 100, [7, 2051]).astype(np.int8)
        cpu_result = tf.raw_ops.SelectV2(condition=condition, t=then, e=elsedo).numpy()

        condition_npu = torch.from_numpy(condition).npu()
        then_npu = torch.from_numpy(then).npu()
        elsedo_npu = torch.from_numpy(elsedo).npu()

        # 修改输入
        output = custom_ops_lib.custom_op(condition_npu, then_npu, elsedo_npu)
        if output is None:
            print("case2 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, cpu_result):
                print("case2 verify result pass!")
            else:
                print("case2 verify result failed!")

    def test_custom_op_case3(self):
        condition = np.random.rand(8,8,1024) > 0.5
        condition = condition.astype(np.bool_)
        then = np.random.uniform(1, 100, [8,8,1024]).astype(np.int32)
        elsedo = np.random.uniform(1, 100, [1,8,1024]).astype(np.int32)
        cpu_result = tf.raw_ops.SelectV2(condition=condition, t=then, e=elsedo).numpy()

        condition_npu = torch.from_numpy(condition).npu()
        then_npu = torch.from_numpy(then).npu()
        elsedo_npu = torch.from_numpy(elsedo).npu()

        # 修改输入
        output = custom_ops_lib.custom_op(condition_npu, then_npu, elsedo_npu)

        if output is None:
            print("case3 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, cpu_result):
                print("case3 verify result pass!")
            else:
                print("case3 verify result failed!")

    def test_custom_op_case4(self):
        condition = np.random.rand(3,7,1025, 513) > 0.5
        condition = condition.astype(np.bool_)
        then = np.random.uniform(1, 100, [3,7,1025, 513]).astype(np.float32)
        elsedo = np.random.uniform(1, 100, [3,7,1025, 513]).astype(np.float32)
        cpu_result = tf.raw_ops.SelectV2(condition=condition, t=then, e=elsedo).numpy()

        condition_npu = torch.from_numpy(condition).npu()
        then_npu = torch.from_numpy(then).npu()
        elsedo_npu = torch.from_numpy(elsedo).npu()

        # 修改输入
        output = custom_ops_lib.custom_op(condition_npu, then_npu, elsedo_npu)
        if output is None:
            print("case4 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, cpu_result):
                print("case4 verify result pass!")
            else:
                print("case4 verify result failed!")

    def test_custom_op_case5(self):
        np.random.seed(42)
        condition = np.random.rand(8,8,1024,1024) > 0.5
        condition = condition.astype(np.bool_)
        condition_tmp = condition.copy()
        then = np.random.uniform(1, 100, [8,8,1024,1024]).astype(np.float32)
        then_tmp = then.copy()
        elsedo = np.random.uniform(1, 100, [8,8,1024,1024]).astype(np.float32)
        elsedo_tmp = elsedo.copy()
        cpu_result = tf.raw_ops.SelectV2(condition=condition, t=then, e=elsedo).numpy()

        condition_npu = torch.from_numpy(condition).npu()
        then_npu = torch.from_numpy(then).npu()
        elsedo_npu = torch.from_numpy(elsedo).npu()

        condition_npu_tmp = torch.from_numpy(condition_tmp).npu()
        then_npu_tmp = torch.from_numpy(then_tmp).npu()
        elsedo_npu_tmp = torch.from_numpy(elsedo_tmp).npu()

        # 多次执行测试性能，第一次为预热，统计性能时选第二次的耗时。
        output = custom_ops_lib.custom_op(condition_npu_tmp, then_npu_tmp, elsedo_npu_tmp)
        # 修改输入
        output = custom_ops_lib.custom_op(condition_npu, then_npu, elsedo_npu)

        if output is None:
            print("case5 execution timed out!")
        else:
            output = output.cpu().numpy()
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
