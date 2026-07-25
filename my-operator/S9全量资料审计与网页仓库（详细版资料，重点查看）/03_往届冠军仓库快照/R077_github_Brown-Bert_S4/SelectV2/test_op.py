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
    t1 = np.array(real_result)
    t2 = np.array(golden)
    t1 = t1.flatten()
    t2 = t2.flatten()
    if golden.dtype == np.float32:
        loss = 1e-4  # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    else:
        loss = 1e-3  # 容忍偏差，一般fp32要求绝对误差和相对误差均不超过百万分之一
    minimum = 10e-10
    result = np.abs(real_result - golden)  # 计算运算结果和预期结果偏差
    # for i in range(result.size):
    #     if t1[i] != t2[i]:
    #         print("real_result[%d]: %f, golden[%d]: %f" % (i, t1[i], i, t2[i]))
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
        condition = np.random.rand(2, 3969, 8) > 0.5
        condition = condition.astype(np.bool_)
        then = np.random.uniform(1, 100, [2, 3969, 8]).astype(np.int32)
        elsedo = np.random.uniform(1, 100, [1, 3969, 8]).astype(np.int32)
        cpu_result = tf.raw_ops.SelectV2(condition=condition, t=then, e=elsedo).numpy()

        condition_npu = torch.from_numpy(condition).npu()
        then_npu = torch.from_numpy(then).npu()
        elsedo_npu = torch.from_numpy(elsedo).npu()

        # 修改输入
        output = run_with_timeout(custom_ops_lib.custom_op, args=(condition_npu, then_npu, elsedo_npu), timeout=30)
        if output is None:
            print("case1 execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, cpu_result):
                print("case1 verify result pass!")
            else:
                print("case1 verify result failed!")


if __name__ == "__main__":
    print(sys.argv)
    if sys.argv[1] == '1':
        TestCustomOP().test_custom_op_case1()
    
