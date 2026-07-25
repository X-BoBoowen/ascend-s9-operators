import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

case_data = {
    'case1': {
        'y_grad':np.random.uniform(-1, 1, [256]).astype(np.float16),
        'input_x':np.random.uniform(-1, 1, [512]).astype(np.float16),
        'v_y':np.random.uniform(-5, 5, [256]).astype(np.float16),
        'v_x':np.random.uniform(-5, 5, [512]).astype(np.float16),
        'dim':0
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

def glu_backward(input_x, dim=-1):
    output = torch.nn.functional.glu(input_x, dim=dim)  # 沿第1维（特征维度）应用GLU
    loss = output.sum()
    loss.backward()
    return input_x.grad
    
class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseNmae='case'+str(num)
        input_y_grad = None
        input_x = None
        input_v_y = None
        input_v_x = None
        input_dim = case_data[caseNmae]["dim"]
        if int(num) == 3:
            input_y_grad = case_data[caseNmae]["y_grad"]
            input_x = case_data[caseNmae]["input_x"]
            input_v_y = case_data[caseNmae]["v_y"]
            input_v_x = case_data[caseNmae]["v_x"]
        else:
            input_y_grad = torch.from_numpy(case_data[caseNmae]["y_grad"])
            input_x = torch.from_numpy(case_data[caseNmae]["input_x"])
            input_v_y = torch.from_numpy(case_data[caseNmae]["v_y"])
            input_v_x = torch.from_numpy(case_data[caseNmae]["v_x"])
        input_x.requires_grad = True
        grad_x = glu_backward(input_x, dim=input_dim)
        
        need_cast = (input_x.dtype is not torch.float32)
        obj_type = input_x.dtype
        if need_cast:
            input_x = input_x.to(torch.float32)
            input_y_grad = input_y_grad.to(torch.float32)
            input_v_y = input_v_y.to(torch.float32)
            input_v_x = input_v_x.to(torch.float32)
            grad_x = grad_x.to(torch.float32)


        jvp_result_golden = torch.ops.aten.glu_backward_jvp(grad_x, input_y_grad, input_x, input_v_y, input_v_x, input_dim)

        if need_cast:
            input_x = input_x.to(obj_type)
            input_y_grad = input_y_grad.to(obj_type)
            input_v_y = input_v_y.to(obj_type)
            input_v_x = input_v_x.to(obj_type)
            grad_x = grad_x.to(obj_type)
            jvp_result_golden = jvp_result_golden.to(obj_type)
            
        jvp_result = custom_ops_lib.custom_op(grad_x.npu(),input_y_grad.npu(),input_x.npu(), input_v_y.npu(), input_v_x.npu(), input_dim, int(num))
        if jvp_result is None:
            print(f"{caseNmae} execution timed out!")
        else:
            jvp_result = jvp_result.cpu()
            if verify_result(jvp_result, jvp_result_golden):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
