import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

# RANDOM_SAMPLES_SEED = 42

def generate_random_samples(shape,dtype):
    #np.random.seed(RANDOM_SAMPLES_SEED)  # 只为_random_samples设置种子
    return np.random.uniform(0, 1,shape).astype(dtype)

def generate_random_2(shape):
    # torch.manual_seed(42)  # 42是一个常用的随机种子，你可以换成任何整数
    ten = torch.rand(shape, dtype=torch.bfloat16)  # 只为_random_samples设置种子
    return ten

case_data = {
    'case1': {
        'input':np.random.uniform(-1, 1, [3,32,1024,1024]).astype(np.float16),
        '_random_samples':generate_random_samples([1,3,3],np.float16),
        'kernel_size':3,
        'output_size':(16, 512,512),
        'output_ratio':(),
        'return_indices':False
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

    
def ensure_tuple(variable):
    # 判断是否为元组
    if isinstance(variable, tuple):
        # 是元组则直接返回
        return variable
    else:
        # 不是元组则转换为单元素元组
        return (variable,)
def assign_values(tpl):
    # 确保输入是元组
    if not isinstance(tpl, tuple):
        tpl = (tpl,)  # 非元组转为单元素元组
    
    # 判断元组是否为空
    if tpl == ():  # 空元组
        x, y, z = 0.0, 0.0, 0.0  # 返回浮点类型的0
    else:
        # 非空元组解包（假设为3个元素）
        if len(tpl) != 3:
            raise ValueError("非空元组必须包含3个元素")
        x, y, z = tpl  # 若元组元素为float，直接沿用；若为int，会自动转为float
    return x, y, z
    
class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseName='case'+str(num) 
        input_x = None
        output = None
        return_indices_cpu = None
        #获取数据
        if int(num) == 3:
            input_x = case_data[caseName]["input"]
            random_samples = case_data[caseName]["_random_samples"]
        else:
            input_x = torch.from_numpy(case_data[caseName]["input"])
            random_samples = torch.from_numpy(case_data[caseName]["_random_samples"])
        # 生成标杆
        if case_data[caseName]['return_indices']:
            if case_data[caseName]['output_ratio']:
                m = torch.nn.FractionalMaxPool3d(case_data[caseName]['kernel_size'], output_ratio=case_data[caseName]['output_ratio'],return_indices=True,_random_samples=random_samples)
            else:
                m = torch.nn.FractionalMaxPool3d(case_data[caseName]['kernel_size'], output_size=case_data[caseName]['output_size'],return_indices=True,_random_samples=random_samples)
            output, return_indices_cpu = m(input_x)
        else:
            if case_data[caseName]['output_ratio']:
                m = torch.nn.FractionalMaxPool3d(case_data[caseName]['kernel_size'], output_ratio=case_data[caseName]['output_ratio'],return_indices=False,_random_samples=random_samples)
            else:
                m = torch.nn.FractionalMaxPool3d(case_data[caseName]['kernel_size'], output_size=case_data[caseName]['output_size'],return_indices=False,_random_samples=random_samples)
            output = m(input_x)
        # 调用自定义算子
        kenel_size_npu = case_data[caseName]['kernel_size']
        kenel_size_npu = ensure_tuple(kenel_size_npu)
        output_ratio_t,output_ratio_h,output_ratio_w = assign_values(case_data[caseName]['output_ratio'])
        output_npu_all= custom_ops_lib.custom_op(input_x.npu(),random_samples.npu(), kenel_size_npu, case_data[caseName]['output_size'], output_ratio_t,output_ratio_h,output_ratio_w, case_data[caseName]['return_indices'],tuple(output.shape))
        # 对比结果
        output_npu = output_npu_all[0]
        indices = output_npu_all[1]
        if case_data[caseName]['return_indices']:
            if output_npu is None or indices is None:
                print(f"{caseName} execution timed out!")
            else:
                if verify_result(indices.cpu(), return_indices_cpu) and verify_result(output_npu.cpu(), output):
                    print(f"{caseName} verify result pass!")
                else:
                    print(f"{caseName} verify result failed!")
        else:
            if output_npu is None:
                print(f"{caseName} execution timed out!")
            else:
                if verify_result(output_npu.cpu(), output):
                    print(f"{caseName} verify result pass!")
                else:
                    print(f"{caseName} verify result failed!")



if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
