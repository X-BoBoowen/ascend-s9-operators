import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import tensorflow as tf
import sys  
import threading
from typing import Optional, Tuple

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# case_data = {
#     'case1': {
#         'A_shape': [2, 1, 1],
#         'data_type': np.float32,
#         'B_shape': [2, 1, 1],
#         'bias_shape': [2, 1, 1]
#     }
# }

case_data = {
    'case1': {
        'A_shape': [633,133],
        'data_type': np.float32,
        'B_shape': [133,338],
        'bias_shape': [1,1]
    }
}
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
    # 容忍偏差
    if golden.dtype == np.float16:
        loss = 1e-3
    else:
        loss = 1e-4
    
    minimum = 10e-10
    result = np.abs(real_result - golden)  # 计算复数模长的绝对误差
    deno = np.maximum(np.abs(real_result), np.abs(golden))  # 分母取两数模长较大者
    result_atol = np.less_equal(result, loss)  # 绝对误差是否满足
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss)  # 相对误差是否满足
    
    # 找到所有不符合要求的位置
    # error_mask = ~(result_atol & result_rtol)
    error_mask = ~(result_rtol)
    error_indices = np.argwhere(error_mask)
    
    if error_indices.size > 0:
        print(f"[ERROR] 共发现 {len(error_indices)} 处误差超限")
        max_errors_to_show = 10  # 最多显示前10个错误
        for idx, pos in enumerate(error_indices[:max_errors_to_show]):
            i, j = pos[0], pos[1]
            real_val = real_result[i, j]
            golden_val = golden[i, j]
            abs_error = result[i, j]
            rel_error = abs_error / (max(np.abs(real_val), np.abs(golden_val)) + minimum)
            
            print(f"位置 ({i}, {j}):")
            print(f"  真实值模长: {np.abs(real_val):.7e} (实部 {real_val.real:.7e}, 虚部 {real_val.imag:.7e})")
            print(f"  期望值模长: {np.abs(golden_val):.7e} (实部 {golden_val.real:.7e}, 虚部 {golden_val.imag:.7e})")
            print(f"  绝对误差: {abs_error:.7e} (阈值 {loss:.1e})")
            print(f"  相对误差: {rel_error:.7e} (阈值 {loss:.1e})\n")
            
        if len(error_indices) > max_errors_to_show:
            print(f"（仅显示前{max_errors_to_show}处错误，剩余错误已省略）")
        return False
    print("test pass")
    return True

def verify_result_base(real_result, golden):
      # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    if golden.dtype == np.float16:
        loss = 1e-3
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
    def test_custom_op_case(self,num):
        print(num)
        caseNmae='case'+num
        tensor_input_real = np.random.uniform(1, 100,case_data[caseNmae]['A_shape']).astype(case_data[caseNmae]['data_type'])
        tensor_input_imag = np.random.uniform(1, 100,case_data[caseNmae]['A_shape']).astype(case_data[caseNmae]['data_type'])

        tensor_values_real = np.random.uniform(1, 100,case_data[caseNmae]['B_shape']).astype(case_data[caseNmae]['data_type'])
        tensor_values_imag = np.random.uniform(1, 100,case_data[caseNmae]['B_shape']).astype(case_data[caseNmae]['data_type'])
        
        complex_tensor_A = torch.complex(torch.from_numpy(tensor_input_real), torch.from_numpy(tensor_input_imag))
        complex_tensor_B = torch.complex(torch.from_numpy(tensor_values_real), torch.from_numpy(tensor_values_imag))

        tensor_bias_real = np.random.uniform(1, 100,case_data[caseNmae]['bias_shape']).astype(case_data[caseNmae]['data_type'])
        tensor_bias_imag = np.random.uniform(1, 100,case_data[caseNmae]['bias_shape']).astype(case_data[caseNmae]['data_type'])

        tensor_bias = None
        tensor_bias = torch.complex(
                torch.from_numpy(tensor_bias_real),
                torch.from_numpy(tensor_bias_imag)
            )

        # print(f"tensor bias: {tensor_bias}")

        golden = torch.matmul(complex_tensor_A, complex_tensor_B).numpy()
        # print(f"golden without bias: {golden}")
        if (tensor_bias is not None):
            golden += tensor_bias.numpy()
            # print(f"golden with bias: {golden}")
            tensor_bias_npu = tensor_bias.npu();
        else:
            tensor_bias_npu = None
        # ---- 新增部分：保存矩阵到文件 ----
        # 定义保存路径
        output_file = "matrices.txt"
        
        # # 提取实部和虚部为NumPy数组
        # A_real = complex_tensor_A.real.numpy()
        # A_imag = complex_tensor_A.imag.numpy()
        # B_real = complex_tensor_B.real.numpy()
        # B_imag = complex_tensor_B.imag.numpy()
        
        # if len(A_real) == 3:
        #     A_real = A_real[0]
        #     A_imag = A_imag[0]
        #     B_real = B_real[0]
        #     B_imag = B_imag[0]
        # # 保存到文件（科学计数法格式，保留4位小数）
        # with open(output_file, 'w') as f:
        #     # 写入矩阵A的实部
        #     f.write("Matrix A Real Part:\n")
        #     np.savetxt(f, A_real, fmt='%.4e', header='', footer='\n')
            
        #     # 写入矩阵A的虚部
        #     f.write("Matrix A Imaginary Part:\n")
        #     np.savetxt(f, A_imag, fmt='%.4e', header='', footer='\n')
            
        #     # 写入矩阵B的实部
        #     f.write("Matrix B Real Part:\n")
        #     np.savetxt(f, B_real, fmt='%.4e', header='', footer='\n')
            
        #     # 写入矩阵B的虚部
        #     f.write("Matrix B Imaginary Part:\n")
        #     np.savetxt(f, B_imag, fmt='%.4e', header='', footer='\n')
        
        # print(f"矩阵已保存到文件: {output_file}")
        # # ---- 保存部分结束 ----

        #  # ---- 新增部分：保存四个矩阵乘积到第二个文件 ----
        # # 计算四个组合的乘积
        # product_rr = np.dot(tensor_input_real, tensor_values_real)    # A_real * B_real
        # product_ri = np.dot(tensor_input_real, tensor_values_imag)    # A_real * B_imag
        # product_ir = np.dot(tensor_input_imag, tensor_values_real)    # A_imag * B_real
        # product_ii = np.dot(tensor_input_imag, tensor_values_imag)    # A_imag * B_imag

        # # 定义第二个输出文件路径
        # product_file = "matrix_products.txt"
        
        # # 保存到新文件（科学计数法，保留4位小数）
        # with open(product_file, 'w') as f:
        #     # 写入A_real * B_real
        #     f.write("===== A_real * B_real =====\n")
        #     np.savetxt(f, product_rr, fmt='%.4e', delimiter='\t', header='', footer='\n\n')
            
        #     # 写入A_real * B_imag
        #     f.write("===== A_real * B_imag =====\n")
        #     np.savetxt(f, product_ri, fmt='%.4e', delimiter='\t', header='', footer='\n\n')
            
        #     # 写入A_imag * B_real
        #     f.write("===== A_imag * B_real =====\n")
        #     np.savetxt(f, product_ir, fmt='%.4e', delimiter='\t', header='', footer='\n\n')
            
        #     # 写入A_imag * B_imag
        #     f.write("===== A_imag * B_imag =====\n")
        #     np.savetxt(f, product_ii, fmt='%.4e', delimiter='\t', header='', footer='\n')
        
        # print(f"矩阵乘积已保存到文件: {product_file}")
        # # ---- 新增部分结束 ----
        
        
        tensor_input_npu = complex_tensor_A.npu()
        tensor_values_npu = complex_tensor_B.npu()

        # 修改输入
        output = run_with_timeout(custom_ops_lib.custom_op, args=(tensor_input_npu, tensor_values_npu, tensor_bias_npu), timeout=5)


        if output is None:
            print(f"{caseNmae} execution timed out!")
        else:
            output = output.cpu().numpy()
            print("Output shape:", output.shape)
            
            #   # 将output保存到文件
            # output_filename = f"output_{caseNmae}.txt"
            # np.savetxt(output_filename, output, fmt="%.4e")  # 科学计数法，保留4位小数
            # print(f"Output saved to {output_filename}")
            
            # # 将golden保存到文件
            # golden_filename = f"golden_{caseNmae}.txt"
            # np.savetxt(golden_filename, golden, fmt="%.4e")  # 科学计数法，保留4位小数
            # print(f"Golden saved to {golden_filename}")

            if verify_result_base(output, golden):
                print(f"{caseNmae} verify result pass!")
            else:
                # print(output)
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
