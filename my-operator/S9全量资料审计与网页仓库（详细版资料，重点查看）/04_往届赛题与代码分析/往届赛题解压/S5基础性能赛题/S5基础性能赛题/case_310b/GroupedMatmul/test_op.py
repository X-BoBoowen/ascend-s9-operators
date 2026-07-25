import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

def get_tensor_list(shape_list, type):
    tensor_list = []
    for i in range(len(shape_list)):
        x = torch.rand([*shape_list[i]])
        torch_tensor = x * 50 - 25
        torch_tensor = torch_tensor.to(type)
        tensor_list.append(torch_tensor)
    return tensor_list

def tensor_to_npu(tensor_list):
    tensor_list_npu= []
    if tensor_list is not None and len(tensor_list) > 0:
        for i in tensor_list:
            tensor_list_npu.append(i.npu())
        return tensor_list_npu
    else:
        return None
  
    
case_data = {
    'case1': {
        'x': get_tensor_list([[416, 7168]],torch.int8),
        'weight': get_tensor_list([[256, 7168, 256]],torch.int8),
        'bias': get_tensor_list([[256]],torch.int32),
        'scale': get_tensor_list([[256, 256]],torch.bfloat16),
        'per_token_scale':get_tensor_list([[256]],torch.float32),
        'group_list':torch.from_numpy(np.array([128,256],dtype=np.int64)),
        'split_item':3,
        'group_type':0,
        'group_list_type':0,
        'act_type':0
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

def verify_result_list(real_result_list, golden_list):
    for i in range(len(real_result_list)):
        real_result = real_result_list[i].cpu()
        golden = golden_list[i]

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
            if torch.sum(result_rtol == False) > real_result.numel() * loss and torch.sum(result_atol == False) > real_result.numel() * loss:  # 误差超出预期时返回打印错误，返回对比失败
                print("[ERROR] result error")
                return False
    print("test pass")
    return True

def replcace_inf(tensor):
    mask_pos_inf = tensor == float('inf')
    mask_neg_inf = tensor == -float('inf')
    max_val = torch.finfo(torch.half).max  # 65504.0
    min_val = torch.finfo(torch.half).min  # 6.103515625e-05

    tensor_replaced = tensor.clone()
    tensor_replaced = torch.where(mask_pos_inf, max_val, tensor_replaced)
    tensor_replaced = torch.where(mask_neg_inf, min_val, tensor_replaced)
    return tensor_replaced

def grouped_matmul_v4(x,weight,bias,scale,per_token_scale, group_list):
    
    groupList = group_list

    pertokenScale = torch.tensor([])
    hasBias = True
    hasPertokenScale = True
  

    pertokenScale = per_token_scale

    n = weight.shape[2]
    scale_dtype = scale.dtype
    if scale_dtype == torch.float32:
        output_dtype = torch.float16
    else:
        output_dtype = torch.bfloat16
    # grouped matmul
    result = torch.empty(0, n, dtype=torch.float32)
    last = 0
    index = 0
    for i in groupList.tolist():
        x_tensor = x[last:i, :].to(torch.int32)
        weight_tensor = weight[index].to(torch.int32)
        scale_tensor = scale[index].clone().detach().to(torch.float32)
        matmul_res = torch.matmul(x_tensor, weight_tensor)
        if not hasBias:
            cur_res = matmul_res.to(torch.float32) * scale_tensor.unsqueeze(0)
        else:
            bias_tensor = bias[index].clone().detach()
            cur_res = (matmul_res + bias_tensor.unsqueeze(0)).to(torch.float32) * scale_tensor.unsqueeze(0)
        if hasPertokenScale:
            pertokenScale_tensor = pertokenScale[last:i]
            cur_res = cur_res * pertokenScale_tensor.reshape(-1,1)
        index += 1
        result = torch.cat([result, cur_res], dim=0)
        last = i
    result = result.to(output_dtype)
    if output_dtype == torch.float16:
        result = replcace_inf(result)
    return [result]
    

class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseNmae='case'+str(num)

        input_x = case_data[caseNmae]["x"]
        input_weight = case_data[caseNmae]["weight"]
        input_bias = case_data[caseNmae]["bias"]
        input_scale = case_data[caseNmae]["scale"]
        per_token_scale = case_data[caseNmae]["per_token_scale"]
        group_list = case_data[caseNmae]["group_list"]
        split_item = case_data[caseNmae]["split_item"]
        group_type = case_data[caseNmae]["group_type"]
        group_list_type = case_data[caseNmae]["group_list_type"]
        act_type = case_data[caseNmae]["act_type"]


        golden = grouped_matmul_v4(input_x[0], input_weight[0], input_bias[0], input_scale[0], per_token_scale[0], group_list)

  
        result = custom_ops_lib.custom_op(tensor_to_npu(input_x), tensor_to_npu(input_weight), 
                                          tensor_to_npu(input_bias), tensor_to_npu(input_scale), 
                                          tensor_to_npu(per_token_scale), 
                                          group_list, 
                                          split_item, 
                                          group_type,
                                          group_list_type,
                                          act_type,
                                          golden, 
                                          int(num))
      
  
        if result is None:
            print(f"{caseNmae} execution timed out!")
        else:
            if verify_result_list(result, golden):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
