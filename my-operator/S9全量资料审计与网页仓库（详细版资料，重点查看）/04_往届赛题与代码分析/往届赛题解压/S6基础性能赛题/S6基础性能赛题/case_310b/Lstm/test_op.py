import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  
from datetime import datetime

case_data = {
    'case1': {
        "input_size": 64,
        "batch_size":1,
        "seq_len":32,
        "num_layers": 1,
        "has_biases": True,
        "batch_first": False,
        "dropout": 0.0,
        "bidirectional": False,
        "train": True
    }
}


def tensor_to_npu(tensor_list):
    tensor_list_npu= []
    if tensor_list is not None and len(tensor_list) > 0:
        for i in tensor_list:
            tensor_list_npu.append(i.npu())
        return tensor_list_npu
    else:
        return None

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

    
class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        start = datetime.now()
        caseNmae='case'+str(num)
        input_size = case_data[caseNmae]["input_size"]
        batch_size = case_data[caseNmae]["batch_size"]
        seq_len = case_data[caseNmae]["seq_len"]
        num_layers = case_data[caseNmae]["num_layers"]
        has_biases = case_data[caseNmae]["has_biases"]
        batch_first = case_data[caseNmae]["batch_first"]
        dropout = case_data[caseNmae]["dropout"]
        bidirectional = case_data[caseNmae]["bidirectional"]
        train = case_data[caseNmae]["train"]
        hidden_size = 2 * input_size

        input_shape = [seq_len, batch_size, input_size]
        if batch_first:
            input_shape = [batch_size, seq_len, input_size]
        num_directions = 2 if bidirectional else 1

        h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size, dtype=torch.float32)
        c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size, dtype=torch.float32)
        params = []
        for layer in range(num_layers):
            # 确定输入维度（第一层用input_size，其他层用hidden_size）
            input_dim = input_size if layer == 0 else hidden_size
            # 正向权重
            params.append(torch.randn(4 * hidden_size, input_dim, dtype=torch.float32))  # weight_ih
            params.append(torch.randn(4 * hidden_size, hidden_size, dtype=torch.float32))  # weight_hh
            if has_biases:
                params.append(torch.randn(4 * hidden_size, dtype=torch.float32))  # bias_ih
                params.append(torch.randn(4 * hidden_size, dtype=torch.float32))  # bias_hh

            # 如果是双向，添加反向权重
            if bidirectional:
                params.append(torch.randn(4 * hidden_size, input_dim, dtype=torch.float32))  # weight_ih_reverse
                params.append(torch.randn(4 * hidden_size, hidden_size, dtype=torch.float32))  # weight_hh_reverse
                if has_biases:
                    params.append(torch.randn(4 * hidden_size, dtype=torch.float32))  # bias_ih_reverse
                    params.append(torch.randn(4 * hidden_size, dtype=torch.float32))  # bias_hh_reverse

        params = tuple(params)
        input = torch.randn(input_shape).to(torch.float32)
        output,hn, cn = torch.lstm(input,
                                     (h0, c0),
                                     params,
                                     has_biases=has_biases,
                                     num_layers=num_layers,
                                     dropout=dropout,
                                     train=train,
                                     bidirectional=bidirectional,
                                     batch_first=batch_first)
        end = datetime.now()
        delta = end - start
        print(f"运行时间：{delta.total_seconds():.6f} 秒")

        a = input.npu()
        b = tensor_to_npu([h0,c0])
        c = tensor_to_npu(list(params))
        end = datetime.now()
        delta = end - start
        print(f"运行时间：{delta.total_seconds():.6f} 秒")

        # 修改输入
        #output_npu, hn_npu, cn_npu = custom_ops_lib.custom_op(input.npu(), tensor_to_npu([h0,c0]), tensor_to_npu(list(params)), 
        output_npu, hn_npu, cn_npu = custom_ops_lib.custom_op(a, b, c, 
                                     has_biases,
                                     num_layers,
                                     dropout,
                                     train,
                                     bidirectional,
                                     batch_first, output, hn, cn)
        if output_npu is None or hn_npu is None or cn_npu is None:
            print(f"{caseNmae} execution timed out!")
        else:

            if verify_result(output_npu.cpu(), output) and verify_result(hn_npu.cpu(), hn) and verify_result(cn_npu.cpu(), cn):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")
        end = datetime.now()
        delta = end - start
        print(f"运行时间：{delta.total_seconds():.6f} 秒")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
