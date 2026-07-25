#!/bin/bash

cann_path=/home/ma-user/Ascend/ascend-toolkit/8.1.RC1
# 执行以下文件将8个补丁文件放入指定的目录
cp -r matmul_policy.h  ${cann_path}/aarch64-linux/ascendc/include/highlevel_api/impl/matmul/policy/
cp -r scheduler_*      ${cann_path}/aarch64-linux/ascendc/include/highlevel_api/impl/matmul/scheduler/base/
cp -r matmul_utils.h   ${cann_path}/aarch64-linux/ascendc/include/highlevel_api/impl/matmul/utils/


