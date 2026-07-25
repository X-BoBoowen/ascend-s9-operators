#!/bin/bash
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
  # 清除上次测试性能文件
#rm -rf ./dist/*
if [ -d "./dist" ]; then
    if [ "$(ls -A "./dist")" ]; then
    echo "已存在whl"
    pip3 install dist/custom_ops*.whl --force-reinstall
    else
        echo "重新生成whl"
        python3 setup.py build bdist_wheel
        pip3 install dist/custom_ops*.whl --force-reinstall
    fi
else
    echo "重新生成whl"
    python3 setup.py build bdist_wheel
    pip3 install dist/custom_ops*.whl --force-reinstall
fi

if [ $# -gt 1 ]; then
    timeout 180  msprof --application="python3 test_op.py $1"
    python3 get_time.py
else
    timeout 180  python3 test_op.py $1
    if [ $? -eq 124 ]; then
        echo "case${i} execution timed out!"
        exit 1
    fi
fi
