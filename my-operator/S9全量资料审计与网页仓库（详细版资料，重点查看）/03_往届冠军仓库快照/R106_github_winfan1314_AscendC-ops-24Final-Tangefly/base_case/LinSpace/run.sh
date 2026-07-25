#!/bin/bash
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
  # 清除上次测试性能文件
#rm -rf ./dist/*
if [ -d "./dist" ]; then
    if [ "$(ls -A "./dist")" ]; then
    pip3 install dist/custom_ops*.whl --force-reinstall >/dev/null 2>&1
    else
        python3 setup.py build bdist_wheel >/dev/null 2>&1
        pip3 install dist/custom_ops*.whl --force-reinstall >/dev/null 2>&1
    fi
else
    python3 setup.py build bdist_wheel >/dev/null 2>&1
    pip3 install dist/custom_ops*.whl --force-reinstall >/dev/null 2>&1
fi


# timeout 180  msprof --application="python3 test_op.py $1"
# python3 get_time.py

timeout 180  python3 test_op.py $1
if [ $? -eq 124 ]; then
    echo "case${i} execution timed out!"
    exit 1
fi
