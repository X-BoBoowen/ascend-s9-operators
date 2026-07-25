#!/bin/bash
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
  # 清除上次测试性能文件
#rm -rf ./dist/*

if [ "x$1" == "x1" ]; then
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
fi


if [ "x$1" == "x5" ]; then
   rm -rf OPPROF_*
   timeout 180  msprof op --launch-skip-before-match=1 "python3 test_op.py $1"

   if [ $? -eq 124 ]; then
        echo "timed out!"
        return 1
   fi

  time_use=$(($(python3 get_time.py)))
  echo "time_use: ${time_use}"

else
   timeout 180  python3 test_op.py $1
   if [ $? -eq 124 ]; then
        echo "timed out!"
        return 1
   fi
fi