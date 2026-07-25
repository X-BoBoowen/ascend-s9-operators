#!/bin/bash

op=$(basename $(pwd))

run_install() {
    pushd $op
    source build.sh
    if [ $? -eq 0 ]; then
        ./custom_opp_euleros_aarch64.run
    else
        echo "build error!"
    fi
    popd
}

run_test() {
    pushd tests
    # 创建存储数据的文件夹
    if [ ! -d "data" ]; then
        mkdir "data"
    fi
    bash run.sh $@
    popd
}

case $1 in
    install)
        run_install
        ;;
    test)
        run_test $2
        ;;
    *)
        echo "Usage: $0 {install|test}"
        exit 1
        ;;
esac