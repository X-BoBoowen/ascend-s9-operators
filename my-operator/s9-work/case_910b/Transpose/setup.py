import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
TRANSPOSE_FAST_LIB_DIR = os.environ.get(
    "TRANSPOSE_FAST_LIB_DIR",
    "/home/ma-user/work/s9/transpose_fast_project_v1/build_out/op_api/lib",
)

exts = []
ext1 = NpuExtension(
    name="custom_ops_lib",
    # 如果还有其他cpp文件参与编译，需要在这里添加
    sources=["./extension/custom_op.cpp"],
    extra_compile_args = [
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    ],
    library_dirs=[TRANSPOSE_FAST_LIB_DIR],
    libraries=["transpose_fast_opapi"],
    extra_link_args=["-Wl,-rpath," + TRANSPOSE_FAST_LIB_DIR],
)
exts.append(ext1)

setup(
    name="custom_ops",
    version='1.0',
    keywords='custom_ops',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension},
)
