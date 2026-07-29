import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension
import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension


PYTORCH_NPU_INSTALL_PATH = os.path.dirname(
    os.path.abspath(torch_npu.__file__)
)

extension = NpuExtension(
    name="square_sum_v1_validation_lib",
    sources=["./extension/custom_op.cpp"],
    extra_compile_args=[
        "-I"
        + os.path.join(
            PYTORCH_NPU_INSTALL_PATH,
            "include/third_party/acl/inc",
        ),
    ],
)

setup(
    name="square_sum_v1_validation",
    version="1.0",
    ext_modules=[extension],
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension},
)
