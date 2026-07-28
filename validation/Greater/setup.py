import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension
import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension


PYTORCH_NPU_INSTALL_PATH = os.path.dirname(
    os.path.abspath(torch_npu.__file__)
)
CUSTOM_OP_LIB_DIR = os.environ["GREATER_CUSTOM_LIB_DIR"]

extension = NpuExtension(
    name="greater_validation_lib",
    sources=["./extension/custom_op.cpp"],
    extra_compile_args=[
        "-I"
        + os.path.join(
            PYTORCH_NPU_INSTALL_PATH,
            "include/third_party/acl/inc",
        ),
    ],
    library_dirs=[CUSTOM_OP_LIB_DIR],
    libraries=["cust_opapi"],
    extra_link_args=["-Wl,-rpath," + CUSTOM_OP_LIB_DIR],
)

setup(
    name="greater_validation",
    version="1.0",
    ext_modules=[extension],
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension},
)
