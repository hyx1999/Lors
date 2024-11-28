import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    'cxx': ['-g'],
    'nvcc': [
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_86,code=sm_86',
    ]
}

# 设置包含路径
prefix = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(prefix, "include")


setup(
    name='lors_kernels',
    ext_modules=[
        CUDAExtension('lors_kernels', [
            'src/bindings.cu',
            'src/sparsify_nm.cu'
        ], include_dirs=[include_dir], extra_compile_args=extra_compile_args),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
