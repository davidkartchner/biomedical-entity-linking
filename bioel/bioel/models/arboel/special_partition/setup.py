from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "special_partition",
        ["special_partition.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(ext_modules=cythonize(extensions))

"""
Build instructions: 
------------------
> cd special_partition
> python setup.py build_ext --inplace
"""
