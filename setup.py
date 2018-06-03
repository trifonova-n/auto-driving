from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    ext_modules = cythonize("auto_driving/_mask.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)
