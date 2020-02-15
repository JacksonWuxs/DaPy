from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules=cythonize('string_transfer.pyx', language_level=3))
