from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
    '_cluster_analysis',
    ['cluster.cpp', 'wrap.cpp'],
    language='c++'
    ),
]

setup(
    name='wrap',
    version='0.0.1',
    author='mdapy',
    ext_modules=ext_modules,
)