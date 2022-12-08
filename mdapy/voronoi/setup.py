# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "_voronoi_analysis",
        ["voro.cpp", "wrap.cpp", "../../thirdparty/voro++/voro++.cc"],
        language="c++",
        include_dirs=["../../thirdparty/voro++"],
    ),
]

setup(
    name="wrap",
    version="0.0.1",
    author="mdapy",
    ext_modules=ext_modules,
)
