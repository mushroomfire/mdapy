# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

setup(
    name="wrap",
    version="0.0.1",
    author="mdapy",
    ext_modules=[
        Pybind11Extension(
            "_nep",
            ["_nep.cpp", "../../thirdparty/nep/nep.cpp"],
            language="c++",
            include_dirs=["../../thirdparty/nep"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
