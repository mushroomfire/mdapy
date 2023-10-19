# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from glob import glob
import sys

if sys.platform.startswith("win"):
    extra_compile_args = ["/openmp:llvm", "/d2FH4-"]
    extra_link_args = ["/openmp:llvm"]
elif sys.platform.startswith("linux"):
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]
elif sys.platform.startswith("darw"):
    extra_compile_args = ["-Xclang", "-fopenmp"]
    extra_link_args = ["-lomp"]


description = "A simple, fast and cross-platform python library to handle the data generated from molecular dynamics simulations"
try:
    with open("README.rst") as f:
        readme = f.read()
except Exception:
    readme = description

setup(
    name="mdapy",
    version="0.9.3",
    author="mushroomfire aka HerrWu",
    author_email="yongchao_wu@bit.edu.cn",
    description=description,
    long_description=readme,
    packages=["mdapy"],
    ext_modules=[
        Pybind11Extension(
            r"_cluster_analysis",
            ["mdapy/cluster/_cluster.cpp"],
            language="c++",
        ),
        Pybind11Extension(
            "_voronoi_analysis",
            [
                "mdapy/voronoi/_voro.cpp",
                "thirdparty/voro++/voro++.cc",
            ],
            language="c++",
            include_dirs=["thirdparty/voro++"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
        Pybind11Extension(
            "_ptm",
            glob("thirdparty/ptm/ptm*.cpp") + ["mdapy/ptm/_ptm.cpp"],
            language="c++",
            include_dirs=["thirdparty/ptm"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
        Pybind11Extension(
            "_rdf",
            ["mdapy/rdf/_rdf.cpp"],
            language="c++",
        ),
        Pybind11Extension(
            "_neigh",
            ["mdapy/neigh/_neigh.cpp"],
            language="c++",
        ),
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    license="BSD 3-Clause License",
    url="https://github.com/mushroomfire/mdapy",
    python_requires=">=3.8,<3.12",
    install_requires=[
        "taichi>=1.6.0",
        "numpy",
        "scipy",
        "pandas",
        "polars>=0.19",
        "pyarrow",
        "matplotlib",
    ],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    project_urls={
        "Homepage": "https://github.com/mushroomfire/mdapy",
        "Documentation": "https://mdapy.readthedocs.io/",
        "Issue Tracker": "https://github.com/mushroomfire/mdapy/issues",
    },
)
