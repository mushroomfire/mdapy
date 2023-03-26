# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from glob import glob
import sys

if sys.platform.startswith("win"):
    omp = "/openmp"
elif sys.platform.startswith("linux"):
    omp = "-fopenmp"

description = "A simple and fast python library to handle the data generated from molecular dynamics simulations"
try:
    with open("README.rst") as f:
        readme = f.read()
except Exception:
    readme = description

setup(
    name="mdapy",
    version="0.8.4",
    author="mushroomfire aka HerrWu",
    author_email="yongchao_wu@bit.edu.cn",
    description=description,
    long_description=readme,
    packages=["mdapy"],
    ext_modules=[
        Pybind11Extension(
            r"_cluster_analysis",
            ["mdapy/cluster/cluster.cpp", "mdapy/cluster/wrap.cpp"],
            language="c++",
        ),
        Pybind11Extension(
            r"_poly",
            [
                "mdapy/polygon/polygon.cpp",
                "thirdparty/voro++/voro++.cc",
            ],
            language="c++",
            include_dirs=["thirdparty/voro++"],
        ),
        Pybind11Extension(
            "_voronoi_analysis",
            [
                "mdapy/voronoi/voro.cpp",
                "mdapy/voronoi/wrap.cpp",
                "thirdparty/voro++/voro++.cc",
            ],
            language="c++",
            include_dirs=["thirdparty/voro++"],
        ),
        Pybind11Extension(
            "_ptm",
            glob("thirdparty/ptm/ptm*.cpp") + ["mdapy/ptm/_ptm.cpp"],
            language="c++",
            include_dirs=["thirdparty/ptm"],
            extra_compile_args=[omp],
            extra_link_args=[omp],
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
    zip_safe=False,
    url="https://github.com/mushroomfire/mdapy",
    python_requires=">=3.7,<3.11",
    install_requires=[
        "taichi>=1.4.0",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "tqdm",
    ],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    project_urls={
        "Homepage": "https://github.com/mushroomfire/mdapy",
        "Documentation": "https://mdapy.readthedocs.io/",
        "Source Code": "https://github.com/mushroomfire/mdapy",
        "Issue Tracker": "https://github.com/mushroomfire/mdapy/issues",
    },
)
