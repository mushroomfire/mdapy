# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile
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

description = "A simple, fast and cross-platform python library to handle the data generated from molecular dynamics simulations."
try:
    readme = []
    with open("README.rst", encoding="utf-8") as f:
        for i in range(25):
            readme.append(f.readline())
    readme = "".join(readme)
except Exception:
    readme = description


ParallelCompile().install()
setup(
    name="mdapy",
    author="mushroomfire aka HerrWu",
    author_email="yongchao_wu@bit.edu.cn",
    description=description,
    long_description=readme,
    long_description_content_type="text/x-rst",
    packages=["mdapy"],
    ext_modules=[
        Pybind11Extension(
            "_cluster_analysis",
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
            "_nep",
            ["mdapy/nep/_nep.cpp", "thirdparty/nep/nep.cpp"],
            language="c++",
            include_dirs=["thirdparty/nep"],
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
        "taichi>=1.7.1",
        "numpy<2.0",
        "scipy",
        "polars>=0.20.26",
        "matplotlib",
        "polyscope",
        "tqdm",
    ],
    extras_require={
        "all": ["k3d", "pyfftw"],
        "k3d": "k3d",
        "pyfftw": "pyfftw",
    },
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
