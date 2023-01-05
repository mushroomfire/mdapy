from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="mdapy",
    version="0.7.4",
    author="mushroomfire aka HerrWu",
    description="A simple and fast python library to handle the data generated from molecular dynamics simulations",
    long_description=readme,
    packages=["mdapy"],
    headers=[
        "mdapy/cluster/cluster.hpp",
        "mdapy/polygon/polygon.hpp",
        "mdapy/voronoi/voro.hpp",
        "thirdparty/voro++/voro++.hh",
    ],
    ext_modules=[
        Pybind11Extension(
            r"_cluster_analysis",
            ["mdapy/cluster/cluster.cpp", "mdapy/cluster/wrap.cpp"],
            language="c++",
        ),
        Pybind11Extension(
            r"poly",
            [
                "mdapy/polygon/polygon.cpp",
                "mdapy/polygon/wrap.cpp",
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
    ],
    # add custom build_ext command
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    url="https://github.com/mushroomfire/mdapy",
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "taichi==1.2.0",
        "tqdm",
        "matplotlib",
        "SciencePlots",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
