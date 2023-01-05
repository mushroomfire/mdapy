# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import os
import shutil

# os.system("python setup.py clean")
# print(os.getcwd())
os.system("python ./setup.py build_ext -i")
shutil.rmtree("build")
