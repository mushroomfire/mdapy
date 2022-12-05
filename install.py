# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import sys
import os
import shutil
import stat

py_version = f"python{sys.version.split()[0][:3]}"


def readonly_handler(func, path, execinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


for i in sys.path:
    if (
        os.path.split(os.path.split(i)[0])[1] in ["lib", "Lib", py_version]
        and os.path.split(i)[1] == "site-packages"
    ):
        print("-" * 50)
        print(f"Installing [mdapy] package to path: {i}")
        print("-" * 50)
        path = i + "/mdapy"
        if os.path.exists(path):
            shutil.rmtree(path, onerror=readonly_handler)
        shutil.copytree("../mdapy", path)
        os.chdir(i + "/mdapy")
        print("Checking dependency package...")
        os.system("python check.py")
        break
