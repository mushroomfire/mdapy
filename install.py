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
        print(f"Installing package to path: {i}")
        path = i + "/pyAnalysis"
        if os.path.exists(path):
            shutil.rmtree(path, onerror=readonly_handler)
        shutil.copytree("../pyAnalysis", path)
        os.chdir(i + "/pyAnalysis")
        print("Checking dependency package...")
        os.system("python check.py")
        break
