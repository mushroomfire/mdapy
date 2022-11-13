import os
import shutil

# os.system("python setup.py clean")
os.system("python setup.py build_ext -i")
shutil.rmtree("build")
