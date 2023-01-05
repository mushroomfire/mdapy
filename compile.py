import os

os.chdir("./mdapy/cluster/")
os.system("python compile.py")
os.chdir("../polygon/")
os.system("python compile.py")
os.chdir("../voronoi/")
os.system("python compile.py")
