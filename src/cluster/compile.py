import os

os.system("gfortran -c deque.f90")
os.system("f2py -c deque.o ClusterAnalysisCompute.f90 -m ClusterAnalysisComputeF --opt='-O3' --fcompiler=gnu95 --compiler=mingw32")