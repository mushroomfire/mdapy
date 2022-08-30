import os
from glob import glob
import platform
from shutil import move


for i in glob('*/*.f90'):
    path, file = os.path.split(i)
    mod = os.path.splitext(file)[0]
    os.system(f'cd {path}')
    if platform.system() == 'Linux':
        if mod[-1] == 'p':
            os.system(f"f2py -c {os.path.abspath(i)} -m {mod} --opt='-O3' --fcompiler=intelem --f90flags='-qopenmp' -lgomp")
            #os.system(f"f2py -c {os.path.abspath(i)} -m {mod} --opt='-O3' --fcompiler=gfortran --f90flags='-fopenmp' -lgomp")
        elif mod[-1] == 's':
            os.system(f"f2py -c {os.path.abspath(i)} -m {mod} --opt='-O3' --fcompiler=intelem")
            #os.system(f"f2py -c {os.path.abspath(i)} -m {mod} --opt='-O3' --fcompiler=gfortran")
    elif platform.system() == 'Windows':
        if mod[-1] == 'p':
            os.system(f"f2py -c {os.path.abspath(i)} -m {mod} --opt='-O3' --fcompiler=gfortran --f90flags='-fopenmp' -lgomp")
        elif mod[-1] == 's':
            os.system(f"f2py -c {os.path.abspath(i)} -m {mod} --opt='-O3' --fcompiler=gfortran")
    os.system('cd ..')
    if platform.system() == 'Linux':
        src_file = f"{glob(f'./{mod}.*.so')[0]}"
    elif platform.system() == 'Windows':
        src_file = f"{glob(f'./{mod}.*.pyd')[0]}"
    dst_file = os.path.join(path, src_file)
    if os.path.exists(dst_file):
        try:
            os.remove(dst_file)
        except PermissionError as exc:
            os.chmod(dst_file, stat.S_IWUSR)
            os.remove(dst_file)
    move(src_file, dst_file)
        
