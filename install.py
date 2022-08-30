#该脚本会自动复制该模块到环境目录，并自动检测依赖库！
#目前只支持python 3.7

import sys
import os
import shutil

py_version = f'python{sys.version.split()[0][:3]}'

for i in sys.path:
    if os.path.split(os.path.split(i)[0])[1] in ['lib', 'Lib', py_version] and os.path.split(i)[1] == 'site-packages':
        print(f'Installing package to path: {i}')
        if os.path.exists(i+'/pyAnalysis'):
            shutil.rmtree(i+'/pyAnalysis')
        shutil.copytree('../pyAnalysis', i+'/pyAnalysis')
        os.chdir(i+'/pyAnalysis')
        print('Checking dependency package...')
        os.system('python check.py')
        break
        
