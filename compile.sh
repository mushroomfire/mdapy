# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.
# To compile the C module for different python versions.
#!/usr/bin/env bash

if [ "$(uname)" == "Darwin" ]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then   
    source /home/herrwu/deepmd-kit/etc/profile.d/conda.sh
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then    
    source D:/Anaconda/etc/profile.d/conda.sh
fi

for i in 7 8 9 10 11
do 
    echo "py3${i}"
    conda activate "py3${i}"
    python -m pip install --upgrade pip setuptools wheel twine
    pip install pybind11
    if [ "$(uname)" == "Darwin" ]; then
        echo "Mac OS X"
        python setup.py bdist_wheel
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then   
        echo "GNU/Linux"
        python setup.py bdist_wheel --plat-name=manylinux1_x86_64
    elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then    
        echo "Windows NT"
        python setup.py bdist_wheel
    fi
    conda deactivate
done

# python -m twine upload dist/*


