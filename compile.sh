# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.
# To compile the C module for different python versions.

#!/usr/bin/env bash

for i in 7 8 9 10
do 
    echo "py3${i}"
    source activate "py3${i}"
    python -m pip install --upgrade pip setuptools wheel twine
    pip install pybind11
    if [ "$(uname)" == "Darwin" ]; then
        echo "Mac OS X"
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then   
        echo "GNU/Linux"
        python setup.py bdist_wheel --plat-name=manylinux1_x84_x64
    elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then    
        echo "Windows NT"
        python setup.py bdist_wheel
    fi
    source deactivate
done

# python -m twine upload dist/*


