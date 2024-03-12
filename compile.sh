# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.
# To compile the C module for different python versions.
#!/usr/bin/env bash

version="0.10.4"

if [ "$(uname)" == "Darwin" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then   
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then    
    source D:/Software/miniconda3/etc/profile.d/conda.sh
fi

for i in 8 9 10 11
do 
    echo "py3${i}"
    conda activate "py3${i}"
    python -m pip install --upgrade pip setuptools wheel twine
    pip install pybind11 build
    if [ "$(uname)" == "Darwin" ]; then
        echo "Mac OS X"
        python setup.py bdist_wheel
        pip install delocate
        name="dist/mdapy-${version}-cp3${i}-cp3${i}-macosx_11_0_arm64.whl"
        delocate-wheel -w wheelhouse -v ${name}
        # python -m build --no-isolation This has bugs in Mac OS.
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then   
        echo "GNU/Linux"
        python -m build --no-isolation
        pip install auditwheel
        name="dist/mdapy-${version}-cp3${i}-cp3${i}-linux_x86_64.whl"
        auditwheel repair ${name} --plat=manylinux_2_35_x86_64
        oldname="wheelhouse/mdapy-${version}-cp3${i}-cp3${i}-manylinux_2_35_x86_64.whl"
        newname="wheelhouse/mdapy-${version}-cp3${i}-cp3${i}-manylinux1_x86_64.whl"
        cp ${name} ${newname}

    elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then    
        echo "Windows NT"
        python -m build --no-isolation
        pip install delvewheel
        name="dist/mdapy-${version}-cp3${i}-cp3${i}-win_amd64.whl"
        delvewheel repair ${name}
    fi
    conda deactivate
done

# python -m twine upload wheelhouse/*
# rename 's/linux/manylinux1/' mdapy-0.9.1*linux* -n


