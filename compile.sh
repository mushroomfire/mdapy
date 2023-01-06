# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.
# To compile the C module for different python versions.

for i in 7 8 9 10
do 
    echo "py3${i}"
    source activate "py3${i}"
    python -m pip install --upgrade pip setuptools wheel twine
    pip install pybind11
    python setup.py bdist_wheel
    source deactivate
done

# python -m twine upload dist/*


