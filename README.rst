.. image:: doc/images/logo.png

*mdapy* : Molecular Dynamics Analysis with Python
=====================================================

Overview
---------

The **mdapy** python library provides an array of powerful, flexible, and straightforward 
tools to analyze atomic trajectories generated from Molecular Dynamics (MD) simulations. The library is fully 
cross-platform, making it accessible to users in **Windows, Linux, and Mac OS**. 
Furthermore, **mdapy** is highly parallelized, allowing users to leverage the resources of multicore CPU. 
**mdapy** can directly handle the DUMP and DATA formats in `LAMMPS <https://www.lammps.org/>`_, POSCAR format in `VASP <https://www.vasp.at/wiki/index.php/The_VASP_Manual>`_, 
universal XYZ format and self-difined MP format. Besides, all data in **mdapy** is stored in NDARRAY format in `NumPy <https://numpy.org/>`_, which enables easy integration 
with the scientific ecosystem in python and facilitates collaboration with other post-progressing 
tools such as `OVITO <https://www.ovito.org/>`_ and `freud <https://github.com/glotzerlab/freud>`_.

Resources
----------

- Homepage: `https://github.com/mushroomfire/mdapy <https://github.com/mushroomfire/mdapy>`_
- Documentation: `https://mdapy.readthedocs.io/ <https://mdapy.readthedocs.io/>`_
- Issue Tracker: `https://github.com/mushroomfire/mdapy/issues <https://github.com/mushroomfire/mdapy/issues>`_
- Tutorial: `https://github.com/mushroomfire/mdapy-tutorial <https://github.com/mushroomfire/mdapy-tutorial>`_

Notes
------

We experienced a break upgrade from mdapy 1.0, the old mdapy (<=0.11.5) can be found in this `branch <https://github.com/mushroomfire/mdapy/tree/mdapy_old>`_

Dependencies
-------------

.. list-table::

    *   -   **Package Name**
        -   **Version**
        -   **Features**
    *   -   `python <https://www.python.org>`_
        -   >=3.8
        -   Language support.
    *   -   `numpy <https://numpy.org/>`_
        -   latest
        -   Data structure.
    *   -   `polars <https://pola-rs.github.io/polars/>`_
        -   latest
        -   Fast read/save file. Data structure to represent the particles information.


Optional Dependencies
----------------------

.. list-table::

    *   -   **Package Name**
        -   **Version**
        -   **Features**
    *   -   `k3d <https://github.com/K3D-tools/K3D-jupyter>`_
        -   latest
        -   Visualize the 3D atoms in Jupyter.
    *   -   `matplotlib <https://matplotlib.org/>`_
        -   latest
        -   Plot the results.
    *   -   `pyfftw <https://github.com/pyFFTW/pyFFTW>`_
        -   latest
        -   Faster FFT.
    *   -   `phonopy <https://github.com/phonopy/phonopy>`_
        -   latest
        -   Do phonon calculation.
    *   -   `lammps <https://github.com/lammps/lammps>`_ 
        -   latest
        -   Compute atomic energy, force and virial using lammps supported potential.

Installation
-------------

Install from pip (recommended).
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install mdapy

Install from source code.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- You should have a C++ compilation environment (-std=c++17 or newer) and openmp supports. 
  Tested by MSVC in Windows 10, GCC in Ubuntu, Clang in MAC OS M1.

- Download the source code and installation.
   
   .. code-block:: bash

      git clone https://github.com/mushroomfire/mdapy.git
      cd mdapy 
      pip install .

- Directly install from github:
  
  .. code-block:: bash 

    pip install https://github.com/mushroomfire/mdapy/archive/master.zip

Check Installation
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -c "import mdapy as mp; print('mdapy version is:', mp.__version__)"

Build the doc
--------------

If you want to build documentation locally, you can install the dependencies:

.. code-block:: bash

   pip install Sphinx nbsphinx pydata_sphinx_theme
   conda install pandoc

Then changing to ./doc dir:

.. code-block:: bash

   make html


Citation
---------
If you find **mdapy** useful, you can `star it! <https://github.com/mushroomfire/mdapy>`_
If you use **mdapy** in your scientific publications, please `cite the paper: <https://doi.org/10.1016/j.cpc.2023.108764>`_

.. code-block:: bibtex

   @article{mdapy2023,
      title = {mdapy: A flexible and efficient analysis software for molecular dynamics simulations},
      journal = {Computer Physics Communications},
      pages = {108764},
      year = {2023},
      issn = {0010-4655},
      doi = {https://doi.org/10.1016/j.cpc.2023.108764},
      url = {https://www.sciencedirect.com/science/article/pii/S0010465523001091},
      author = {Yong-Chao Wu and Jian-Li Shao},
      keywords = {Simulation analysis, Molecular dynamics, Polycrystal, TaiChi, Parallel computing}
      }