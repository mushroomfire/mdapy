*mdapy* : Molecular Dynamics Analysis with Python
=====================================================

Overview
--------

The **mdapy** is a python library providing a set of simple, flexible and powerful tools to analyze the atomic trajectories generated from Molecular Dynamics (MD) simulations. Benefit by the `TaiChi <https://github.com/taichi-dev/taichi>`_ project, we can effectively accelerate the pure python code close to those written in C++. Moreover, **mdapy** is highly parallelized to make full use of resources of both multicore CPU and GPU. **mdapy** can directly handle the DUMP and DATA format in `LAMMPS <https://www.lammps.org/>`_. All data in **mdapy** is stored in NDARRAY format in `Numpy <https://numpy.org/>`_\ , enabling integration with the scientific ecosystem in python and corporation with other post-progressing codes, such as `OVITO <https://www.ovito.org/>`_ and `freud <https://github.com/glotzerlab/freud>`_. 

Dependencies
------------

* python (3.7-3.10)
* taichi==1.2.0
* numpy
* scipy
* pandas
* tqdm
* matplotlib
* SciencePlots
* pyfftw (Optional, for fast FFT)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   gettingstarted/installation
   gettingstarted/examples

.. toctree::
   :maxdepth: 1
   :caption: Mdapy Python API

   modules
