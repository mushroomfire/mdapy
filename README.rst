.. image:: https://s3.pterclub.com/images/2023/01/06/logo.png

*mdapy* : Molecular Dynamics Analysis with Python
=====================================================

Overview
--------

The **mdapy** is a python library providing a set of simple, 
flexible and powerful tools to analyze the atomic trajectories 
generated from Molecular Dynamics (MD) simulations. 
Benefit by the `TaiChi <https://github.com/taichi-dev/taichi>`_ project, 
we can effectively accelerate the pure python code close to those written 
in C++. Moreover, **mdapy** is highly parallelized to make full use of 
resources of both multicore CPU and GPU. **mdapy** can directly handle the DUMP 
and DATA format in `LAMMPS <https://www.lammps.org/>`_. All data in **mdapy** is 
stored in NDARRAY format in `Numpy <https://numpy.org/>`_\ , enabling integration 
with the scientific ecosystem in python and corporation with other post-progressing 
codes, such as `OVITO <https://www.ovito.org/>`_ and `freud <https://github.com/glotzerlab/freud>`_. 

Resources
----------

- Homepage: `https://github.com/mushroomfire/mdapy <https://github.com/mushroomfire/mdapy>`_
- Documentation: `https://mdapy.readthedocs.io/ <https://mdapy.readthedocs.io/>`_
- Source Code: `https://github.com/mushroomfire/mdapy <https://github.com/mushroomfire/mdapy>`_
- Issue Tracker: `https://github.com/mushroomfire/mdapy/issues <https://github.com/mushroomfire/mdapy/issues>`_

Dependencies
------------

* `python <https://www.python.org/>`_ (3.7-3.10)
* `taichi==1.2.0 <https://github.com/taichi-dev/taichi>`_
* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/>`_
* `pandas <https://pandas.pydata.org/>`_
* `tqdm <https://github.com/tqdm/tqdm>`_
* `matplotlib <https://matplotlib.org/>`_
* `SciencePlots <https://github.com/garrettj403/SciencePlots>`_
* `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ (Optional, for fast FFT)

Installation
-------------

Install from pip (recommended).
--------------------------------------

.. code-block:: bash

   pip install mdapy

Install from source code.
---------------------------

.. note:: You should install pybind11 and have a c++ compilation environment.

   .. code-block:: bash
      
      pip install pybind11

- Download source code
   
   .. code-block:: bash

      git clone https://github.com/mushroomfire/mdapy.git
      cd mdapy 
      python setup.py install

Usage
------

.. code-block:: python

   import mdapy as mp
   mp.init('cpu') # use cpu, mp.init('gpu') will use gpu to compute.

   system = mp.System('./CoCuFeNiPd-4M.dump') # read dump file to generate a system class
   system.cal_centro_symmetry_parameter() # calculate the centrosymmetry parameters
   system.cal_atomic_entropy() # calculate the atomic entropy
   system.write_dump() # save results to a new dump file
