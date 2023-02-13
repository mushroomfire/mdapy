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
stored in NDARRAY format in `NumPy <https://numpy.org/>`_\ , enabling integration 
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
* `taichi>=1.4.0 <https://github.com/taichi-dev/taichi>`_
* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/>`_
* `pandas <https://pandas.pydata.org/>`_
* `tqdm <https://github.com/tqdm/tqdm>`_
* `matplotlib <https://matplotlib.org/>`_

Optional Dependencies
----------------------

* `SciencePlots <https://github.com/garrettj403/SciencePlots>`_ (Optional, for plotting results)
* `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ (Optional, for fast FFT)

.. toctree::
   :maxdepth: 3
   :caption: Getting Started
   
   gettingstarted/installation
   gettingstarted/quicklystart
   gettingstarted/examples
   gettingstarted/benchmark

.. toctree::
   :maxdepth: 1
   :caption: Mdapy Python API

   modules
