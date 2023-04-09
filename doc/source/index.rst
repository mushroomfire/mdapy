*mdapy* : Molecular Dynamics Analysis with Python
=====================================================

Overview
--------

The **mdapy** python library is developed by the **Yong-Chao Wu & Jian-Li Shao Group at the 
Beijing Institute of Technology**, which provides an array of powerful, flexible, and straightforward 
tools to analyze atomic trajectories generated from Molecular Dynamics (MD) simulations. The library is fully 
cross-platform, making it accessible to users in **Windows, Linux, and Mac OS**. 
Benefited by the `TaiChi <https://github.com/taichi-dev/taichi>`_ project, 
we can effectively accelerate the pure python code, bringing it closer to the speed of code written in C++. 
Furthermore, **mdapy** is highly parallelized, allowing users to leverage the resources of both multicore CPU and GPU. 
**mdapy** can directly handle the DUMP and DATA formats in `LAMMPS <https://www.lammps.org/>`_. 
Besides, all data in **mdapy** is stored in NDARRAY format in `NumPy <https://numpy.org/>`_\ , which enables easy integration 
with the scientific ecosystem in python and facilitates collaboration with other post-progressing 
tools such as `OVITO <https://www.ovito.org/>`_ and `freud <https://github.com/glotzerlab/freud>`_.

Resources
----------

- Homepage: `https://github.com/mushroomfire/mdapy <https://github.com/mushroomfire/mdapy>`_
- Documentation: `https://mdapy.readthedocs.io/ <https://mdapy.readthedocs.io/>`_
- Issue Tracker: `https://github.com/mushroomfire/mdapy/issues <https://github.com/mushroomfire/mdapy/issues>`_

Dependencies
------------

* `python <https://www.python.org/>`_ (3.7-3.10)
* `taichi>=1.4.0 <https://github.com/taichi-dev/taichi>`_
* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/>`_
* `pandas <https://pandas.pydata.org/>`_
* `pyarrow <https://arrow.apache.org/docs/python/index.html>`_
* `matplotlib <https://matplotlib.org/>`_

Optional Dependencies
----------------------

* `SciencePlots <https://github.com/garrettj403/SciencePlots>`_ (Optional, for plotting results)
* `tqdm <https://github.com/tqdm/tqdm>`_ (Optional, for progress bar when reading/saving multi DUMP files)
* `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ (Optional, for fast FFT)
* `pyfnntw>=0.4.1 <https://github.com/cavemanloverboy/FNNTW>`_ (Optional, for fast KNN search)

.. toctree::
   :maxdepth: 3
   :caption: Getting Started
   
   gettingstarted/installation
   gettingstarted/quicklystart
   gettingstarted/examples
   gettingstarted/use_mdapy_efficiently
   gettingstarted/reading_and_saving_file
   gettingstarted/benchmark
   gettingstarted/main_features

.. toctree::
   :maxdepth: 1
   :caption: Mdapy Python API

   modules
