*mdapy* : Molecular Dynamics Analysis with Python
=====================================================

Overview
--------

The **mdapy** python library provides an array of powerful, flexible, and straightforward 
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

* `python <https://www.python.org/>`_ (3.8-3.11)
* `taichi>=1.6.0 <https://github.com/taichi-dev/taichi>`_
* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/>`_
* `pandas <https://pandas.pydata.org/>`_
* `polars>=0.19.0 <https://pola-rs.github.io/polars/>`_
* `pyarrow <https://arrow.apache.org/docs/python/>`_
* `matplotlib <https://matplotlib.org/>`_

Optional Dependencies
----------------------

* `SciencePlots <https://github.com/garrettj403/SciencePlots>`_ (Optional, for plotting results)
* `tqdm <https://github.com/tqdm/tqdm>`_ (Optional, for progress bar when reading/saving multi DUMP files)
* `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ (Optional, for fast FFT)
* `pyfnntw>=0.4.1 <https://github.com/cavemanloverboy/FNNTW>`_ (Optional, for fast KNN search)

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
