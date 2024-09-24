*mdapy* : Molecular Dynamics Analysis with Python
=====================================================


.. raw:: html

   <iframe src="https://mushroomfire.com/mdapy.html" height="170px" width="700px", frameBorder="0"></iframe>



Overview
--------

The **mdapy** python library provides an array of powerful, flexible, and straightforward 
tools to analyze atomic trajectories generated from Molecular Dynamics (MD) simulations. The library is fully 
cross-platform, making it accessible to users in **Windows, Linux, and Mac OS**. 
Benefited by the `TaiChi <https://github.com/taichi-dev/taichi>`_ project, 
we can effectively accelerate the pure python code, bringing it closer to the speed of code written in C++. 
Furthermore, **mdapy** is highly parallelized, allowing users to leverage the resources of both multicore CPU and GPU. 
**mdapy** can directly handle the DUMP and DATA formats in `LAMMPS <https://www.lammps.org/>`_, POSCAR format in `VASP <https://www.vasp.at/wiki/index.php/The_VASP_Manual>`_, universal XYZ format and CIF format. 
Besides, all data in **mdapy** is stored in NDARRAY format in `NumPy <https://numpy.org/>`_\ , which enables easy integration 
with the scientific ecosystem in python and facilitates collaboration with other post-progressing 
tools such as `OVITO <https://www.ovito.org/>`_ and `freud <https://github.com/glotzerlab/freud>`_.

Resources
----------

- Homepage: `https://github.com/mushroomfire/mdapy <https://github.com/mushroomfire/mdapy>`_
- Documentation: `https://mdapy.readthedocs.io/ <https://mdapy.readthedocs.io/>`_
- Issue Tracker: `https://github.com/mushroomfire/mdapy/issues <https://github.com/mushroomfire/mdapy/issues>`_
- Tutorial: `https://github.com/mushroomfire/mdapy-tutorial <https://github.com/mushroomfire/mdapy-tutorial>`_

Dependencies
------------

* `python <https://www.python.org/>`_ (3.8-3.11)
* `taichi>=1.7.1 <https://github.com/taichi-dev/taichi>`_
* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/>`_
* `polars>=1.8.0 <https://pola-rs.github.io/polars/>`_
* `matplotlib <https://matplotlib.org/>`_
* `polyscope <https://polyscope.run/py/>`_
* `tqdm <https://github.com/tqdm/tqdm>`_ (Show progress bar when reading/saving multi files)

Optional Dependencies
----------------------

* `k3d <https://github.com/K3D-tools/K3D-jupyter>`_ (Visualizing the 3D atoms)
* `pyfftw <https://github.com/pyFFTW/pyFFTW>`_ (Fast FFT)
* `phonopy <https://github.com/phonopy/phonopy>`_ (Do phonon calculation)
* `lammps <https://github.com/lammps/lammps>`_ (Compute atomic energy, force and virial using lammps supported potential)

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
   gettingstarted/calculate_GSFE
   gettingstarted/calculate_phonon
   gettingstarted/visualizing_atoms
   gettingstarted/use_mdapy_efficiently
   gettingstarted/matters_need_attention
   gettingstarted/reading_and_saving_file
   gettingstarted/benchmark
   gettingstarted/main_features

.. toctree::
   :maxdepth: 1
   :caption: Mdapy Python API

   modules
