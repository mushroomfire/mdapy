*mdapy* : Molecular Dynamics Analysis with Python
=====================================================


.. raw:: html

   <iframe src="https://mushroomfire.com/mdapy.html" height="200px" width="1000px", frameBorder="0"></iframe>



Overview
--------

The **mdapy** python library provides an array of powerful, flexible, and straightforward 
tools to analyze atomic trajectories generated from Molecular Dynamics (MD) simulations. The library is fully 
cross-platform, making it accessible to users in **Windows, Linux, and Mac OS**. 
Furthermore, **mdapy** is highly parallelized, allowing users to leverage the resources of multicore CPU. 
**mdapy** can directly handle the DUMP and DATA formats in `LAMMPS <https://www.lammps.org/>`_, POSCAR format in `VASP <https://www.vasp.at/wiki/index.php/The_VASP_Manual>`_, and universal XYZ format. 

Resources
----------

- Homepage: `https://github.com/mushroomfire/mdapy <https://github.com/mushroomfire/mdapy>`_
- Documentation: `https://mdapy.readthedocs.io/ <https://mdapy.readthedocs.io/>`_
- Issue Tracker: `https://github.com/mushroomfire/mdapy/issues <https://github.com/mushroomfire/mdapy/issues>`_
- Tutorial: `https://github.com/mushroomfire/mdapy-tutorial <https://github.com/mushroomfire/mdapy-tutorial>`_

Dependencies
------------

* `numpy <https://numpy.org/>`_
* `polars <https://pola-rs.github.io/polars/>`_

Citation
---------
If you find **mdapy** useful, you can `star it! <https://github.com/mushroomfire/mdapy>`_
If you use **mdapy** in your scientific publications, please `cite the paper: <https://doi.org/10.1016/j.cpc.2023.108764>`_

.. code-block:: bibtex

   @article{mdapy023,
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
   
   gettingstarted/use_mdapy_efficiently

Mdapy Python API
-----------------
.. toctree::
   :maxdepth: 2

   source/mdapy



