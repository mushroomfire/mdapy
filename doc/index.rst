===================================================
🚀 mdapy - Molecular Dynamics Analysis with Python
===================================================

Overview
--------

**mdapy** is a fast, full-featured Python library for analyzing Molecular
Dynamics (MD) simulation data. It combines high-performance C++ kernels,
a lightweight Python interface, built-in ray-tracing visualization, and
machine-learning potential workflows in one package.

.. code-block:: bash

   pip install mdapy

Why mdapy?
----------

- **Fast core**: Nanobind-wrapped C++ kernels with OpenMP acceleration.
- **Lightweight**: the core package depends only on NumPy and Polars.
- **Practical**: structure analysis, model building, rendering, and NEP workflows in one API.
- **Cross-platform**: wheels for Windows, Linux, and macOS.

Key Capabilities
----------------

- Neighbor search: fixed-radius neighbors, kNN, and Voronoi neighbors.
- Structural analysis: PTM, CNA, CSP, IDS, SRO, RDF, ADF, structure factor, bond analysis, and more.
- Connectivity: build bond pairs directly from neighbor lists with a global cutoff or type-/element-specific cutoffs.
- Model building: FCC/BCC/HCP/diamond crystals, HEAs, and polycrystals.
- Rendering: Tachyon CPU/GPU rendering with configurable colors, radii, shadows, and transparent backgrounds.
- Machine-learning workflows: NEP/qNEP evaluation, elastic constants, EOS, stacking faults, and phonons.

Quick Links
-----------

- `GitHub repository <https://github.com/mushroomfire/mdapy>`_
- `PyPI package <https://pypi.org/project/mdapy/>`_
- `Issue tracker <https://github.com/mushroomfire/mdapy/issues>`_
- `Paper (Computer Physics Communications) <https://doi.org/10.1016/j.cpc.2023.108764>`_

Dependencies
------------

- `numpy <https://numpy.org/>`_
- `polars <https://pola.rs/>`_

Citation
--------

If you use **mdapy** in research, please cite:

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

Documentation
-------------

Use the sections below to jump directly to installation, tutorials, release
notes, or the Python API reference.

Release Notes
--------------
.. toctree::
   :maxdepth: 2

   releasenotes

Getting Started
-----------------
.. toctree::
   :maxdepth: 2
   
   gettingstarted/installation
   gettingstarted/atomic_structure_generation
   gettingstarted/use_mdapy_efficiently
   gettingstarted/elastic_constant
   gettingstarted/phonon_calculation
   gettingstarted/eam_analysis
   gettingstarted/load_save
   gettingstarted/render_structure
   gettingstarted/benchmark
   

Mdapy Python API
-----------------
.. toctree::
   :maxdepth: 2

   source/mdapy



