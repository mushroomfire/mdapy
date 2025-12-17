===================================================
🚀 mdapy - Molecular Dynamics Analysis with Python
===================================================

.. image:: https://img.shields.io/pypi/v/mdapy.svg
   :target: https://pypi.org/project/mdapy/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/mdapy.svg
   :target: https://pypi.org/project/mdapy/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/mushroomfire/mdapy.svg
   :target: https://github.com/mushroomfire/mdapy/blob/master/LICENSE
   :alt: License

.. image:: https://readthedocs.org/projects/mdapy/badge/?version=latest
   :target: https://mdapy.readthedocs.io/en/latest/?badge=latest
   :alt: ReadTheDocs

.. image:: https://img.shields.io/github/stars/mushroomfire/mdapy.svg?style=social
   :target: https://github.com/mushroomfire/mdapy
   :alt: GitHub stars

A simple, fast, and powerful Python library designed for Molecular Dynamics (MD) simulation data analysis! 💪

--------

📖 **Introduction**
========================================

**mdapy** provides a comprehensive suite of powerful, 
flexible, and straightforward tools for analyzing atomic 
trajectories generated from Molecular Dynamics (MD) simulations.

🎯 **Why Choose mdapy?**
--------

✨ **Cross-Platform Support**
   Fully compatible with Windows, Linux, and Mac OS - run anywhere seamlessly

⚡ **Blazing Fast Performance**
   - Accelerated by C++ using Nanobind
   - Highly parallelized, fully utilizing multicore CPU resources
   - Lightning-fast file I/O for large datasets

🔧 **Comprehensive Format Support**
   Native support for mainstream MD formats:
   
   - LAMMPS (DUMP/DATA)
   - VASP (POSCAR)
   - Universal XYZ format
   - Custom MP format

🔗 **Seamless Integration**
   - All data stored in Polars DataFrame and transfer with NumPy NDARRAY format
   - Easy integration with Python scientific ecosystem
   - Perfect compatibility with post-processing tools like OVITO, ASE and freud

--------

🎨 **Core Features**
========================================

🔬 **Neighbor Search**
--------

- Fixed cutoff distance neighbor
- Fixed number nearest neighbor
- Voronoi neighbor

📊 **Structural Analysis**
--------

- Ackland Jones Analysis
- Common Neighbor Parameter
- CentroSymmetry Parameter
- Common Neighbor Analysis
- Polyhedral Template Matching
- Identify Fcc Planar Faults
- Identify Diamond Structure
- Structure Entropy
- AtomicStrain
- Steinhardt Bond Orientation
- Radial Distribution Function
- Bond Analysis
- Angular Distribution Function
- Structure Factor

📦 **Model Building**
--------

- **Large scale** polycrystalline structure generation
- Standard lattice configuration creation
- High entropy alloy structure

🤖 **Machine Learning Potential**
--------

- NEP/qNEP potential support
- Elastic constant
- Minimization with cell optimization
- Equation of state
- Stacking faults energy
- Phonon dispersion 

📚 **Miscellaneous**
--------

- Generate EAM potential
- Average EAM potential 
- Mean Square Displacement
- Lindemann Parameter
- Cluster Analysis 
- Void Analysis 
- Warren Cowley Parameter
- Average Atomic Temperature
- Average by Neighbor 
- Atomic Voronoi Volume
- Multi-dimensional Spatial Binning
- Parallelly Compress file to .gz format

--------

🚀 **Quick Start**
========================================

💾 **Installation**
--------

**Method 1: Install via pip (Recommended)**

.. code-block:: bash

   pip install mdapy

**Method 2: Install with all optional features**

.. code-block:: bash

   # Install with matplotlib
   pip install mdapy[plot]

.. code-block:: bash

   # Install with k3d
   pip install mdapy[k3d]

.. code-block:: bash

   # Install all optional packages (matplotlib, k3d, phonopy, pyfftw), except lammps
   pip install mdapy[all]

**Method 3: Install from source**

.. code-block:: bash

   git clone https://github.com/mushroomfire/mdapy.git
   cd mdapy
   pip install .

**Method 4: Install directly from GitHub**

.. code-block:: bash

   pip install https://github.com/mushroomfire/mdapy/archive/master.zip --force-reinstall

📋 **System Requirements for Method 3 and 4**
~~~~~~~~

- Python 3.8+
- C++ compilation environment (C++17 or newer)
- OpenMP support
- Tested platforms: Windows 10 (MSVC), Ubuntu (GCC), Mac OS M1 (Clang)

✅ **Verify Installation**
~~~~~~~~

.. code-block:: bash

   python -c "import mdapy as mp; print('mdapy version is:', mp.__version__)"

--------

💡 **Usage Examples**
========================================

🎬 **Basic Usage**

.. code-block:: python

   import mdapy as mp
   
   # Load LAMMPS DUMP file
   system = mp.System('your_file.dump')
   
   # View system information
   print(system)
   
   # Calculate radial distribution function
   rdf = system.cal_radial_distribution_function(rc=5.0)
   rdf.plot()
   
   # Identify crystal structures
   system.cal_polyhedral_template_matching()

--------

📚 **Documentation & Resources**
========================================

📖 **Full Documentation**
   https://mdapy.readthedocs.io/

🎓 **Tutorial Repository**
   https://github.com/mushroomfire/mdapy-tutorial

💬 **Issue Tracker**
   https://github.com/mushroomfire/mdapy/issues

🏠 **Project Homepage**
   https://github.com/mushroomfire/mdapy

--------

📦 **Dependencies**
========================================

**Core Dependencies**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Package
     - Version
     - Purpose
   * - numpy
     - latest
     - Array computations
   * - polars
     - latest
     - Data processing

**Optional Dependencies**

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Package
     - Version
     - Purpose
   * - matplotlib
     - latest
     - Data visualization
   * - k3d
     - latest
     - Atom 3D visualization
   * - pyfftw
     - latest
     - Fast Fourier Transform
   * - phonopy
     - latest
     - Phonon calculations
   * - lammps
     - latest
     - Potential calculations

--------

🔄 **Version Updates**
========================================

⚠️ **Important Notice**

We have undergone a major upgrade from mdapy 1.0 (almost rewriting everyting). The old version of mdapy (≤0.11.5) can be found in this `branch <https://github.com/mushroomfire/mdapy/tree/mdapy_old>`_.

--------

📜 **Citation**
========================================

If you use mdapy in your scientific publications, please cite the following paper:

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

--------

⭐ **Support the Project**
========================================

If you find mdapy useful, please give us a `Star ⭐ <https://github.com/mushroomfire/mdapy>`_!

Your support is our motivation for continuous improvement! 🙏

--------

📄 **License**
========================================

This project is licensed under the BSD 3-Clause License - see the `LICENSE <https://github.com/mushroomfire/mdapy/blob/master/LICENSE>`_ file for details.

--------

🤝 **Contributing**
========================================

Issues and Pull Requests are welcome!

Let's make mdapy better together! 💪

--------

📧 **Contact**
========================================

If you have any questions or suggestions, feel free to reach us through:

- 📬 Submit an Issue: https://github.com/mushroomfire/mdapy/issues
- 💬 Email to Author: 934313174@qq.com 

--------