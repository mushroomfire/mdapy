====================================================================
mdapy — Molecular Dynamics Analysis in Python
====================================================================

.. image:: https://img.shields.io/pypi/v/mdapy.svg
   :target: https://pypi.org/project/mdapy/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/mdapy.svg
   :target: https://pypi.org/project/mdapy/
   :alt: Python versions

.. image:: https://img.shields.io/pypi/dm/mdapy.svg
   :target: https://pypi.org/project/mdapy/
   :alt: PyPI downloads

.. image:: https://img.shields.io/github/license/mushroomfire/mdapy.svg
   :target: https://github.com/mushroomfire/mdapy/blob/master/LICENSE
   :alt: License

.. image:: https://readthedocs.org/projects/mdapy/badge/?version=latest
   :target: https://mdapy.readthedocs.io/en/latest/
   :alt: Documentation

.. image:: https://img.shields.io/github/stars/mushroomfire/mdapy.svg?style=social
   :target: https://github.com/mushroomfire/mdapy
   :alt: GitHub stars

|

**mdapy** is a fast, full-featured Python library for analyzing Molecular Dynamics (MD)
simulation data — from structural characterization and machine-learning potential
workflows to built-in ray-tracing visualization, all in a clean Pythonic API.

.. code-block:: bash

   pip install mdapy

|

----

Why mdapy?
----------

.. list-table::
   :widths: 5 25 70
   :header-rows: 0

   * - ⚡
     - **Blazing Fast**
     - C++ core (via Nanobind) with full OpenMP parallelism. Analyses that take minutes
       elsewhere finish in seconds.

   * - 🐍
     - **Pythonic by Design**
     - One-liner analyses. All data lives in a
       `Polars <https://pola.rs>`_ DataFrame and interops natively with NumPy.

   * - 🌍
     - **Truly Cross-Platform**
     - Pre-built wheels for Windows, Linux, and macOS (including Apple Silicon).
       No compiler needed for standard installs.

   * - 🎨
     - **Built-in Ray-Tracing**
     - Tachyon-powered CPU *and* GPU (NVIDIA OptiX) renderer baked right in —
       no third-party renderer required.

   * - 🤝
     - **Ecosystem Friendly**
     - First-class interop with OVITO, ASE, freud, phonopy, and LAMMPS.

   * - 📖
     - **Thoroughly Documented**
     - Full API reference, tutorials, and Jupyter notebooks on
       `ReadTheDocs <https://mdapy.readthedocs.io>`_.

   * - 🔄
     - **Actively Maintained**
     - Regular releases, responsive issue tracker, and a growing feature set.

----

Feature Overview
----------------

Neighbor Search
~~~~~~~~~~~~~~~

+---------------------------+---------------------------------------------+
| Fixed-radius cutoff       | Efficient cell-list algorithm               |
+---------------------------+---------------------------------------------+
| k-Nearest neighbors       | Exact kNN via KD-tree                       |
+---------------------------+---------------------------------------------+
| Voronoi neighbors         | Topology-based, powered by Voro++           |
+---------------------------+---------------------------------------------+

Structural Analysis
~~~~~~~~~~~~~~~~~~~

- **Structure identification** — Polyhedral Template Matching (PTM),
  Common Neighbor Analysis (CNA), Ackland-Jones Analysis,
  Common Neighbor Parameter, Identify Diamond Structure
- **Defect characterization** — Centro-Symmetry Parameter (CSP),
  Identify FCC Planar Faults (ISF / TB / ESF), Atomic Strain,
  Wigner-Seitz defect analysis
- **Order parameters** — Steinhardt Bond Orientation (q₂…q₁₂, averaged),
  Structure Entropy, Warren-Cowley SRO parameter
- **Distributions** — Radial Distribution Function, Structure Factor
  (total + partial + X-ray weighted), Angular Distribution Function,
  Bond Analysis

Model Building
~~~~~~~~~~~~~~

- Single-crystal lattices (FCC, BCC, HCP, diamond, and more)
- **Large-scale polycrystalline** structures via Voronoi tessellation
- High-entropy alloy (HEA) configurations

Machine-Learning Potential Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **NEP / qNEP** potential evaluation (energy, forces, stress, virials)
- ASE-compatible ``NEP4ASE`` calculator
- Elastic constant tensor calculation
- Equation-of-state (EOS) fitting
- Stacking-fault energy curves
- Phonon dispersion via phonopy

Visualization
~~~~~~~~~~~~~

- **TachyonRender** — CPU or NVIDIA OptiX GPU ray-tracing, renders
  any ``System`` to a PNG/JPEG with one method call
- Per-element colors and radii, transparent background, anti-aliasing,
  ambient occlusion, shadows — all configurable

Utilities
~~~~~~~~~

- EAM potential generation and averaging
- Mean Squared Displacement (FFT-accelerated or direct)
- Lindemann parameter, Void analysis, Cluster analysis
- Spatial binning (multi-dimensional)
- Atomic temperature averaging
- Parallel ``.gz`` compression
- Multi-frame XYZ trajectory reader / splitter

----

Installation
------------

**One command (recommended):**

.. code-block:: bash

   pip install mdapy

**With optional extras:**

.. code-block:: bash

   pip install mdapy[plot]      # + matplotlib
   pip install mdapy[k3d]       # + k3d 3-D widget
   pip install mdapy[all]       # + matplotlib, k3d, pyfftw, phonopy

**From source** (requires C++17 compiler + OpenMP):

.. code-block:: bash

   git clone https://github.com/mushroomfire/mdapy.git
   cd mdapy
   pip install .

Tested compilers: MSVC (Windows 10), GCC (Ubuntu), Clang (macOS M1).

**Verify:**

.. code-block:: bash

   python -c "import mdapy as mp; print('mdapy', mp.__version__, '— ready!')"

----

Quick Examples
--------------

**Load & analyse in three lines:**

.. code-block:: python

   import mdapy as mp

   sys = mp.System("dump.lammps")
   sys.cal_polyhedral_template_matching()   # adds 'structure_types' column
   sys.cal_centro_symmetry_parameter(N=12)  # CSP for FCC
   print(sys.data)                          # Polars DataFrame

**Radial distribution function:**

.. code-block:: python

   rdf = sys.cal_radial_distribution_function(rc=6.0)
   rdf.plot()           # matplotlib figure — one line

**Machine-learning potential workflow:**

.. code-block:: python

   from mdapy import get_elastic_constant

   C = get_elastic_constant("nep.txt", "POSCAR")
   print(C)   # 6×6 elastic tensor in GPa

**Ray-tracing render:**

.. code-block:: python

   from mdapy.render import TachyonRender

   ren = TachyonRender(backend="auto")          # GPU if available, else CPU
   ren.render_system(sys, width=1920, height=1080,
                     output_figure="structure.png")

**Build a polycrystal:**

.. code-block:: python

   import mdapy as mp

   poly = mp.CreatePolycrystal(
       box_length=[100, 100, 100],
       num_grains=50,
       metal="Al",
   )
   poly.compute()
   poly.system.write("polycrystal.dump")

----

Supported File Formats
----------------------

+------------------------+---------------------------+
| Format                 | Read / Write              |
+========================+===========================+
| LAMMPS DUMP            | ✅ / ✅                   |
+------------------------+---------------------------+
| LAMMPS DATA            | ✅ / ✅                   |
+------------------------+---------------------------+
| VASP POSCAR/CONTCAR    | ✅ / ✅                   |
+------------------------+---------------------------+
| XYZ (extended)         | ✅ / ✅                   |
+------------------------+---------------------------+
| MP (mdapy native)      | ✅ / ✅                   |
+------------------------+---------------------------+
| ASE Atoms              | ✅ (import / export)      |
+------------------------+---------------------------+
| OVITO DataCollection   | ✅ (import)               |
+------------------------+---------------------------+

----

Documentation & Resources
--------------------------

+----------------------------------+------------------------------------------------------+
| 📖 Full documentation            | https://mdapy.readthedocs.io                         |
+----------------------------------+------------------------------------------------------+
| 🎓 Jupyter notebook tutorials    | https://github.com/mushroomfire/mdapy-tutorial       |
+----------------------------------+------------------------------------------------------+
| 🏠 Source code                   | https://github.com/mushroomfire/mdapy                |
+----------------------------------+------------------------------------------------------+
| 🐛 Issue tracker                 | https://github.com/mushroomfire/mdapy/issues         |
+----------------------------------+------------------------------------------------------+

----

Dependencies
------------

**Required:**

- `numpy <https://numpy.org>`_ — array engine
- `polars <https://pola.rs>`_ — fast DataFrame backend

**Optional:**

- ``matplotlib`` — plotting (``pip install mdapy[plot]``)
- ``k3d`` — interactive 3-D notebook viewer (``pip install mdapy[k3d]``)
- ``pyfftw`` — faster FFT for MSD calculations
- ``phonopy`` — phonon dispersion
- ``lammps`` — LAMMPS-based potential calculations

----

Running the Tests
-----------------

.. code-block:: bash

   pip install pytest scikit-learn ase freud-analysis pyfftw pymatgen
   pip install git+https://github.com/bigd4/PyNEP.git
   # also install OVITO and LAMMPS via your preferred method
   cd tests && pytest

----

Citation
--------

If mdapy contributes to a scientific publication, please cite:

.. code-block:: bibtex

   @article{mdapy2023,
     title   = {mdapy: A flexible and efficient analysis software for
                molecular dynamics simulations},
     journal = {Computer Physics Communications},
     pages   = {108764},
     year    = {2023},
     issn    = {0010-4655},
     doi     = {10.1016/j.cpc.2023.108764},
     url     = {https://www.sciencedirect.com/science/article/pii/S0010465523001091},
     author  = {Yong-Chao Wu and Jian-Li Shao},
   }

----

Version Notice
--------------

mdapy 1.0 is a ground-up rewrite with a new, cleaner API.
The legacy release (≤ 0.11.5) is preserved on the
`mdapy_old branch <https://github.com/mushroomfire/mdapy/tree/mdapy_old>`_.

----

Contributing
------------

Bug reports, feature requests, and pull requests are all very welcome!
Please open an `issue <https://github.com/mushroomfire/mdapy/issues>`_ or
submit a PR on GitHub.

----

License
-------

BSD 3-Clause — see `LICENSE <https://github.com/mushroomfire/mdapy/blob/master/LICENSE>`_
for details.

----

Contact
-------

- **Issues / feature requests:** https://github.com/mushroomfire/mdapy/issues
- **Email:** 934313174@qq.com

If mdapy helps your research, a ⭐ on
`GitHub <https://github.com/mushroomfire/mdapy>`_ is always appreciated!
