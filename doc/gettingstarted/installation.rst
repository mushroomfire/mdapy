Installation
=============

**Method 1: Install via pip (Recommended)**

.. code-block:: bash

   pip install mdapy

**Method 2: Install with all optional features**

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

   pip install https://github.com/mushroomfire/mdapy/archive/master.zip

**System Requirements for Method 3 and 4**

- Python 3.8+
- C++ compilation environment (C++17 or newer)
- OpenMP support
- Tested platforms: Windows 10 (MSVC), Ubuntu (GCC), Mac OS M1 (Clang)

**Verify Installation**

.. code-block:: bash

   python -c "import mdapy as mp; print('mdapy version is:', mp.__version__)"
