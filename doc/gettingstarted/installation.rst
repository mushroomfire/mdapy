Installation
=============

**Method 1: Install via pip (Recommended)**

.. code-block:: bash

   pip install mdapy


.. note::

   **GPU-accelerated ray-tracing (NVIDIA OptiX 7)**

   Pre-compiled wheels with GPU rendering support are published for
   **Linux x86_64** and **Windows x86_64**.  The GPU wheel is selected
   automatically by pip on supported platforms:

   - **Linux** (glibc ≥ 2.28, e.g. Ubuntu 20.04+): pip automatically installs
     the GPU-capable wheel.  Older Linux systems receive the CPU-only wheel.
   - **Windows**: the published wheel always includes GPU support.
   - **macOS**: CPU-only (Apple Silicon does not support CUDA).

   The GPU wheel works on CPU-only machines too — ``backend='gpu'`` will raise
   a ``RuntimeError`` if no suitable GPU is found, while ``backend='cpu'`` and
   all other features continue to work normally.

   **Runtime requirements (GPU features only):**

   - NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal or newer)
   - NVIDIA display driver ≥ 520 (ships with OptiX 7 support)
   - No CUDA Toolkit installation needed — the runtime is statically linked

   Check GPU availability at runtime:

   .. code-block:: python

      from mdapy import render
      print(render.is_gpu_available())   # True if GPU backend is usable


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
