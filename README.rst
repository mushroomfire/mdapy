.. image:: https://img.pterclub.com/images/2023/01/06/logo.png

*mdapy* : Molecular Dynamics Analysis with Python
=====================================================

Overview
--------

The **mdapy** is a python library developed by **Yong-Chao Wu & Jian-Li Shao Group in 
Beijing Institute of Technology**, providing a set of simple, 
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
* `pyfnntw>=0.4.1 <https://github.com/cavemanloverboy/FNNTW>`_ (Optional, for fast KNN search)


Installation
-------------

Install from pip (recommended).
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install mdapy

Install from source code.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- You should install pybind11 and have a C++ compilation environment. Tested by MSVC in Windows and GCC in Ubuntu.

   .. code-block:: bash
      
      pip install pybind11

- Download source code
   
   .. code-block:: bash

      git clone https://github.com/mushroomfire/mdapy.git
      cd mdapy 
      pip install .

Usage
------

.. code-block:: python

   import mdapy as mp
   mp.init('cpu') # use cpu, mp.init('gpu') will use gpu to compute.

   system = mp.System('./example/CoCuFeNiPd-4M.dump') # read dump file to generate a system class
   system.cal_centro_symmetry_parameter() # calculate the centrosymmetry parameters
   system.cal_atomic_entropy() # calculate the atomic entropy
   system.write_dump() # save results to a new dump file


Main Features
--------------

1. Structure Analysis
   
   - `Ackland Jones Analysis <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.ackland_jones_analysis>`_
   - `CentroSymmetry Parameter <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.centro_symmetry_parameter>`_ 
   - `Common Neighbor Analysis <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.common_neighbor_analysis>`_ 
   - `Common Neighbor Parameter <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.common_neighbor_parameter>`_
   - `Atomic Structure Entropy <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.entropy>`_ 
   - `Steinhardt Bondorder <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.steinhardt_bond_orientation>`_ 
   - `Radiul Distribution Function <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.pair_distribution>`_
   - `Polyhedral Template Matching <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.polyhedral_template_matching>`_
   - `Identify stacking faults (SFs) and twinning boundary (TBs) <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.identify_SFs_TBs>`_

2. Potential Analysis 

   - `Generate EAM/alloy Potential <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.eam_generate>`_
   - `Read EAM/alloy Potential <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.potential>`_
   - `Average EAM/alloy Potential <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.eam_average>`_
   - `Calculate Atomic Force and Energy by EAM/alloy <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.calculator>`_

3. Melting Analysis 

   - `Mean Squared Displacement <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.mean_squared_displacement>`_
   - `Lindemann Parameter <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.lindemann_parameter>`_
   - `Identify Solid/Liquid Phase <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.steinhardt_bond_orientation>`_

4. Geometry Structure Creation 

   - `Generate Standard Lattice Structure <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.lattice_maker>`_
   - `Generate Polycrystal <https://mdapy.readthedocs.io/en/latest/mdapy.html#mdapy.create_polycrystalline.CreatePolycrystalline>`_

5. Neighbor Search 

   - `Neighbor Atoms within Fixed Distance <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.neighbor>`_
   - `Neighbor Atoms within Fixed Number <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.kdtree>`_

6. Other 

   - `Void Distribution <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.void_distribution>`_
   - `Cluster Analysis <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.cluser_analysis>`_
   - `Warren Cowley Parameter <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.warren_cowley_parameter>`_
   - `Average Atomic Temperature <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.temperature>`_
   - `Atomic Voronoi Volume <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.voronoi_analysis>`_
   - `Multi-dimensional Spatial Binning <https://mdapy.readthedocs.io/en/latest/mdapy.html#module-mdapy.spatial_binning>`_


Citation
---------
If you find **mdapy** useful, you can start it!
If you use **mdapy** in your scientific publication, because the paper is under
review on *Computer Physics Communication* and you can cite its preprint version as below:

.. code-block:: bash

   Wu, Yong-Chao and Shao, Jian-Li, Mdapy: A Flexible and Efficient Analysis Software for 
   Molecular Dynamics Simulations. Available at SSRN: https://ssrn.com/abstract=4360931 
   or http://dx.doi.org/10.2139/ssrn.4360931.


Trouble Shoot
-------------

If you encounter ImportError in Linux: 

.. code-block:: bash

   version 'GLIBCXX_3.4.29' not found. 

You can try: 

.. code-block:: bash

   conda install -c conda-forge gxx_linux-64

Release Notes
--------------

V0.8.4 (3/30/2023)
^^^^^^^^^^^^^^^^^^^

- Optimize **Pair Distribution** module.
- Optimize **Neighbor** module.
- Update many **Benchmark** cases.

V0.8.3 (3/20/2023)
^^^^^^^^^^^^^^^^^^^

- Make **Polyhedral Template Mathing** parallel.

V0.8.2
^^^^^^^^^

- Fix bugs of unwrap positions.
- Fix a typo error in msd.

V0.8.1
^^^^^^^

- Add **Steinhardt Bondorder Parameter** method, which can be used to identify the lattice structure and distinguish
  the solid/liquid phase during melting process.
- Add **Polyhedral Template Mathing** method.
- Add **IdentifySFsTBs** method to identify the stacking faults (SFs) and twinning boundary (TBs) in FCC lattice.


V0.8.0
^^^^^^^

- Add **Ackland Jones Analysis (AJA)** method.
- Add **Common Neighbor Parameter (CNP)** method.
- Update the nearest neighbor search in CSP method.

V0.7.9
^^^^^^^

- Fix bug of create_polycrystalline module in Linux.

V0.7.8
^^^^^^^

- Update TaiChi version to 1.4.0.
- Set SciencePlots as a optional package.
- Fix bug in create_polycrystalline.