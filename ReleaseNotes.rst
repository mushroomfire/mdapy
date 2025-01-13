Release Notes
--------------

Mdapy 0.11.6 (Under development)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🏆 Highlights
---------------
- Fix a bug for writting data with charge format.
- Add a_face_area_threshold and r_face_area_threshold parameters when building voronoi neighbor.
- Add a feature to calculate AngularDistributionFunctions.


Mdapy 0.11.5 (Nov 27, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🏆 Highlights
---------------
- Add feature to calculate local WCP parameter.
- Remove NEP model output.

Mdapy 0.11.4 (Nov 13, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🏆 Highlights
---------------
- Add feature to Minimizer class for optimizing the box.
- Add many options for calculating SteinhardtBondOrientation, including the Voronoi neighbor, weighted by Voronoi face area and average version.

🐞 Bug fixes
-----------------
- Fix a bug for reading xyz format with multi space.


Mdapy 0.11.3 (September 25, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

🛠️ Other improvements
---------------------------
- Upgrade the minimum polars to the latest version 1.8.0, so many code changes have been done.
- Update doc.

Mdapy 0.11.2 (September 23, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
🏆 Highlights
---------------
- Add a new feature to calculate the distribution of bond length and bond angle.

🐞 Bug fixes
-----------------
- Fix a typo bug for LammpsPotential.

🛠️ Other improvements
---------------------------
- Add test script for lammps interface.

Mdapy 0.11.1 (September 19, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
🏆 Highlights
---------------
- Refactor the visualization code in Jupyter, make it more beautiful and convinient.
- Add minimizer engine with FIRE method.
- Add a new feature to calculate the atomic shear strain.
- Add OrthogonalBox module to change triclinic box to rectangular box.
- Add adaptive common neighbor analysis method.
- Add IdentifyDiamondStructure method.

🚀 Performance improvements
-----------------------------
- Improve the performance for split_dump and split_xyz method.

✨ Enhancements
-----------------
- Add an example to compute Vacancy Formation energy.
- Make CreatePolycrystalline module can accept any files as matrix.
- Let the max type in LammpsPotential module is determined by given elemental list, enabling more flexible usage.

🐞 Bug fixes
-----------------
- Fix a bug for cell_opt class.
- Fix a bug for neighbor class.
- Fix a bug for RDF with partial interaction in multi-elemental system.
- Fix a bug for reading cp2k output file with OT method.
- Let GUI model compatible with latest polars.

🛠️ Other improvements
---------------------------
- Update NEP interface to V1.3, compatible with GPUMD 3.9.5.
- Updated some documentations.

V0.11.0 (June 22, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fix a bug for create_polycrystalline module.
- Fix a bug for reading xyz.

V0.10.9 (June 20, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Refactor the code about generating box, supporting all triclinic box now. The previous version only support lammps style triclinic box.
- Add the farthest point sampling method to select the configuration by descriptor.
- Add stress_max parameter for DFT2NEPXYZ class.
- Fix a bug for obtaining descriptor in NEP interface.
- Fix a bug for reading xyz.
- Refactor code of unwraping position.
- Let read/write xyz containing the global information, such as energy, virial and stress.
- Add cutoff_radius and displacement parameter in Phonon class.
- Rewrite the plot_dispersion function to make users custom figures easily.
- Reading POSCAR dose not need to rotate the box.
- Add MANIFEST.in file.

V0.10.8 (May 17, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Let tqdm as a dependency package.
- Fix frame counter in DFT2NEPXYZ.
- Fix a bug for NEP potential with triclinic box.
- Add write_cp2k method.
- Fix a bug when reading xyz.
- Fix a bug when writing cif.
- Fix a bug for phonon calculation.

V0.10.7 (April 25, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Add phonon calculation feature based on the phonopy, supporting any kind of potential format.
- Add atomic virial computation for eam/alloy potential.
- Fix a bug for NEP interface.
- Fix a bug for read_data.
- Add feature of cell_opt, using lammps as calculation backend.
- Add phonopy and lammps as optional package.
- Add force_max and mode parameter for DFT2NEPXYZ class.
- Support taichi>=1.7.1.
- Support polars>=0.20.22.
- Update readme.

V0.10.6 (April 13, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^

- Fix a typo bug in DFT2NEPXYZ class.

V0.10.5 (April 12, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^

- Refactor the code structure. Delete calculator file. Remove timer into tool_function file.
- Support `NEP <https://gpumd.org/potentials/nep.html>`_ model to evaluate the energy, force and virial.
- Add feature for generating initial geometry model with perturbation, which is helpful to prepare the initial database for deep learning. The function is similar to `init_bulk` and `init_surf` in `dpgen <https://github.com/deepmodeling/dpgen>`_.
- Add feature for converge cp2k output to xyz format for `NEP <https://gpumd.org/potentials/nep.html>`_ trainning.
- Add feature to split dump/xyz containing multi frames into seperate frames.
- Optimize timer decorators.
- Fix bug when writing cif and POSCAR.
- Fix bug for create_polycrystalline when input wrong box, and optimize the performance of deleting overlap atoms.
- Make mdapy support polars>=0.20.19.

V0.10.4 (March 12, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Add feature for plotting phonon dispersion based on the data generated by `phonopy <https://phonopy.github.io/phonopy/>`_.
- Fix bug for writing xyz.
- Make mdapy support polars>=0.20.15


V0.10.3 (January 30, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fix bug when read/write POSCAR with reduced positions.
- Fix bug when read data file with multi space.
- Fix bug when read dump with reduced positions.
- Add support for write data with type name list.
- Support read/write simple `Crystallographic Information File cif <https://www.iucr.org/resources/cif/spec/version1.1/cifsyntax>`_ format.

V0.10.2 (January 8, 2024)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Significantly optimize the performance of Neighbor class.
- Add label for colorbar while visualizing in jupyter.

V0.10.1 (December 18, 2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Provide a GUI interface based on the polyscope.
- Provide a GUI software for Windows platform.
- Fix a bug when generating System from array with velocity.
- Modify repr of System to print entire DataFrame.
- Fix a bug in pair_distribution class.
- Fix a bug when loading dump file.
- Fix a bug in PTM module.
- Update polars version to 0.20.0
- Updated README.

V0.10.0 (November 28, 2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Make mdapy support polars>=0.19.17
- Make mdapy support taichi>=1.7.0
- Fix a bug in void_distribution class
- Fix a bug when system with small size
- Change the display when system pos changed

V0.9.9 (November 21, 2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Rewrite the pltset and add a set_figure feature, which makes plotting easier.
- We can use elemental list to calculate the atomic temperature now.
- Fix a bug when loading xyz file.
- Update support for Polars>=0.19.14.
- Prepared to support Python 3.12.
- Remove the SciencePlots and pyfnntw as optional dependency for concise.

V0.9.8 (November 13, 2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Update support for Polars>=0.19.13
- Support clustering with multi cutoff distance for different elemental pairs.
- Add species clustering feature.
- Let `k3d <https://matplotlib.org/>`_ be an optional dependency. One can install it only if you need visualize the System in Jupyter environment.

V0.9.7 (11/5/2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Experimentally support Visualizing System (only in Jupyter environment).
- Add `k3d <https://matplotlib.org/>`_ as a dependency.
- Add `jupyter <https://jupyter.org/>`_ as a dependency.

V0.9.6 (11/2/2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- One can explicitly assign the type number when writing to data file.
- Support load/save POSCAR format.
- Support load/save XYZ format.

V0.9.5 (10/24/2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fix the documentations.
- Add a dynamic `logo <https://mdapy.readthedocs.io/en/latest/>`_.
- Improve the memory use for System class.
- Improve the README.
- Add plot for 3D spatial binning.

V0.9.4 (10/20/2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Remove dependency for **Pandas** and **Pyarrow**. mdapy uses **Polars** to be the newer DataFrame structure.
- Updated Documentation.
- Improve the importing speed.
- Minor improvement on compilation speed.

V0.9.3 (10/19/2023)
^^^^^^^^^^^^^^^^^^^^^

- Support generating special crystalline orientations for FCC and BCC lattice.
- Fix bug for warpping positions.
- Fix bug for write dump.
- Fix bug for generate System class from np.ndarray.
- Update an example to calculate the Generalized Stacking Fault Energy (GSFE).

V0.9.2 (10/12/2023)
^^^^^^^^^^^^^^^^^^^^^^

- Fix capacity of cross-platform.
- Updated doc.

V0.9.1 (10/11/2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Add **Polars** as dependency package. Now we still use pandas, but mdapy maybe move to polars in the future.
- Optimize the performance of reading and saving Dump and Data file.
- Support loading/saving compressed Dump file (such as sample.dump.gz).
- Support the lowest python version to 3.8.0.
- Add pyproject.toml.

V0.9.0 (9/23/2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Support triclinic box now!!!
- Add Select feature.
- Rewrite the load and save module.
- Make many method suitable for small system.
- Fix some bugs.

V0.8.9 (9/5/2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fix installation in python 3.11.5.


V0.8.8 (8/24/2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fix memory leak in SpatialBinning class, not the correct issue.
- Fix bug in SteinhardtBondOrientation class.
- Fix bug in read data.
- Fix bug in spatial_binning.
- Updated the IdentifySFTBinFCC class to identify the twinning and extrinsic stacking fault.

V0.8.7 (5/25/2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Updated Taichi to 1.6.0, which decreases the import time and supports Python 3.11.
- Fix bug in read data.
- Updated mdapy citation. We are pleased that our article for mdapy has been accepted by **Computer Physics Communications**.

V0.8.6 (4/22/2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Add repr for System class.
- Add Replicate class.
- Improve the performance of **reading/writing DATA file with pyarrow**.
- Improve the performance of **building Voronoi diagram** with new version voro++. 

V0.8.5 (4/9/2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Compile it on MAC OS with M1. Now **mdapy** is fully cross-platform.
- Obviously improve the performance of **reading/writing DUMP with pyarrow**.
- Add **pyarrow** as a dependency package.
- Fix bug of **create_polycrystalline** module. One can give box with any number, the old version only works for positive float.
- Fix bug of **spatial_binning** module for empty region.
- Let **tqdm** as an Optional dependency. 

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