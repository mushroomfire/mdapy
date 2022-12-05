# *mdapy* : Molecular Dynamics Analysis with Python
## Overview
The **mdapy** is a python library providing a set of simple, flexible and powerful tools to analyze the atomic trajectories generated from Molecular Dynamics (MD) simulations. Benefit by the [TaiChi](https://github.com/taichi-dev/taichi) project, we can effectively accelerate the pure python code close to those written in C++. Moreover, **mdapy** is highly parallelized to make full use of resources of both multicore CPU and GPU. **mdapy** can directly handle the *DUMP* and *DATA* format in [LAMMPS](https://www.lammps.org/). All data in **mdapy** is stored in ndarray format in [Numpy](https://numpy.org/), enabling integration with the scientific ecosystem in python and corporation with other post-progressing codes, such as [OVITO](https://www.ovito.org/) and [freud](https://github.com/glotzerlab/freud). 
## Dependencies
- python (>=3.7 and < 3.11)
- taichi
- numpy
- scipy
- pandas
- pyfftw
- tqdm
- matplotlib
- SciencePlots
## Install
1. Download source code
   ```bash
   git clone https://github.com/mushroomfire/mdapy.git
   cd mdapy 
   ```
2. Create a virtual environment (highly recommoned using anaconda)
   - It helps you to avoid the potential conflicts of dependencies in your current python environment and is very suitable for beginners. If you have installed all dependencies listed above, you can just skip this step!
   ```bash
   conda create -n mda python==3.8.0 # python version larger than 3.7 and lower than 3.11 is okay.
   conda activate mda
   ```
3. Install dependencies
    ```bash
   pip install -r requirements.txt
   ```
4. Install mdapy
   ```python
   python install.py
   ```
5. Update mdapy
   - Just download the latest souce code and repeat the step 3 and step 4.
## Usage

## Benchmark
- All python script files can be found in mdapy/benchmark folder.
- CPU: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz
- Memory: 64 Gb
- mdapy version: 0.7.2, freud version: 2.12.0, ovito version: 3.7.11
1. Build neighbor list.
![](benchmark/buildneighbor_mdapy_freud.png)
2. Build FCC lattice structure.
![](benchmark/build_lattice_mdapy_freud.png)
3. Calculate radial distribution functions
![](benchmark/rdf_mdapy_freud.png)
4. Calculate atomic entropy.
![](benchmark/entropy_mdapy_ovito.png)
5. Cluster analysis.
![](benchmark/cluster_mdapy_ovito.png)

## Citations
If you use **mdapy** to process data for publication, please use this citation:
```latex
@article{mdapy2022,
    title = {mdapy: A simple and fast analysis software for molecular dynamics simulations with python},
    author = {Yong-Chao Wu, XiaoYa Chang, Jian-Li Shao},
    journal = {},
    volume = {},
    pages = {},
    year = {2022},
    issn = {},
    doi = {},
    url = {},
    keywords = {Simulation analysis, Molecular dynamics},
}
```