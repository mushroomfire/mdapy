# mdapy, OVITO and freud

The analyses that mdapy shares with [OVITO](https://www.ovito.org) and
[freud](https://freud.readthedocs.io), computed on the same systems with
the same parameters, from 4 000 to 2 million atoms. A program is timed only
where it gives the result of mdapy, atom by atom.

## Results

A node of the Triton cluster of Aalto University, alone for the run: two
Intel Xeon Gold 6248 (40 cores, two NUMA nodes), 191 GB, Linux. Python 3.13,
mdapy 2.0.0.dev0 built with GCC 12.3, OVITO 3.17.1 and freud 3.6.1 from
pip, each with 40 threads.

![Time of OVITO and of freud over the time of mdapy for each case and size](images/comparison.png)

### What they show

- **OVITO**: from 256 000 atoms on, mdapy is faster in every analysis they
  share. It builds neighbor lists 6.4 to 7.6 times faster in a quarter of
  the memory, Voronoi cells 22 to 25 times faster, Steinhardt parameters
  over Voronoi faces 65 times, finds dislocations 4.6 to 4.9 times and
  clusters 6 times faster; the other analyses take 1.2 to 3.8 times less
  time. Atomic strain is the closest: OVITO is faster below some 250 000
  atoms (3 times at 4 000 and 1.3 times at 32 000), the two are on par at
  256 000, and mdapy is 1.16 times faster at 2 million.
- **freud**: from 32 000 atoms on, mdapy is faster in every analysis:
  neighbor lists 9 to 27 times, in a seventh of the memory; the 12 nearest
  neighbors 4 to 9 times; Voronoi cells 29 to 41 times; Steinhardt
  parameters and the radial distribution function 3.4 to 4.9 times;
  cluster analysis 1.2 to 2.6 times.
- **One neighbor search for several analyses**: the six analyses of copper
  from one list take 0.79 s at 2 million atoms, against 1.42 s for the six
  cases timed apart (0.11 s against 0.21 s at 256 000; the radial
  distribution function apart reaches 6 Å instead of 5 Å). OVITO's pipeline
  of the six takes about the sum of its cases (2.98 s against 3.11 s).
- **Voronoi faces on demand**: the volumes and coordination of 2 million
  cells take mdapy 0.95 s and 195 MB; with every face, its neighbor, area
  and edges, 1.12 s and 1.7 GB.
- **A trajectory**: the mean squared displacement of 20 frames read from a
  dump, most of the time spent reading, takes about 1.4 times less time
  than with OVITO from 32 000 atoms on.
- **4 000 atoms** take a few milliseconds; mdapy is faster there in every
  analysis but cluster analysis, Wigner–Seitz occupancies, atomic strain
  and the mean squared displacement against OVITO, and cluster analysis
  against freud (0.004 s against 0.001 s); the 12 nearest neighbors are on
  par with freud.
- **Not run**: Steinhardt parameters over Voronoi faces at 2 million atoms,
  which took OVITO 678 s in an earlier run.

### Times and memory

Every case at every size: the shortest of three times, and the memory each
computation took beyond the system, the peak resident memory of the
process minus its peak before the timings.

#### OVITO

| Case | Atoms | mdapy 2.0.0.dev0 (s) | OVITO 3.17.1 (s) | OVITO / mdapy | mdapy (MB) | OVITO (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 4,000 | 0.004 | 0.009 | 2.68 | 0 | 9 |
| Neighbor list, 5 Å | 32,000 | 0.007 | 0.036 | 5.04 | 18 | 131 |
| Neighbor list, 5 Å | 256,000 | 0.028 | 0.181 | 6.39 | 195 | 724 |
| Neighbor list, 5 Å | 2,048,000 | 0.184 | 1.390 | 7.55 | 1427 | 5407 |
| 12 nearest neighbors | 4,000 | 0.002 | 0.008 | 3.65 | 0 | 1 |
| 12 nearest neighbors | 32,000 | 0.005 | 0.012 | 2.39 | 5 | 13 |
| 12 nearest neighbors | 256,000 | 0.032 | 0.050 | 1.55 | 59 | 108 |
| 12 nearest neighbors | 2,048,000 | 0.183 | 0.442 | 2.41 | 413 | 837 |
| Adaptive common neighbor analysis | 4,000 | 0.002 | 0.015 | 8.03 | 0 | 0 |
| Adaptive common neighbor analysis | 32,000 | 0.005 | 0.021 | 4.55 | 0 | 8 |
| Adaptive common neighbor analysis | 256,000 | 0.034 | 0.073 | 2.14 | 16 | 48 |
| Adaptive common neighbor analysis | 2,048,000 | 0.213 | 0.597 | 2.80 | 127 | 201 |
| Common neighbor analysis, cutoff 3.09 Å | 4,000 | 0.003 | 0.006 | 2.09 | 0 | 0 |
| Common neighbor analysis, cutoff 3.09 Å | 32,000 | 0.001 | 0.009 | 6.50 | 0 | 6 |
| Common neighbor analysis, cutoff 3.09 Å | 256,000 | 0.012 | 0.024 | 1.93 | 21 | 28 |
| Common neighbor analysis, cutoff 3.09 Å | 2,048,000 | 0.067 | 0.155 | 2.30 | 205 | 87 |
| Ackland–Jones analysis | 4,000 | 0.002 | 0.005 | 2.25 | 0 | 1 |
| Ackland–Jones analysis | 32,000 | 0.006 | 0.010 | 1.77 | 1 | 8 |
| Ackland–Jones analysis | 256,000 | 0.042 | 0.068 | 1.62 | 14 | 49 |
| Ackland–Jones analysis | 2,048,000 | 0.275 | 0.560 | 2.04 | 125 | 178 |
| Polyhedral template matching | 4,000 | 0.013 | 0.026 | 1.96 | 0 | 0 |
| Polyhedral template matching | 32,000 | 0.034 | 0.040 | 1.17 | 6 | 10 |
| Polyhedral template matching | 256,000 | 0.224 | 0.277 | 1.24 | 82 | 40 |
| Polyhedral template matching | 2,048,000 | 1.679 | 2.233 | 1.33 | 579 | 269 |
| Centro-symmetry parameter, 12 neighbors | 4,000 | 0.002 | 0.010 | 5.53 | 0 | 0 |
| Centro-symmetry parameter, 12 neighbors | 32,000 | 0.005 | 0.015 | 3.24 | 0 | 8 |
| Centro-symmetry parameter, 12 neighbors | 256,000 | 0.034 | 0.059 | 1.72 | 16 | 41 |
| Centro-symmetry parameter, 12 neighbors | 2,048,000 | 0.217 | 0.480 | 2.21 | 139 | 207 |
| Diamond structure | 4,096 | 0.003 | 0.004 | 1.36 | 0 | 2 |
| Diamond structure | 32,768 | 0.006 | 0.010 | 1.52 | 1 | 21 |
| Diamond structure | 262,144 | 0.038 | 0.068 | 1.76 | 22 | 80 |
| Diamond structure | 2,000,376 | 0.199 | 0.517 | 2.60 | 173 | 345 |
| Steinhardt q2–q6, 12 neighbors | 4,000 | 0.001 | 0.014 | 9.62 | 0 | 0 |
| Steinhardt q2–q6, 12 neighbors | 32,000 | 0.006 | 0.017 | 2.89 | 0 | 11 |
| Steinhardt q2–q6, 12 neighbors | 256,000 | 0.045 | 0.064 | 1.40 | 24 | 57 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.350 | 0.517 | 1.47 | 138 | 164 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 4,000 | 0.009 | 0.294 | 33.51 | 3 | 1 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 32,000 | 0.045 | 1.407 | 31.50 | 54 | 12 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.345 | 22.451 | 65.02 | 449 | 44 |
| Radial distribution function, 6 Å, 200 bins | 4,000 | 0.002 | 0.006 | 3.70 | 0 | 0 |
| Radial distribution function, 6 Å, 200 bins | 32,000 | 0.004 | 0.010 | 2.64 | 0 | 4 |
| Radial distribution function, 6 Å, 200 bins | 256,000 | 0.026 | 0.045 | 1.71 | 14 | 17 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.169 | 0.358 | 2.12 | 138 | 67 |
| Bond angle distribution, 3 Å, 180 bins | 4,000 | 0.001 | 0.013 | 10.42 | 0 | 2 |
| Bond angle distribution, 3 Å, 180 bins | 32,000 | 0.004 | 0.021 | 4.88 | 1 | 32 |
| Bond angle distribution, 3 Å, 180 bins | 256,000 | 0.029 | 0.079 | 2.69 | 27 | 264 |
| Bond angle distribution, 3 Å, 180 bins | 2,048,000 | 0.196 | 0.598 | 3.05 | 232 | 1426 |
| Voronoi volumes and coordination | 4,000 | 0.005 | 0.047 | 9.23 | 0 | 0 |
| Voronoi volumes and coordination | 32,000 | 0.016 | 0.370 | 23.62 | 0 | 2 |
| Voronoi volumes and coordination | 256,000 | 0.120 | 2.964 | 24.77 | 37 | 11 |
| Voronoi volumes and coordination | 2,048,000 | 0.951 | 23.037 | 24.22 | 195 | 88 |
| Voronoi cells with their faces | 4,000 | 0.006 | 0.049 | 8.42 | 1 | 2 |
| Voronoi cells with their faces | 32,000 | 0.020 | 0.384 | 18.95 | 23 | 18 |
| Voronoi cells with their faces | 256,000 | 0.137 | 3.166 | 23.07 | 220 | 128 |
| Voronoi cells with their faces | 2,048,000 | 1.123 | 25.169 | 22.41 | 1696 | 1034 |
| Cluster analysis, 2 Å | 3,998 | 0.004 | 0.002 | 0.62 | 0 | 0 |
| Cluster analysis, 2 Å | 31,966 | 0.007 | 0.017 | 2.64 | 3 | 6 |
| Cluster analysis, 2 Å | 255,756 | 0.030 | 0.165 | 5.56 | 40 | 26 |
| Cluster analysis, 2 Å | 2,046,192 | 0.331 | 1.998 | 6.03 | 221 | 0 |
| Wigner–Seitz occupancies | 3,974 | 0.001 | 0.001 | 0.77 | 0 | 1 |
| Wigner–Seitz occupancies | 31,811 | 0.003 | 0.006 | 2.03 | 1 | 8 |
| Wigner–Seitz occupancies | 254,734 | 0.021 | 0.036 | 1.72 | 16 | 44 |
| Wigner–Seitz occupancies | 2,037,841 | 0.094 | 0.318 | 3.40 | 131 | 244 |
| Mean squared displacement, 20 frames read from a dump | 4,000 | 0.053 | 0.047 | 0.87 | 22 | 11 |
| Mean squared displacement, 20 frames read from a dump | 32,000 | 0.186 | 0.255 | 1.38 | 179 | 52 |
| Mean squared displacement, 20 frames read from a dump | 256,000 | 1.314 | 1.876 | 1.43 | 585 | 330 |
| Mean squared displacement, 20 frames read from a dump | 2,048,000 | 10.616 | 14.676 | 1.38 | 2404 | 813 |
| Six analyses of copper, one neighbor search | 4,000 | 0.008 | 0.066 | 7.80 | 1 | 7 |
| Six analyses of copper, one neighbor search | 32,000 | 0.017 | 0.095 | 5.59 | 21 | 87 |
| Six analyses of copper, one neighbor search | 256,000 | 0.107 | 0.375 | 3.51 | 197 | 493 |
| Six analyses of copper, one neighbor search | 2,048,000 | 0.793 | 2.977 | 3.75 | 1427 | 1755 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 4,000 | 0.006 | 0.020 | 3.17 | 0 | 1 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 32,000 | 0.010 | 0.026 | 2.65 | 20 | 12 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 256,000 | 0.057 | 0.094 | 1.64 | 180 | 83 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 2,048,000 | 0.414 | 0.748 | 1.81 | 1429 | 209 |
| Dislocation analysis | 4,000 | 0.031 | 0.121 | 3.87 | 3 | 31 |
| Dislocation analysis | 32,000 | 0.129 | 0.534 | 4.13 | 56 | 138 |
| Dislocation analysis | 256,000 | 0.813 | 3.746 | 4.61 | 435 | 447 |
| Dislocation analysis | 2,048,000 | 5.807 | 28.577 | 4.92 | 3086 | 3780 |
| Atomic strain, 3 Å | 4,000 | 0.003 | 0.001 | 0.32 | 0 | 0 |
| Atomic strain, 3 Å | 32,000 | 0.004 | 0.003 | 0.77 | 0 | 6 |
| Atomic strain, 3 Å | 256,000 | 0.014 | 0.014 | 1.00 | 46 | 44 |
| Atomic strain, 3 Å | 2,048,000 | 0.073 | 0.085 | 1.16 | 243 | 283 |

#### freud

| Case | Atoms | mdapy 2.0.0.dev0 (s) | freud 3.6.1 (s) | freud / mdapy | mdapy (MB) | freud (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 4,000 | 0.004 | 0.008 | 2.27 | 0 | 23 |
| Neighbor list, 5 Å | 32,000 | 0.007 | 0.062 | 8.83 | 18 | 201 |
| Neighbor list, 5 Å | 256,000 | 0.028 | 0.588 | 20.76 | 195 | 1577 |
| Neighbor list, 5 Å | 2,048,000 | 0.184 | 4.978 | 27.02 | 1427 | 9706 |
| 12 nearest neighbors | 4,000 | 0.002 | 0.002 | 1.04 | 0 | 2 |
| 12 nearest neighbors | 32,000 | 0.005 | 0.021 | 4.01 | 5 | 53 |
| 12 nearest neighbors | 256,000 | 0.032 | 0.185 | 5.73 | 59 | 430 |
| 12 nearest neighbors | 2,048,000 | 0.183 | 1.645 | 8.98 | 413 | 3305 |
| Steinhardt q2–q6, 12 neighbors | 4,000 | 0.001 | 0.003 | 2.13 | 0 | 1 |
| Steinhardt q2–q6, 12 neighbors | 32,000 | 0.006 | 0.022 | 3.73 | 0 | 17 |
| Steinhardt q2–q6, 12 neighbors | 256,000 | 0.045 | 0.185 | 4.08 | 24 | 128 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.350 | 1.727 | 4.93 | 138 | 960 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 4,000 | 0.009 | 0.104 | 11.89 | 3 | 6 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 32,000 | 0.045 | 0.813 | 18.19 | 54 | 61 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.345 | 6.917 | 20.03 | 449 | 464 |
| Radial distribution function, 6 Å, 200 bins | 4,000 | 0.002 | 0.009 | 5.18 | 0 | 30 |
| Radial distribution function, 6 Å, 200 bins | 32,000 | 0.004 | 0.017 | 4.37 | 0 | 34 |
| Radial distribution function, 6 Å, 200 bins | 256,000 | 0.026 | 0.089 | 3.39 | 14 | 61 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.169 | 0.707 | 4.18 | 138 | 215 |
| Voronoi volumes and coordination | 4,000 | 0.005 | 0.072 | 14.15 | 0 | 5 |
| Voronoi volumes and coordination | 32,000 | 0.016 | 0.601 | 38.41 | 0 | 61 |
| Voronoi volumes and coordination | 256,000 | 0.120 | 4.821 | 40.29 | 37 | 464 |
| Voronoi volumes and coordination | 2,048,000 | 0.951 | 38.768 | 40.76 | 195 | 3689 |
| Voronoi cells with their faces | 4,000 | 0.006 | 0.076 | 12.89 | 1 | 6 |
| Voronoi cells with their faces | 32,000 | 0.020 | 0.582 | 28.68 | 23 | 60 |
| Voronoi cells with their faces | 256,000 | 0.137 | 4.850 | 35.34 | 220 | 464 |
| Voronoi cells with their faces | 2,048,000 | 1.123 | 38.797 | 34.55 | 1696 | 3687 |
| Cluster analysis, 2 Å | 3,998 | 0.004 | 0.001 | 0.24 | 0 | 0 |
| Cluster analysis, 2 Å | 31,966 | 0.007 | 0.008 | 1.18 | 3 | 0 |
| Cluster analysis, 2 Å | 255,756 | 0.030 | 0.077 | 2.60 | 40 | 25 |
| Cluster analysis, 2 Å | 2,046,192 | 0.331 | 0.852 | 2.57 | 221 | 251 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 4,000 | 0.006 | 0.012 | 1.97 | 0 | 32 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 32,000 | 0.010 | 0.035 | 3.50 | 20 | 45 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 256,000 | 0.057 | 0.233 | 4.09 | 180 | 140 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 2,048,000 | 0.414 | 1.942 | 4.69 | 1429 | 856 |

## Cases

| Case | System | mdapy | OVITO | freud |
|---|---|---|---|---|
| Neighbor list, 5 Å | fcc Cu | `NeighborList.build(system, 5.0)` | `CutoffNeighborFinder(5.0, data).find_all()` | `AABBQuery(box, points).query(points, {"r_max": 5.0, "exclude_ii": True}).toNeighborList()` |
| 12 nearest neighbors | fcc Cu | `NeighborList.build(system, num_neighbors=12)` | `NearestNeighborFinder(12, data).find_all()` | the same query with `{"num_neighbors": 12, "exclude_ii": True}` |
| Adaptive common neighbor analysis | fcc Cu | `CommonNeighborAnalysis()` | `CommonNeighborAnalysisModifier()` | |
| Common neighbor analysis, cutoff 3.09 Å | fcc Cu | `CommonNeighborAnalysis(3.086)` | `CommonNeighborAnalysisModifier(mode=FixedCutoff, cutoff=3.086)` | |
| Ackland–Jones analysis | fcc Cu | `AcklandJonesAnalysis()` | `AcklandJonesModifier()` | |
| Polyhedral template matching | fcc Cu | `PolyhedralTemplateMatching()` | `PolyhedralTemplateMatchingModifier()` | |
| Centro-symmetry parameter | fcc Cu | `CentroSymmetryParameter(12)` | `CentroSymmetryModifier(num_neighbors=12)` | |
| Diamond structure | diamond Si | `DiamondStructure()` | `IdentifyDiamondModifier()` | |
| Steinhardt q2–q6, 12 neighbors | fcc Cu | `SteinhardtParameters((2, 3, 4, 5, 6), num_neighbors=12)` | `SteinhardtParametersModifier(neighbor_mode=NearestNeighbors, num_neighbors=12, max_l=6)` | `Steinhardt([2, 3, 4, 5, 6])` with `{"num_neighbors": 12, "exclude_ii": True}` |
| Steinhardt q2–q6, Voronoi | fcc Cu | `SteinhardtParameters((2, 3, 4, 5, 6), voronoi=True, weighted=True)` | `SteinhardtParametersModifier(neighbor_mode=Voronoi, max_l=6)` | `Voronoi()`, then `Steinhardt([2, 3, 4, 5, 6], weighted=True)` on its neighbors |
| Radial distribution function | fcc Cu | `RadialDistributionFunction(6.0, bins=200)` | `CoordinationAnalysisModifier(cutoff=6.0, number_of_bins=200)` | `RDF(bins=200, r_max=6.0)` |
| Bond angle distribution | fcc Cu | `BondAngleDistribution(cutoff=3.0, bins=180)` | `CreateBondsModifier(cutoff=3.0)`, `BondAngleDistributionModifier(bins=180)` | |
| Voronoi volumes and coordination | fcc Cu | `VoronoiAnalysis()` | `VoronoiAnalysisModifier()` | `Voronoi()` |
| Voronoi cells with their faces | fcc Cu | `VoronoiAnalysis(faces=True)` | `VoronoiAnalysisModifier(generate_bonds=True)` | `Voronoi()` |
| Cluster analysis, 2 Å | random points | `ClusterAnalysis(cutoff=2.0)` | `ClusterAnalysisModifier(cutoff=2.0, sort_by_size=True)` | `Cluster()` with `{"r_max": 2.0}` |
| Wigner–Seitz occupancies | fcc Cu, 0.5 % vacancies | `WignerSeitz(reference)` | `WignerSeitzAnalysisModifier()` | |
| Atomic strain, 3 Å | strained fcc Cu | `AtomicStrain(reference, 3.0)` | `AtomicStrainModifier(cutoff=3.0)` | |
| Mean squared displacement, 20 frames | fcc Cu walking at random, read from a dump | `MeanSquaredDisplacement(unwrap=False).compute(read_trajectory(path))` | `DiffusionModifier()` over the frames of the file | |
| Dislocation analysis | fcc Cu with three dislocation loops | `DislocationAnalysis("fcc")` | `DislocationAnalysisModifier()` | |
| Six analyses of copper, one neighbor search | fcc Cu | `NeighborList.build(system, 5.0)` given to the analyses of adaptive common neighbors, Ackland–Jones, centro-symmetry, Steinhardt (12 neighbors), the radial distribution function within 5 Å and bond angles within 3 Å | one pipeline of the six modifiers | |
| Steinhardt q2–q6 and the RDF, one neighbor search | fcc Cu | `NeighborList.build(system, 5.0)` given to both | one pipeline of the two modifiers | one `AABBQuery` for both |

The fcc copper has its atoms displaced at random by 0.1 Å along each axis,
as at room temperature; the silicon by 0.05 Å. The random points have the
density of copper and, with a cutoff of 2 Å, are close to percolation, so
that clusters of every size form; points with a neighbor within 1e-4 Å of
2 Å are left out, so that single precision gives the same bonds. OVITO's
coordination analysis also counts the neighbors of each atom along with the
radial distribution function.

## How it is measured

- Each program runs in a process of its own, with the same number of
  threads (`MDAPY_NUM_THREADS`, `OVITO_THREAD_COUNT` and
  `freud.parallel.set_num_threads`).
- Before the timings, a separate process computes the case with mdapy and
  with the other program and compares the results. With OVITO, which
  computes in double precision as mdapy does: the same structure types,
  cluster sizes, occupancies and bins, floating-point values within 1e-10
  (1e-8 for Steinhardt parameters weighted by Voronoi faces, whose areas
  come from two tessellations), for neighbor lists the same number of
  neighbors of each atom and the same sum of their distances, and for
  Voronoi cells the same volumes and numbers of faces, and with the faces
  as many of OVITO's bonds at each atom, but for faces of no area that one
  tessellation of millions of atoms finds and the other not (at most one
  atom in 100 000). freud computes in single precision, which
  rounds positions in a box of 600 Å by some 3e-5 Å: values agree within
  1e-4; the numbers of neighbors and the bins of the radial distribution
  function within the pairs closer than 1e-4 Å to a cutoff or the edge of a
  bin; Steinhardt parameters of the 12 nearest agree with those mdapy
  computes from the positions freud has, within 3e-4, for the atoms whose
  12th and 13th neighbors are farther apart than 1e-4 Å, nearly all (freud's
  harmonics in single precision differ by 2e-5 for all but a few atoms in a
  million, by up to 2e-4 for those); every Voronoi face of freud above
  0.01 Å² is one of mdapy, and the faces freud leaves out, of distant
  neighbors, are all below 0.02 Å², so its Steinhardt parameters over Voronoi faces agree within 1e-3.
  Dislocation analysis gives the same structures and a total length of
  the lines within 3 % (the circuits of the two programs choose
  differently); the mean squared displacements agree within 1e-10. A
  program that differs is not timed.
- Each process first computes the case for 500 atoms, so that loading
  libraries and starting threads are not timed, then three times for the
  system; the shortest time counts.
- The time runs from the call to the result, with every neighbor search.
  Making the system and converting it into OVITO's data collection or
  freud's box and points are not timed. OVITO computes a new pipeline each
  time, so nothing is cached.
- A process that grows beyond 170 GB or runs for more than 15 minutes is
  stopped, and a size is skipped where the previous one predicts more
  memory.

## Running it

The comparison is `benchmarks/comparison.py` in the source repository. It
needs OVITO (`pip install ovito`), freud (`pip install freud-analysis`)
and, for the figure, matplotlib.

```bash
python benchmarks/comparison.py run                     # every case
python benchmarks/comparison.py run cna voronoi --sizes 4000 32000 --threads 1
python benchmarks/comparison.py run --sizes 4000 32000 256000 2048000 \
    --threads 40 --memory 170 --timeout 900                # as above
python benchmarks/comparison.py table results.json      # Markdown tables
python benchmarks/comparison.py plot results.json ratio.png
```

`run` appends one line of JSON per case and size to `results.json` (the
option `--output`): the machine, the versions, the agreement of each
program, and the times and peak memory of each. A new case is a `Case` in
`_cases()`: how to make its system, the mdapy call, and for each other
program its call and how to compare its result with that of mdapy.
