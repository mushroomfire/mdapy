# mdapy, OVITO and freud

The analyses that mdapy shares with [OVITO](https://www.ovito.org) and
[freud](https://freud.readthedocs.io), computed on the same systems with
the same parameters, from 4 000 to 2 million atoms. A program is timed only
where it gives the result of mdapy, atom by atom.

## Results

A node of the Triton cluster of Aalto University, alone for the run: two
Intel Xeon Gold 6248 (40 cores, two NUMA nodes), 191 GB, Linux. Python 3.13,
mdapy 2.0.0.dev0 built with GCC 12.3, OVITO 3.17.0 and freud 3.6.1 from
pip, each with 40 threads.

![Time of OVITO and of freud over the time of mdapy for each case and size](images/comparison.png)

### What they show

- **OVITO**: from 256 000 atoms on, mdapy is faster in every analysis they
  share. It builds neighbor lists 7 times faster in a quarter of the
  memory, Voronoi cells 23 to 26 times faster, Steinhardt parameters over
  Voronoi faces 33 to 67 times and cluster analysis 6 times; the other
  analyses take 1.4 to 3.6 times less time. Atomic strain is the closest:
  OVITO is faster below some 200 000 atoms (1.7 times at 4 000 and 1.4
  times at 32 000), mdapy above (1.05 times at 256 000, 1.5 times at 2
  million).
- **freud**: from 32 000 atoms on, mdapy is faster in every analysis:
  neighbor lists 8 to 26 times, in a seventh of the memory; the 12 nearest
  neighbors 2.8 to 9 times; Voronoi cells 35 to 42 times; Steinhardt
  parameters and the radial distribution function 3.4 to 4.5 times;
  cluster analysis 1.8 to 2.8 times.
- **One neighbor search for several analyses**: the six analyses of copper
  from one list take 0.81 s at 2 million atoms, against 1.48 s for the six
  cases timed apart (0.11 s against 0.21 s at 256 000; the radial
  distribution function apart reaches 6 Å instead of 5 Å). OVITO's pipeline
  of the six takes about the sum of its cases (2.93 s against 3.04 s).
- **Voronoi faces on demand**: the volumes and coordination of 2 million
  cells take mdapy 0.92 s and 130 MB; with every face, its neighbor, area
  and edges, 1.08 s and 1.6 GB.
- **4 000 atoms** take a few milliseconds; mdapy is faster there in every
  analysis but cluster analysis, Wigner–Seitz occupancies and atomic
  strain against OVITO, and cluster analysis against freud (0.001 s
  against 0.004 s); the 12 nearest neighbors are on par with freud.
- **Not run**: Steinhardt parameters over Voronoi faces at 2 million atoms,
  which took OVITO 678 s in an earlier run.

### Times and memory

Every case at every size: the shortest of three times, and the memory each
computation took beyond the system, the peak resident memory of the
process minus its peak before the timings.

#### OVITO

| Case | Atoms | mdapy 2.0.0.dev0 (s) | OVITO 3.17.0 (s) | OVITO / mdapy | mdapy (MB) | OVITO (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 4,000 | 0.004 | 0.008 | 1.95 | 0 | 10 |
| Neighbor list, 5 Å | 32,000 | 0.008 | 0.033 | 4.15 | 19 | 129 |
| Neighbor list, 5 Å | 256,000 | 0.026 | 0.181 | 6.88 | 173 | 722 |
| Neighbor list, 5 Å | 2,048,000 | 0.190 | 1.346 | 7.10 | 1356 | 5300 |
| 12 nearest neighbors | 4,000 | 0.002 | 0.008 | 3.56 | 0 | 1 |
| 12 nearest neighbors | 32,000 | 0.007 | 0.012 | 1.65 | 3 | 12 |
| 12 nearest neighbors | 256,000 | 0.029 | 0.050 | 1.72 | 49 | 100 |
| 12 nearest neighbors | 2,048,000 | 0.183 | 0.441 | 2.41 | 343 | 751 |
| Adaptive common neighbor analysis | 4,000 | 0.002 | 0.015 | 7.75 | 0 | 0 |
| Adaptive common neighbor analysis | 32,000 | 0.004 | 0.020 | 4.65 | 0 | 8 |
| Adaptive common neighbor analysis | 256,000 | 0.039 | 0.075 | 1.95 | 3 | 50 |
| Adaptive common neighbor analysis | 2,048,000 | 0.215 | 0.598 | 2.78 | 66 | 119 |
| Common neighbor analysis, cutoff 3.09 Å | 4,000 | 0.003 | 0.007 | 2.16 | 0 | 0 |
| Common neighbor analysis, cutoff 3.09 Å | 32,000 | 0.002 | 0.009 | 6.07 | 0 | 5 |
| Common neighbor analysis, cutoff 3.09 Å | 256,000 | 0.010 | 0.023 | 2.25 | 16 | 29 |
| Common neighbor analysis, cutoff 3.09 Å | 2,048,000 | 0.075 | 0.158 | 2.09 | 137 | 31 |
| Ackland–Jones analysis | 4,000 | 0.002 | 0.004 | 2.15 | 0 | 1 |
| Ackland–Jones analysis | 32,000 | 0.006 | 0.010 | 1.70 | 0 | 9 |
| Ackland–Jones analysis | 256,000 | 0.039 | 0.068 | 1.76 | 4 | 50 |
| Ackland–Jones analysis | 2,048,000 | 0.282 | 0.563 | 2.00 | 65 | 118 |
| Polyhedral template matching | 4,000 | 0.013 | 0.026 | 1.99 | 0 | 1 |
| Polyhedral template matching | 32,000 | 0.031 | 0.041 | 1.31 | 13 | 10 |
| Polyhedral template matching | 256,000 | 0.197 | 0.278 | 1.42 | 80 | 36 |
| Polyhedral template matching | 2,048,000 | 1.532 | 2.228 | 1.45 | 507 | 177 |
| Centro-symmetry parameter, 12 neighbors | 4,000 | 0.002 | 0.011 | 6.19 | 0 | 0 |
| Centro-symmetry parameter, 12 neighbors | 32,000 | 0.005 | 0.015 | 3.24 | 0 | 5 |
| Centro-symmetry parameter, 12 neighbors | 256,000 | 0.035 | 0.058 | 1.67 | 3 | 42 |
| Centro-symmetry parameter, 12 neighbors | 2,048,000 | 0.248 | 0.478 | 1.93 | 73 | 124 |
| Diamond structure | 4,096 | 0.002 | 0.004 | 1.69 | 0 | 2 |
| Diamond structure | 32,768 | 0.004 | 0.010 | 2.47 | 0 | 21 |
| Diamond structure | 262,144 | 0.028 | 0.070 | 2.49 | 13 | 80 |
| Diamond structure | 2,000,376 | 0.157 | 0.517 | 3.30 | 112 | 270 |
| Steinhardt q2–q6, 12 neighbors | 4,000 | 0.001 | 0.014 | 10.37 | 0 | 0 |
| Steinhardt q2–q6, 12 neighbors | 32,000 | 0.007 | 0.017 | 2.56 | 0 | 10 |
| Steinhardt q2–q6, 12 neighbors | 256,000 | 0.044 | 0.063 | 1.42 | 14 | 57 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.366 | 0.513 | 1.40 | 66 | 79 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 4,000 | 0.009 | 0.296 | 32.61 | 2 | 0 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 32,000 | 0.043 | 1.401 | 32.87 | 56 | 12 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.335 | 22.583 | 67.32 | 423 | 41 |
| Radial distribution function, 6 Å, 200 bins | 4,000 | 0.004 | 0.005 | 1.19 | 0 | 0 |
| Radial distribution function, 6 Å, 200 bins | 32,000 | 0.004 | 0.009 | 2.18 | 0 | 4 |
| Radial distribution function, 6 Å, 200 bins | 256,000 | 0.022 | 0.038 | 1.76 | 11 | 19 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.169 | 0.295 | 1.75 | 73 | 0 |
| Bond angle distribution, 3 Å, 180 bins | 4,000 | 0.004 | 0.014 | 3.37 | 0 | 2 |
| Bond angle distribution, 3 Å, 180 bins | 32,000 | 0.004 | 0.021 | 5.40 | 0 | 37 |
| Bond angle distribution, 3 Å, 180 bins | 256,000 | 0.031 | 0.078 | 2.53 | 18 | 273 |
| Bond angle distribution, 3 Å, 180 bins | 2,048,000 | 0.197 | 0.588 | 2.98 | 167 | 1328 |
| Voronoi volumes and coordination | 4,000 | 0.005 | 0.047 | 9.47 | 0 | 0 |
| Voronoi volumes and coordination | 32,000 | 0.015 | 0.371 | 23.95 | 0 | 2 |
| Voronoi volumes and coordination | 256,000 | 0.119 | 2.958 | 24.95 | 29 | 7 |
| Voronoi volumes and coordination | 2,048,000 | 0.923 | 23.841 | 25.83 | 130 | 22 |
| Voronoi cells with their faces | 4,000 | 0.003 | 0.050 | 17.11 | 0 | 2 |
| Voronoi cells with their faces | 32,000 | 0.017 | 0.392 | 22.71 | 26 | 18 |
| Voronoi cells with their faces | 256,000 | 0.131 | 3.151 | 23.99 | 206 | 145 |
| Voronoi cells with their faces | 2,048,000 | 1.083 | 25.281 | 23.34 | 1595 | 943 |
| Cluster analysis, 2 Å | 3,998 | 0.004 | 0.003 | 0.71 | 0 | 0 |
| Cluster analysis, 2 Å | 31,966 | 0.004 | 0.017 | 4.31 | 1 | 6 |
| Cluster analysis, 2 Å | 255,756 | 0.028 | 0.163 | 5.85 | 39 | 28 |
| Cluster analysis, 2 Å | 2,046,192 | 0.341 | 2.066 | 6.06 | 222 | 0 |
| Wigner–Seitz occupancies | 3,974 | 0.001 | 0.001 | 0.70 | 0 | 1 |
| Wigner–Seitz occupancies | 31,811 | 0.002 | 0.006 | 2.99 | 0 | 8 |
| Wigner–Seitz occupancies | 254,734 | 0.017 | 0.036 | 2.16 | 2 | 36 |
| Wigner–Seitz occupancies | 2,037,841 | 0.096 | 0.316 | 3.31 | 47 | 147 |
| Six analyses of copper, one neighbor search | 4,000 | 0.009 | 0.064 | 7.52 | 1 | 8 |
| Six analyses of copper, one neighbor search | 32,000 | 0.017 | 0.095 | 5.49 | 20 | 72 |
| Six analyses of copper, one neighbor search | 256,000 | 0.109 | 0.370 | 3.41 | 174 | 492 |
| Six analyses of copper, one neighbor search | 2,048,000 | 0.813 | 2.933 | 3.61 | 1340 | 1697 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 4,000 | 0.003 | 0.019 | 6.86 | 1 | 1 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 32,000 | 0.009 | 0.026 | 2.86 | 20 | 12 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 256,000 | 0.058 | 0.088 | 1.52 | 188 | 85 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 2,048,000 | 0.420 | 0.720 | 1.71 | 1340 | 134 |
| Atomic strain, 3 Å | 4,000 | 0.004 | 0.002 | 0.59 | 0 | 0 |
| Atomic strain, 3 Å | 32,000 | 0.005 | 0.003 | 0.73 | 0 | 6 |
| Atomic strain, 3 Å | 256,000 | 0.012 | 0.013 | 1.05 | 46 | 45 |
| Atomic strain, 3 Å | 2,048,000 | 0.063 | 0.095 | 1.52 | 232 | 236 |

#### freud

| Case | Atoms | mdapy 2.0.0.dev0 (s) | freud 3.6.1 (s) | freud / mdapy | mdapy (MB) | freud (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 4,000 | 0.004 | 0.007 | 1.66 | 0 | 22 |
| Neighbor list, 5 Å | 32,000 | 0.008 | 0.061 | 7.70 | 19 | 201 |
| Neighbor list, 5 Å | 256,000 | 0.026 | 0.591 | 22.41 | 173 | 1586 |
| Neighbor list, 5 Å | 2,048,000 | 0.190 | 4.946 | 26.07 | 1356 | 9668 |
| 12 nearest neighbors | 4,000 | 0.002 | 0.002 | 0.98 | 0 | 2 |
| 12 nearest neighbors | 32,000 | 0.007 | 0.020 | 2.78 | 3 | 52 |
| 12 nearest neighbors | 256,000 | 0.029 | 0.178 | 6.12 | 49 | 433 |
| 12 nearest neighbors | 2,048,000 | 0.183 | 1.628 | 8.90 | 343 | 3267 |
| Steinhardt q2–q6, 12 neighbors | 4,000 | 0.001 | 0.004 | 2.68 | 0 | 0 |
| Steinhardt q2–q6, 12 neighbors | 32,000 | 0.007 | 0.023 | 3.37 | 0 | 16 |
| Steinhardt q2–q6, 12 neighbors | 256,000 | 0.044 | 0.188 | 4.23 | 14 | 126 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.366 | 1.655 | 4.53 | 66 | 909 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 4,000 | 0.009 | 0.103 | 11.41 | 2 | 7 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 32,000 | 0.043 | 0.813 | 19.06 | 56 | 61 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.335 | 6.936 | 20.68 | 423 | 461 |
| Radial distribution function, 6 Å, 200 bins | 4,000 | 0.004 | 0.008 | 1.90 | 0 | 30 |
| Radial distribution function, 6 Å, 200 bins | 32,000 | 0.004 | 0.017 | 4.26 | 0 | 35 |
| Radial distribution function, 6 Å, 200 bins | 256,000 | 0.022 | 0.084 | 3.90 | 11 | 54 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.169 | 0.711 | 4.22 | 73 | 208 |
| Voronoi volumes and coordination | 4,000 | 0.005 | 0.072 | 14.47 | 0 | 5 |
| Voronoi volumes and coordination | 32,000 | 0.015 | 0.598 | 38.59 | 0 | 61 |
| Voronoi volumes and coordination | 256,000 | 0.119 | 4.811 | 40.58 | 29 | 461 |
| Voronoi volumes and coordination | 2,048,000 | 0.923 | 38.658 | 41.89 | 130 | 3651 |
| Voronoi cells with their faces | 4,000 | 0.003 | 0.072 | 24.61 | 0 | 7 |
| Voronoi cells with their faces | 32,000 | 0.017 | 0.598 | 34.64 | 26 | 61 |
| Voronoi cells with their faces | 256,000 | 0.131 | 4.851 | 36.93 | 206 | 461 |
| Voronoi cells with their faces | 2,048,000 | 1.083 | 38.538 | 35.58 | 1595 | 3651 |
| Cluster analysis, 2 Å | 3,998 | 0.004 | 0.001 | 0.25 | 0 | 0 |
| Cluster analysis, 2 Å | 31,966 | 0.004 | 0.007 | 1.83 | 1 | 0 |
| Cluster analysis, 2 Å | 255,756 | 0.028 | 0.078 | 2.78 | 39 | 27 |
| Cluster analysis, 2 Å | 2,046,192 | 0.341 | 0.860 | 2.52 | 222 | 252 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 4,000 | 0.003 | 0.011 | 4.01 | 1 | 30 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 32,000 | 0.009 | 0.036 | 4.01 | 20 | 44 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 256,000 | 0.058 | 0.239 | 4.13 | 188 | 143 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 2,048,000 | 0.420 | 1.910 | 4.54 | 1340 | 848 |

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
  neighbors, are all below 0.02 Å², so its Steinhardt parameters over Voronoi faces agree within 1e-3. A
  program that differs is not timed.
- Each process first computes the case for 500 atoms, so that loading
  libraries and starting threads are not timed, then three times for the
  system; the shortest time counts.
- The run gave common neighbor analysis with a fixed cutoff and atomic
  strain at 2 million atoms twice the time mdapy took in every other run,
  so four cases (these two, cluster analysis and neighbor lists) at 4 000
  and 2 million atoms were run twice more, for every program, in a second
  job on the same kind of node; the tables show the last of these runs,
  which agreed with the one before within 10 %.
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
