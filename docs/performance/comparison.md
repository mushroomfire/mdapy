# mdapy, OVITO and freud

The analyses that mdapy shares with [OVITO](https://www.ovito.org) and
[freud](https://freud.readthedocs.io), computed on the same systems with
the same parameters, from 4 000 to 2 million atoms. A program is timed only
where it gives the result of mdapy, atom by atom.

## Results

A node of the Triton cluster of Aalto University, alone for the run: two
Intel Xeon Gold 6230 (40 cores, two NUMA nodes), 191 GB, Linux. Python 3.13,
mdapy 2.0.0.dev0 built with GCC 12.3, OVITO 3.17.0 and freud 3.6.1 from
pip, each with 40 threads.

![Time of OVITO and of freud over the time of mdapy for each case and size](images/comparison.png)

### What they show

- **OVITO**: from 256 000 atoms on, mdapy is faster in every analysis they
  share. It builds neighbor lists 4 to 6 times faster in a quarter of the
  memory, Voronoi cells 24 times faster, Steinhardt parameters over Voronoi
  faces 37 to 73 times and cluster analysis 6 times; the other analyses
  take 1.5 to 3.8 times less time. Atomic strain is the closest: OVITO is
  faster below some 100 000 atoms (twice at 4 000, 1.1 times at 32 000),
  mdapy above (1.06 times at 256 000, 1.2 times at 2 million).
- **freud**: mdapy is faster in every analysis at every size: neighbor
  lists 9 to 23 times from 32 000 atoms on, in a seventh of the memory; the
  12 nearest neighbors 4 to 8 times; Voronoi cells 38 times; Steinhardt
  parameters and the radial distribution function 2.8 to 4.6 times; cluster
  analysis 2.4 to 3 times.
- **One neighbor search for several analyses**: the six analyses of copper
  from one list take 1.21 s at 2 million atoms, against 2.07 s for the six
  cases timed apart (0.16 s against 0.29 s at 256 000; the radial
  distribution function apart reaches 6 Å instead of 5 Å). OVITO's pipeline
  of the six takes about the sum of its cases (4.15 s against 4.35 s).
- **Memory**: mdapy's Voronoi analysis returns every face of every cell,
  with the atom across it, its area and its edges (1.6 GB at 2 million
  atoms); OVITO's keeps only the volumes and coordination unless asked for
  bonds or polyhedra.
- **4 000 atoms** take a few milliseconds; mdapy is faster there in every
  analysis but atomic strain.
- **Not run**: Steinhardt parameters over Voronoi faces at 2 million atoms,
  which took OVITO 678 s in an earlier run.

### Times and memory

Every case at every size: the shortest of three times, and the memory each
computation took beyond the system, the peak resident memory of the
process minus its peak before the timings.

#### OVITO

| Case | Atoms | mdapy 2.0.0.dev0 (s) | OVITO 3.17.0 (s) | OVITO / mdapy | mdapy (MB) | OVITO (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 4,000 | 0.004 | 0.016 | 3.96 | 0 | 10 |
| Neighbor list, 5 Å | 32,000 | 0.009 | 0.042 | 4.70 | 18 | 92 |
| Neighbor list, 5 Å | 256,000 | 0.038 | 0.216 | 5.70 | 186 | 758 |
| Neighbor list, 5 Å | 2,048,000 | 0.272 | 1.655 | 6.08 | 1362 | 5324 |
| 12 nearest neighbors | 4,000 | 0.003 | 0.015 | 5.01 | 0 | 0 |
| 12 nearest neighbors | 32,000 | 0.007 | 0.019 | 2.80 | 2 | 12 |
| 12 nearest neighbors | 256,000 | 0.042 | 0.076 | 1.81 | 49 | 100 |
| 12 nearest neighbors | 2,048,000 | 0.252 | 0.647 | 2.57 | 343 | 751 |
| Adaptive common neighbor analysis | 4,000 | 0.001 | 0.027 | 20.06 | 0 | 0 |
| Adaptive common neighbor analysis | 32,000 | 0.007 | 0.033 | 4.98 | 0 | 6 |
| Adaptive common neighbor analysis | 256,000 | 0.049 | 0.108 | 2.22 | 3 | 45 |
| Adaptive common neighbor analysis | 2,048,000 | 0.311 | 0.862 | 2.77 | 65 | 118 |
| Common neighbor analysis, cutoff 3.09 Å | 4,000 | 0.004 | 0.011 | 3.08 | 0 | 0 |
| Common neighbor analysis, cutoff 3.09 Å | 32,000 | 0.005 | 0.015 | 2.90 | 0 | 6 |
| Common neighbor analysis, cutoff 3.09 Å | 256,000 | 0.016 | 0.035 | 2.11 | 16 | 28 |
| Common neighbor analysis, cutoff 3.09 Å | 2,048,000 | 0.104 | 0.228 | 2.19 | 137 | 31 |
| Ackland–Jones analysis | 4,000 | 0.002 | 0.007 | 3.64 | 0 | 1 |
| Ackland–Jones analysis | 32,000 | 0.008 | 0.019 | 2.44 | 0 | 8 |
| Ackland–Jones analysis | 256,000 | 0.057 | 0.100 | 1.77 | 5 | 50 |
| Ackland–Jones analysis | 2,048,000 | 0.392 | 0.823 | 2.10 | 67 | 119 |
| Polyhedral template matching | 4,000 | 0.020 | 0.043 | 2.15 | 0 | 1 |
| Polyhedral template matching | 32,000 | 0.041 | 0.054 | 1.31 | 13 | 10 |
| Polyhedral template matching | 256,000 | 0.265 | 0.392 | 1.48 | 79 | 33 |
| Polyhedral template matching | 2,048,000 | 1.991 | 3.153 | 1.58 | 507 | 177 |
| Centro-symmetry parameter, 12 neighbors | 4,000 | 0.003 | 0.018 | 5.94 | 0 | 0 |
| Centro-symmetry parameter, 12 neighbors | 32,000 | 0.008 | 0.024 | 2.99 | 0 | 8 |
| Centro-symmetry parameter, 12 neighbors | 256,000 | 0.047 | 0.089 | 1.90 | 3 | 42 |
| Centro-symmetry parameter, 12 neighbors | 2,048,000 | 0.303 | 0.707 | 2.33 | 73 | 125 |
| Diamond structure | 4,096 | 0.003 | 0.005 | 1.55 | 0 | 2 |
| Diamond structure | 32,768 | 0.008 | 0.014 | 1.92 | 0 | 21 |
| Diamond structure | 262,144 | 0.038 | 0.099 | 2.59 | 13 | 80 |
| Diamond structure | 2,000,376 | 0.213 | 0.739 | 3.46 | 100 | 270 |
| Steinhardt q2–q6, 12 neighbors | 4,000 | 0.003 | 0.024 | 7.81 | 0 | 0 |
| Steinhardt q2–q6, 12 neighbors | 32,000 | 0.009 | 0.028 | 3.18 | 1 | 10 |
| Steinhardt q2–q6, 12 neighbors | 256,000 | 0.061 | 0.094 | 1.54 | 15 | 58 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.495 | 0.755 | 1.53 | 70 | 111 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 4,000 | 0.009 | 0.545 | 62.54 | 2 | 1 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 32,000 | 0.067 | 2.447 | 36.76 | 50 | 12 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.524 | 38.419 | 73.29 | 433 | 36 |
| Radial distribution function, 6 Å, 200 bins | 4,000 | 0.005 | 0.009 | 1.89 | 0 | 0 |
| Radial distribution function, 6 Å, 200 bins | 32,000 | 0.008 | 0.012 | 1.48 | 0 | 4 |
| Radial distribution function, 6 Å, 200 bins | 256,000 | 0.032 | 0.055 | 1.73 | 23 | 18 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.245 | 0.433 | 1.77 | 73 | 0 |
| Bond angle distribution, 3 Å, 180 bins | 4,000 | 0.004 | 0.024 | 5.90 | 0 | 2 |
| Bond angle distribution, 3 Å, 180 bins | 32,000 | 0.008 | 0.032 | 4.04 | 0 | 35 |
| Bond angle distribution, 3 Å, 180 bins | 256,000 | 0.045 | 0.102 | 2.26 | 18 | 255 |
| Bond angle distribution, 3 Å, 180 bins | 2,048,000 | 0.324 | 0.768 | 2.37 | 166 | 1352 |
| Voronoi volumes and coordination | 4,000 | 0.007 | 0.083 | 11.81 | 1 | 0 |
| Voronoi volumes and coordination | 32,000 | 0.027 | 0.663 | 24.88 | 21 | 2 |
| Voronoi volumes and coordination | 256,000 | 0.223 | 5.310 | 23.78 | 205 | 8 |
| Voronoi volumes and coordination | 2,048,000 | 1.754 | 42.578 | 24.28 | 1583 | 21 |
| Cluster analysis, 2 Å | 3,998 | 0.001 | 0.004 | 3.26 | 0 | 0 |
| Cluster analysis, 2 Å | 31,966 | 0.005 | 0.030 | 5.73 | 2 | 6 |
| Cluster analysis, 2 Å | 255,756 | 0.041 | 0.251 | 6.10 | 27 | 28 |
| Cluster analysis, 2 Å | 2,046,192 | 0.439 | 2.804 | 6.39 | 219 | 0 |
| Wigner–Seitz occupancies | 3,974 | 0.002 | 0.002 | 1.02 | 0 | 1 |
| Wigner–Seitz occupancies | 31,811 | 0.004 | 0.008 | 2.15 | 0 | 8 |
| Wigner–Seitz occupancies | 254,734 | 0.020 | 0.055 | 2.76 | 3 | 36 |
| Wigner–Seitz occupancies | 2,037,841 | 0.122 | 0.470 | 3.84 | 17 | 148 |
| Six analyses of copper, one neighbor search | 4,000 | 0.010 | 0.110 | 10.61 | 1 | 7 |
| Six analyses of copper, one neighbor search | 32,000 | 0.025 | 0.149 | 6.01 | 20 | 84 |
| Six analyses of copper, one neighbor search | 256,000 | 0.157 | 0.533 | 3.40 | 187 | 462 |
| Six analyses of copper, one neighbor search | 2,048,000 | 1.210 | 4.152 | 3.43 | 1339 | 1720 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 4,000 | 0.007 | 0.032 | 4.45 | 1 | 1 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 32,000 | 0.015 | 0.038 | 2.48 | 20 | 12 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 256,000 | 0.081 | 0.132 | 1.62 | 174 | 73 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 2,048,000 | 0.592 | 1.031 | 1.74 | 1340 | 93 |
| Atomic strain, 3 Å | 4,000 | 0.004 | 0.002 | 0.51 | 0 | 0 |
| Atomic strain, 3 Å | 32,000 | 0.005 | 0.004 | 0.91 | 1 | 6 |
| Atomic strain, 3 Å | 256,000 | 0.016 | 0.017 | 1.06 | 46 | 44 |
| Atomic strain, 3 Å | 2,048,000 | 0.090 | 0.108 | 1.20 | 232 | 269 |

#### freud

| Case | Atoms | mdapy 2.0.0.dev0 (s) | freud 3.6.1 (s) | freud / mdapy | mdapy (MB) | freud (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Neighbor list, 5 Å | 4,000 | 0.004 | 0.009 | 2.26 | 0 | 24 |
| Neighbor list, 5 Å | 32,000 | 0.009 | 0.082 | 9.19 | 18 | 197 |
| Neighbor list, 5 Å | 256,000 | 0.038 | 0.772 | 20.40 | 186 | 1567 |
| Neighbor list, 5 Å | 2,048,000 | 0.272 | 6.180 | 22.70 | 1362 | 9667 |
| 12 nearest neighbors | 4,000 | 0.003 | 0.003 | 1.04 | 0 | 2 |
| 12 nearest neighbors | 32,000 | 0.007 | 0.027 | 3.93 | 2 | 51 |
| 12 nearest neighbors | 256,000 | 0.042 | 0.247 | 5.86 | 49 | 431 |
| 12 nearest neighbors | 2,048,000 | 0.252 | 2.125 | 8.45 | 343 | 3268 |
| Steinhardt q2–q6, 12 neighbors | 4,000 | 0.003 | 0.004 | 1.30 | 0 | 1 |
| Steinhardt q2–q6, 12 neighbors | 32,000 | 0.009 | 0.031 | 3.52 | 1 | 15 |
| Steinhardt q2–q6, 12 neighbors | 256,000 | 0.061 | 0.268 | 4.36 | 15 | 123 |
| Steinhardt q2–q6, 12 neighbors | 2,048,000 | 0.495 | 2.267 | 4.58 | 70 | 908 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 4,000 | 0.009 | 0.164 | 18.80 | 2 | 7 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 32,000 | 0.067 | 1.374 | 20.64 | 50 | 61 |
| Steinhardt q2–q6, Voronoi faces weighted by area | 256,000 | 0.524 | 10.567 | 20.16 | 433 | 461 |
| Radial distribution function, 6 Å, 200 bins | 4,000 | 0.005 | 0.010 | 2.02 | 0 | 0 |
| Radial distribution function, 6 Å, 200 bins | 32,000 | 0.008 | 0.022 | 2.79 | 0 | 6 |
| Radial distribution function, 6 Å, 200 bins | 256,000 | 0.032 | 0.123 | 3.83 | 23 | 56 |
| Radial distribution function, 6 Å, 200 bins | 2,048,000 | 0.245 | 1.046 | 4.26 | 73 | 239 |
| Voronoi volumes and coordination | 4,000 | 0.007 | 0.129 | 18.45 | 1 | 7 |
| Voronoi volumes and coordination | 32,000 | 0.027 | 1.051 | 39.43 | 21 | 60 |
| Voronoi volumes and coordination | 256,000 | 0.223 | 8.399 | 37.62 | 205 | 461 |
| Voronoi volumes and coordination | 2,048,000 | 1.754 | 67.523 | 38.51 | 1583 | 3650 |
| Cluster analysis, 2 Å | 3,998 | 0.001 | 0.001 | 1.14 | 0 | 0 |
| Cluster analysis, 2 Å | 31,966 | 0.005 | 0.012 | 2.36 | 2 | 0 |
| Cluster analysis, 2 Å | 255,756 | 0.041 | 0.123 | 2.99 | 27 | 26 |
| Cluster analysis, 2 Å | 2,046,192 | 0.439 | 1.217 | 2.77 | 219 | 249 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 4,000 | 0.007 | 0.014 | 1.89 | 1 | 0 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 32,000 | 0.015 | 0.047 | 3.02 | 20 | 18 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 256,000 | 0.081 | 0.317 | 3.91 | 174 | 142 |
| Steinhardt q2–q6 and the RDF, one neighbor search | 2,048,000 | 0.592 | 2.588 | 4.37 | 1340 | 848 |

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
  Voronoi cells the same volumes and numbers of faces, but for faces of no
  area that one tessellation of millions of atoms finds and the other not
  (at most one atom in 100 000). freud computes in single precision, which
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
