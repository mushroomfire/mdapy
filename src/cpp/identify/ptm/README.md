# PTM

The polyhedral template matching library of P. M. Larsen (MIT license, see
LICENSE), as OVITO ships it in `src/3rdparty/ptm` (OVITO commit 692fc68e,
20 October 2026). mdapy calls it from `identify/ptm.cpp`.

Changes, none of which alters a result:

- `ptm_voronoi_cell.cpp`: errors of the Voronoi code throw
  `std::runtime_error` instead of ending the process with `exit`.
- `ptm_voronoi_cell.cpp`, `ptm_voronoi_cell.h`, `ptm_neighbour_ordering.cpp`:
  variables that shadowed others renamed, and statements after a `for` on
  one line moved to their own lines, for the warnings mdapy builds with.
- `ptm_index.cpp`, `ptm_initialize_data.cpp`: unused variables marked.

Larsen, Schmidt and Schiøtz, Robust structural identification via
polyhedral template matching, Modelling Simul. Mater. Sci. Eng. 24, 055007
(2016).
