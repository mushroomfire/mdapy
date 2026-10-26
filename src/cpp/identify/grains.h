// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Grain segmentation: the grains of a polycrystal, groups of crystalline
// atoms of nearly the same lattice orientation, merged step by step from
// the orientations polyhedral template matching gives, as OVITO's Grain
// segmentation modifier does.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::identify {

struct GrainOptions {
    // merge by a minimum spanning tree of the misorientations instead of
    // the graph clustering by node pair sampling
    bool spanning_tree = false;
    // graph clustering: choose the threshold from the merges
    bool automatic = true;
    // the merges kept: graph clustering, the largest log merge distance;
    // spanning tree, the largest misorientation in degrees
    double threshold = 0.0;
    std::int64_t min_size = 100;
    // atoms of coherent fcc/hcp and cubic/hexagonal diamond interfaces, as
    // stacking faults and twin boundaries, join the grains around them
    bool interfaces = true;
    // atoms in no grain join the nearest grain through their neighbors
    bool adopt_orphans = true;
};

struct GrainInput {
    std::int64_t n = 0;
    const std::int8_t* structure = nullptr;  // PTM types (kPtm...)
    const double* orientation = nullptr;     // n x 4: x, y, z, w
    // the neighbors of atom i, offsets[i] to offsets[i + 1]: in the order of
    // its template, or nearest first for an atom of no structure, with the
    // length of each bond
    const std::int64_t* offsets = nullptr;
    const index_t* neighbors = nullptr;
    const double* lengths = nullptr;
    // for the centers of the grains
    const double* positions = nullptr;  // n x 3
    const double* matrix = nullptr;     // rows a, b, c
    const double* origin = nullptr;     // 3
    const bool* pbc = nullptr;          // 3
    const double* masses = nullptr;     // n, or nullptr for equal weights
};

struct GrainOutput {
    std::vector<std::int64_t> grain;  // of each atom, -1 for none
    // of each grain, largest first
    std::vector<std::int64_t> size;
    std::vector<std::int8_t> structure;
    std::vector<double> orientation;  // x, y, z, w
    std::vector<double> center;       // 3 each, of mass, in the box
    std::vector<double> radius;       // of the smallest sphere about the center holding the grain
    std::vector<double> gyration;     // radius of gyration
    // the merges of clusters both of at least 20 atoms: the merge distance
    // (graph) or misorientation in degrees (spanning tree) and the size of
    // the smaller cluster
    std::vector<double> merge_distance, merge_size;
    double suggested = 0.0;  // graph clustering: the threshold found
    double threshold = 0.0;  // the threshold applied
};

// Throws std::invalid_argument for neighbors outside the atoms.
void grain_segmentation(const GrainInput& in, const GrainOptions& options, GrainOutput& out,
                        const Control& control = {});

}  // namespace mdapy::identify
