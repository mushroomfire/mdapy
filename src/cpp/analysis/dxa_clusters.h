// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The crystal clusters of the dislocation analysis: the atoms polyhedral
// template matching finds in a structure, joined into clusters of one
// lattice frame each, the transitions between the frames of clusters that
// touch, and the grains the clusters on the two sides of a stacking fault
// or another defect of the crystal make together.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "analysis/dxa_math.h"
#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::analysis::dxa {

// A lattice the analysis works in: the ideal vector to each neighbor of
// each template of polyhedral template matching, in units of the cubic
// lattice constant (hcp that of fcc, hexagonal diamond that of cubic
// diamond, with the same nearest neighbors), in the order of the
// template; for each neighbor, two other neighbors of the central atom that
// are its neighbors too, the three vectors from it not in a plane; the
// rotations that take the neighbors of the first template onto themselves,
// and the index of the product of each two of them.
struct Lattice {
    int structure = 0;  // PTM type
    int neighbors = 0;
    std::vector<std::array<Vec, 16>> templates;
    std::array<std::array<int, 2>, 16> common{};
    std::vector<Mat> group;
    std::vector<std::vector<std::int16_t>> product;  // group[product[a][b]] = group[a] group[b]
};

// The lattice of a PTM type, or nullptr for one the analysis does not know:
// fcc, hcp, bcc, cubic and hexagonal diamond.
const Lattice* lattice(int structure);

// The atoms: the PTM type and template of each, and their neighbors in the
// order of the template (16 each, -1 after the last); positions and the
// box, for the orientations of the clusters.
struct ClusterInput {
    std::int64_t n = 0;
    const std::int8_t* structure = nullptr;
    const std::int8_t* templates = nullptr;
    const index_t* neighbors = nullptr;
    const double* positions = nullptr;  // n x 3
    const double* matrix = nullptr;     // rows a, b, c
    const double* inverse = nullptr;
    const bool* pbc = nullptr;
    int crystal = 0;  // the PTM type of the crystal, whose defects join its grains
};

// A transition between the frames of two clusters that touch: the matrix
// that takes a vector of the frame of the first into that of the second,
// and the number of pairs of neighbors across.
struct Transition {
    std::int64_t from, to;
    Mat matrix;
    std::int64_t area;
};

// The clusters: of each atom, its cluster (-1 for none) and the element of
// the group of its lattice that turns its template into the frame of the
// cluster; of each cluster, its structure, size, orientation (the matrix
// that takes its lattice vectors to vectors in Å, by least squares), its
// grain (a cluster of the crystal) and the matrix from its frame into that
// of the grain; the transitions, each pair once.
struct Clusters {
    std::vector<std::int64_t> cluster;
    std::vector<std::int16_t> rotation;
    std::vector<std::int8_t> structure;
    std::vector<std::int64_t> size;
    std::vector<double> orientation;  // 9 each
    std::vector<std::int64_t> grain;
    std::vector<double> to_grain;  // 9 each
    std::vector<Transition> transitions;
};

Clusters build_clusters(const ClusterInput& input, const Control& control = {});

}  // namespace mdapy::analysis::dxa
