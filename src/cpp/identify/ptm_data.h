// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The templates of polyhedral template matching and what is known of them
// before any atom is matched: their symmetries, the graphs of their convex
// hulls and the canonical forms of these, built once from the geometry.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "identify/ptm_tools.h"

namespace mdapy::identify::ptm {

// The structures, in OVITO's numbers.
enum Type : int {
    kNone = 0,
    kFcc = 1,
    kHcp = 2,
    kBcc = 3,
    kIco = 4,
    kSc = 5,
    kCubicDiamond = 6,
    kHexagonalDiamond = 7,
    kGraphene = 8,
};

// The chemical orderings, in OVITO's numbers.
enum Ordering : int {
    kOrderingNone = 0,
    kPure = 1,
    kL10 = 2,
    kL12Majority = 3,  // L1_2, an atom of the majority element (Cu of Cu3Au)
    kL12Minority = 4,  // L1_2, an atom of the minority element (Au of Cu3Au)
    kB2 = 5,
    kZincblende = 6,
    kBoronNitride = 7,
};

// A permutation of the points of a template, the central atom staying 0.
using Permutation = std::array<std::int8_t, kMaxPoints>;

// One triangulation of the hull of a template, up to the rotations of the
// template: its canonical form and every labelling with that form.
struct Graph {
    Form form;
    std::vector<std::array<std::int8_t, kMaxNeighbors>> labellings;
};

struct Reference {
    int type = kNone;
    int neighbors = 0;
    int facets = 0;      // of the hull
    int max_degree = 0;  // of the nodes of the hull
    std::vector<Graph> graphs;
    // the templates: the central atom and its neighbors, their mean
    // distance from the centre 1; the first is matched, the others are the
    // same turned by an element of the conventional group
    std::vector<std::array<Point, kMaxPoints>> templates;
    // their pseudo-inverses, for the deformation gradient
    std::vector<std::array<Point, kMaxPoints>> penrose;
    // the permutations of the first template by its own rotations
    std::vector<Permutation> symmetries;
    // the conventional group, which takes the orientation of an atom into
    // the fundamental zone: for each element, the template its turn gives
    // and the neighbors it moves: the new neighbor j is the old moves[j]
    const std::vector<Quaternion>* group = nullptr;
    std::vector<int> group_template;
    std::vector<Permutation> moves;
    double distance = 0.0;  // of the first neighbor from the centre
};

// The data of a structure.
const Reference& reference(int type);

// The canonical codes of the chemical orderings of fcc: the neighbors of
// another element than the central atom, as bits, smallest over the
// rotations of the template.
struct FccOrderings {
    std::uint32_t l10, l12_majority, l12_minority;
};
const FccOrderings& fcc_orderings();

// The canonical code of a pattern of neighbors of another element (bit i
// for point i): the smallest over the symmetries of the template.
std::uint32_t canonical_pattern(const Reference& ref, const std::int8_t* other);

// Turns q by the element of the group that brings it nearest the identity:
// returns its index, q with w >= 0.
int into_fundamental_zone(const std::vector<Quaternion>& group, Quaternion& q);

}  // namespace mdapy::identify::ptm
