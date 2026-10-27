// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The elastic mapping of the dislocation analysis: the ideal vector of each
// edge of the Delaunay tetrahedra, the vector between its two atoms in the
// stress-free crystal, found through the neighbors of the crystal atoms;
// and the tetrahedra where the mapping is compatible, the good crystal.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "analysis/dxa_clusters.h"
#include "analysis/tessellation.h"
#include "core/parallel.h"

namespace mdapy::analysis::dxa {

// The ideal vectors between atoms. Every atom gets a cluster, its own or,
// for one in none, that of a neighbor through the edges of the tetrahedra;
// an ideal vector is given in the frame of the cluster of its first atom.
class Mapping {
public:
    Mapping(const ClusterInput& atoms, const Clusters& clusters, const Tetrahedra& tetrahedra,
            const Control& control = {});

    // the cluster of each atom, -1 for one of a part with no crystal
    const std::vector<std::int64_t>& vertex_cluster() const { return vertex_cluster_; }

    // the matrix that takes vectors of the frame of cluster c1 into that of
    // c2: the same cluster, a transition, or through their grain
    std::optional<Mat> transition(std::int64_t c1, std::int64_t c2) const;

    // the ideal vector from atom i to atom j, in the frame of the cluster of
    // i, through at most four steps along neighbors of crystal atoms
    std::optional<Vec> ideal(std::int64_t i, std::int64_t j) const;

    // whether the mapping is compatible on tetrahedron t: its six edges
    // have ideal vectors, which add up to zero around each face, and the
    // transitions around each face come back to the same frame
    bool good(std::int64_t t) const;

private:
    std::optional<Vec> path(std::int64_t i, std::int64_t j) const;

    const ClusterInput& atoms_;
    const Clusters& clusters_;
    const Tetrahedra& tetrahedra_;
    std::vector<std::int64_t> vertex_cluster_;
    // for each atom outside the clusters, the crystal atoms that count it
    // among their neighbors, and the place it has there
    std::vector<std::int64_t> reverse_offsets_;
    std::vector<std::int64_t> reverse_atoms_;
    std::vector<std::int8_t> reverse_places_;
};

}  // namespace mdapy::analysis::dxa
