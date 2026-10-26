// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The pieces of polyhedral template matching (Larsen, Schmidt and Schiøtz
// 2016): convex hulls of the neighbors of an atom, canonical forms of their
// graphs (Weinberg 1966), the rotation that best turns a template onto the
// neighbors (Horn 1987, solved as Theobald 2005 does), and the codes of the
// correspondences between neighbors and templates that OVITO stores.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace mdapy::identify::ptm {

inline constexpr int kMaxNeighbors = 16;                   // of a template
inline constexpr int kMaxPoints = kMaxNeighbors + 1;       // with the central atom
inline constexpr int kMaxInput = 19;                       // the atom and 18 neighbors
inline constexpr int kMaxFacets = 2 * kMaxNeighbors - 4;   // of a triangulated sphere
inline constexpr int kMaxEdges = 3 * kMaxNeighbors - 6;

using Point = std::array<double, 3>;
using Facet = std::array<std::int8_t, 3>;
using Quaternion = std::array<double, 4>;  // w, x, y, z

// ------------------------------------------------------------------ hull

// The convex hull of a growing set of points, the central atom first. Its
// facets turn so that (b - a) x (c - a) points inwards.
struct Hull {
    Facet facets[kMaxFacets];
    Point normals[kMaxFacets];  // unit, inwards
    bool added[kMaxPoints];
    Point centre;  // of the first tetrahedron, inside the hull
    int count = 0;
    int size = 0;  // the points it was grown to
    bool ok = false;
};

// Grows the hull to the first n points: from scratch unless hull.ok, else by
// the points it lacks. Returns 0 with the facets over the neighbors in out
// (their indices from 0, the central atom left out), 1 if the central atom
// lies on the hull, a negative number for points that span no volume or a
// hull that comes out wrong.
int grow_hull(Hull& hull, int n, const Point* points, Facet* out);

// Turns a facet so that (b - a) x (c - a) points to the side of inside, the
// smallest index first; normal is that direction, of unit length.
void orient_facet(const Point* points, Facet& facet, const Point& inside, Point& normal);

// The number of facets at each of n nodes; returns the largest.
int degrees(int count, const Facet* facets, int n, std::int8_t* degree);

// ----------------------------------------------------------- graph forms

// The canonical form of a triangulated sphere of n nodes with colours: the
// smallest code of the walks of Weinberg over every directed edge, and the
// order in which the walk of that code reaches the nodes.
struct Form {
    std::array<std::int8_t, 2 * kMaxEdges> code;
    int length = 0;        // 2 x the number of edges
    std::uint64_t hash = 0;
    std::array<std::int8_t, kMaxNeighbors> labels;  // node -> place in the walk
};

// The form of a graph; false for facets that do not close a sphere.
bool canonical_form(int count, const Facet* facets, int n, const std::int8_t* colours, Form& form);

// Every labelling that gives the canonical code: one for each automorphism
// of the coloured graph that keeps its orientation.
std::vector<std::array<std::int8_t, kMaxNeighbors>> canonical_labellings(
    int count, const Facet* facets, int n, const std::int8_t* colours, Form& form);

// ------------------------------------------------------------ rotations

// The unit quaternion of the rotation R that maximizes sum_i b_i . R a_i,
// from S = sum_i a_i b_i^T (row-major) and an upper bound of the largest
// eigenvalue, such as (sum |a|^2 + sum |b|^2) / 2.
Quaternion best_rotation(const double* S, double bound);

// The rotation matrix (row-major) of a unit quaternion.
void rotation_matrix(const Quaternion& q, double* R);

// The quaternion of a rotation matrix (row-major), w >= 0.
Quaternion quaternion(const double* R);

// a b.
Quaternion product(const Quaternion& a, const Quaternion& b);

// The eigenvalues and eigenvectors (columns of vectors) of a symmetric
// matrix of size n <= 4, by Jacobi rotations.
void symmetric_eigen(int n, double* a, double* values, double* vectors);

// ------------------------------------------------------ correspondences

// Appends to the first n numbers of order the numbers from 0 to size - 1
// that are not among them, in increasing order.
void complete_order(int n, int size, std::int8_t* order);

// A permutation of k of the numbers 0 to base - 1 as one number: the swaps
// that bring each into place, digit i in base (base - i).
std::uint64_t encode_permutation(int base, int k, const std::int8_t* permutation);
void decode_permutation(int base, int k, std::uint64_t code, std::int8_t* permutation);

}  // namespace mdapy::identify::ptm
