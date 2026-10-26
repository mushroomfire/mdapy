// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Planar faults of fcc crystals: the hcp-like atoms of stacking faults and
// twin boundaries, as OVITO's Identify fcc planar faults modifier finds them.

#pragma once

#include <cstdint>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::identify {

// Planar fault types.
enum PlanarFault : std::int8_t {
    kNonHcp = 0,
    kOtherHcp = 1,
    kIntrinsicFault = 2,
    kTwinBoundary = 3,
    kMultilayerFault = 4,
};

// The planar fault type of each of n atoms, from their PTM structure types
// (kPtmFcc, kPtmHcp, ...) and their neighbors in the order of the points of
// the PTM template (n x 16; the first 12 of each hcp atom). A first pass
// classifies each hcp atom by the hcp neighbors in its basal plane and the
// hcp and fcc neighbors stacked above and below it; a second pass, over the
// hcp atoms in the order of their indices and changing their types as it
// goes, as OVITO does, turns twin and other atoms whose basal neighbors are
// all of one fault into that fault, and the stacking-fault neighbors above
// and below a multilayer atom into multilayer atoms.
void planar_faults(const std::int8_t* structure, const index_t* neighbors, std::int64_t n,
                   std::int8_t* faults, const Control& control = {});

}  // namespace mdapy::identify
