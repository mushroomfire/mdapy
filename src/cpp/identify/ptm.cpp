// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "identify/ptm.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "core/buffer.h"
#include "identify/ptm/ptm_constants.h"
#include "identify/ptm/ptm_functions.h"
#include "identify/ptm/ptm_initialize_data.h"
#include "neighbor/frame.h"

namespace mdapy::identify {
namespace {

// What the callback of the library needs: the neighbors of any atom in the
// order the first pass found.
struct Neighbors {
    const PtmInput* input;
    const std::uint64_t* ordering;
};

// The library's callback: the central atom and its neighbors, at most
// requested points in all, in the order the first pass found, as OVITO
// gives them.
int get_neighbours(void* data, std::size_t /*unused*/, std::size_t atom, int requested,
                   ptm_atomicenv_t* env) {
    const Neighbors& nbrs = *static_cast<const Neighbors*>(data);
    const PtmInput& in = *nbrs.input;
    const auto i = static_cast<std::int64_t>(atom);
    const std::int64_t first = in.offsets[i];
    const int available = static_cast<int>(in.offsets[i + 1] - first);
    const int count = std::min(requested - 1, available);
    int unused = 0;
    ptm_decode_correspondences(PTM_MATCH_FCC, nbrs.ordering[i], env->correspondences, &unused);
    env->atom_indices[0] = atom;
    env->points[0][0] = env->points[0][1] = env->points[0][2] = 0.0;
    env->numbers[0] = in.types ? in.types[i] : 0;
    const double* xi = in.positions + 3 * i;
    for (int k = 0; k < count; ++k) {
        const std::int64_t p = first + env->correspondences[k + 1] - 1;
        const auto jj = static_cast<std::int64_t>(in.j[p]);
        neighbor::pair_vector(xi, in.positions + 3 * jj, in.shifts + 3 * p, in.matrix,
                              env->points[k + 1]);
        env->atom_indices[k + 1] = static_cast<std::size_t>(jj);
        env->numbers[k + 1] = in.types ? in.types[jj] : 0;
    }
    env->num = count + 1;
    return count + 1;
}

struct Local {
    ptm_local_handle_t handle;
    Local() : handle(ptm_initialize_local()) {}
    ~Local() { ptm_uninitialize_local(handle); }
    Local(const Local&) = delete;
    Local& operator=(const Local&) = delete;
};

void initialize() {
    static std::once_flag once;
    static int status = PTM_NO_ERROR;
    std::call_once(once, [] { status = ptm_initialize_global(); });
    if (status != PTM_NO_ERROR) {
        throw std::runtime_error("the PTM library failed to initialize");
    }
}

}  // namespace

void polyhedral_template_matching(const PtmInput& in, const PtmOutput& out,
                                  const Control& control) {
    initialize();
    const std::int64_t n = in.n;
    constexpr std::int64_t kGrain = 512;
    // the neighbors of each atom ordered by the solid angles of the faces of
    // its Voronoi cell
    Buffer<std::uint64_t> ordering(static_cast<std::size_t>(n));  // every atom writes its own
    parallel_for(
        n, kGrain,
        [&](std::int64_t begin, std::int64_t end) {
            Local local;
            double points[PTM_MAX_INPUT_POINTS - 1][3];
            for (std::int64_t i = begin; i < end; ++i) {
                const std::int64_t first = in.offsets[i];
                const int count = static_cast<int>(
                    std::min<std::int64_t>(in.offsets[i + 1] - first, PTM_MAX_INPUT_POINTS - 1));
                for (int k = 0; k < count; ++k) {
                    const auto jj = static_cast<std::int64_t>(in.j[first + k]);
                    neighbor::pair_vector(in.positions + 3 * i, in.positions + 3 * jj,
                                          in.shifts + 3 * (first + k), in.matrix, points[k]);
                }
                ptm_preorder_neighbours(local.handle, count, points,
                                        &ordering[static_cast<std::size_t>(i)]);
            }
        },
        control);
    std::int32_t flags = 0;
    const std::int32_t checks[8] = {PTM_CHECK_FCC, PTM_CHECK_HCP,  PTM_CHECK_BCC,
                                    PTM_CHECK_ICO, PTM_CHECK_SC,   PTM_CHECK_DCUB,
                                    PTM_CHECK_DHEX, PTM_CHECK_GRAPHENE};
    for (int s = 0; s < 8; ++s) {
        if (in.structures & (1 << s)) {
            flags |= checks[s];
        }
    }
    const Neighbors data{&in, ordering.data()};
    parallel_for(
        n, kGrain,
        [&](std::int64_t begin, std::int64_t end) {
            Local local;
            ptm_result_t result;
            ptm_atomicenv_t env;
            for (std::int64_t i = begin; i < end; ++i) {
                std::memset(&result, 0, sizeof(result));
                std::memset(&env, 0, sizeof(env));
                const int error = ptm_index(local.handle, static_cast<std::size_t>(i),
                                            get_neighbours, const_cast<Neighbors*>(&data),
                                            flags, in.deformation, &result, &env);
                if (error != PTM_NO_ERROR) {
                    throw std::runtime_error("the PTM library failed on an atom");
                }
                const int type = result.structure_type;
                const bool found = type != PTM_MATCH_NONE;
                if (out.structure) out.structure[i] = static_cast<std::int8_t>(type);
                if (out.rmsd) out.rmsd[i] = found ? result.rmsd : 0.0;
                if (out.interatomic_distance) {
                    out.interatomic_distance[i] = found ? result.interatomic_distance : 0.0;
                }
                if (out.orientation) {
                    // the library's (w, x, y, z) as OVITO's (x, y, z, w)
                    for (int e = 0; e < 4; ++e) {
                        out.orientation[4 * i + e] = found ? result.orientation[(e + 1) % 4] : 0.0;
                    }
                }
                if (out.deformation) {
                    for (int e = 0; e < 9; ++e) {
                        out.deformation[9 * i + e] = found && in.deformation ? result.F[e] : 0.0;
                    }
                }
                if (out.ordering) {
                    out.ordering[i] = static_cast<std::int8_t>(found ? result.ordering_type : 0);
                }
                if (out.correspondences) {
                    out.correspondences[i] =
                        found ? ptm_encode_correspondences(type, env.num, env.correspondences,
                                                           result.template_index)
                              : 0;
                }
                if (out.template_index) {
                    out.template_index[i] =
                        static_cast<std::int8_t>(found ? result.template_index : 0);
                }
                if (out.neighbors) {
                    const int count = found ? ptm_num_nbrs[type] : 0;
                    for (int k = 0; k < PTM_MAX_NBRS; ++k) {
                        out.neighbors[PTM_MAX_NBRS * i + k] =
                            k < count ? static_cast<index_t>(env.atom_indices[k + 1])
                                      : static_cast<index_t>(-1);
                    }
                }
            }
        },
        control);
}

}  // namespace mdapy::identify
