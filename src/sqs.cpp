// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
//
// Special Quasirandom Structure (SQS) generation, modeled after ATAT's mcsqs
// (van de Walle 2009).  Trigonometric correlation-function basis:
//
//     phi_{2t-1}(s) = -cos(2*pi*t*s/m),  t=1..floor(m/2)
//     phi_{2t  }(s) = -sin(2*pi*t*s/m),  t=1..ceil(m/2)-1
//
// for an m-component sublattice (m >= 2).  An "instance" is one cluster of
// atoms in the supercell.  A "channel" groups instances that share the same
// (distance shell, function-tuple); its correlation is
//
//     pi_alpha = (1 / N_instances) * sum_instance prod_p phi_{f_p}(s_{a_p}),
//
// and its target under a fully-random configuration is prod_p < phi_{f_p} >.
//
// Two objective functions are available:
//   "abs"   : O = sum_alpha w_alpha * |pi_alpha - target_alpha|
//   "atat"  : ATAT mcsqs formula with the d1 perfect-match reward (default).
//             Tracks per-body-order maxdist[n_pts-2] = max diameter such that
//             all clusters with d <= maxdist match within tolerance.
//             d1 = min_p maxdist[p].  Objective is
//             O = (sum |Delta_pi|*w over d>=d1) / (sum w)
//                 - w_dist * sum_p rho^p * maxdist[p] / d_min.
//
// Incremental update: on swap(i,j) we only revisit instances touching i or j;
// each affected instance updates its channel's sigma sum in O(1).  The full
// objective is then recomputed in O(N_channels) which is cheap (typically
// 50-500 channels) and avoids subtle bugs around the d1 reordering.
//
// OpenMP parallelism: n_replicas independent Markov chains run concurrently
// from different seeds; the chain with the smallest objective wins.

#include "type.h"
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <map>
#include <omp.h>

namespace {

std::vector<std::vector<double>> build_trigo_basis(int m) {
    const int n_func = m - 1;
    std::vector<std::vector<double>> table(n_func, std::vector<double>(m, 0.0));
    // 2π hard-coded so we don't need M_PI (MSVC <cmath> hides it unless
    // _USE_MATH_DEFINES is set before any system header).
    const double TWO_PI = 6.283185307179586476925286766559;
    for (int s = 0; s < m; ++s) {
        for (int t = 1; t <= m / 2; ++t) {
            table[2 * t - 2][s] = -std::cos(TWO_PI * s * t / m);
        }
        for (int t = 1; t <= (m + 1) / 2 - 1; ++t) {
            table[2 * t - 1][s] = -std::sin(TWO_PI * s * t / m);
        }
    }
    return table;
}

// One cluster instance in the supercell.  `channel_idx` selects which
// (shell, function-tuple) channel it contributes to.
struct Instance {
    int atoms[4];
    int n_pts;
    int channel_idx;
};

// One correlation channel: aggregates sum of sigma across all its instances.
struct Channel {
    int    shell;
    int    funcs[4];
    int    n_pts;
    int    n_instances;
    double target;       // < pi >_random
    double diameter;     // copied from shell_diameter[shell] for hot-loop locality
};

class SQSEngine {
public:
    int n_atoms = 0;
    int n_species = 0;
    int n_func = 0;

    std::vector<double>              concentrations;   // n_species
    std::vector<std::vector<double>> corrfunc;         // [n_func][n_species]
    std::vector<double>              point_corr;       // < phi_k > per func

    std::vector<Channel>             channels;
    std::vector<Instance>            instances;
    std::vector<std::vector<int>>    atom2instances;   // [atom] -> instance indices

    std::vector<double>              shell_weight;     // per shell
    std::vector<double>              shell_diameter;   // per shell, for reporting

    // ATAT-style objective parameters.
    // mode = 0  -> "abs"  (simple sum of weighted |Delta|)
    // mode = 1  -> "atat" (d1 perfect-match reward, mcsqs default formula)
    int    objective_mode = 1;
    double atat_tol       = 1e-3;   // |Delta_pi| below this counts as "match"
    double atat_w_dist    = 1.0;    // multiplier on the d1 reward term
    double atat_w_npts    = 1.0;    // rho exponentiated per extra body order
    int    max_n_pts      = 2;      // max body order across all channels (set in finalize)

    // (shell, n_pts, packed_funcs) -> channel index, used during construction
    std::map<std::tuple<int,int,int64_t>, int> channel_lookup_;

    // ----- construction -------------------------------------------------------
    void set_species(int m, const std::vector<double>& conc) {
        if (m < 2) throw std::runtime_error("n_species must be >= 2");
        if ((int)conc.size() != m) throw std::runtime_error("conc size mismatch");
        n_species = m;
        n_func    = m - 1;
        concentrations = conc;
        corrfunc = build_trigo_basis(m);
        point_corr.assign(n_func, 0.0);
        for (int k = 0; k < n_func; ++k) {
            double v = 0.0;
            for (int s = 0; s < m; ++s) v += conc[s] * corrfunc[k][s];
            point_corr[k] = v;
        }
    }

    // tuples_flat: (n_clusters * n_pts) atom indices.
    // shell_ids:   (n_clusters,) shell index per cluster.
    // Each cluster expands into n_func^n_pts (shell, function-tuple) channels;
    // distinct cluster instances sharing the same channel are aggregated.
    void add_cluster_body(
        int n_pts,
        const std::vector<int>& tuples_flat,
        const std::vector<int>& shell_ids
    ) {
        if (n_pts < 2 || n_pts > 4) throw std::runtime_error("n_pts must be 2..4");
        const int n_clusters = (int)shell_ids.size();
        if ((int)tuples_flat.size() != n_clusters * n_pts)
            throw std::runtime_error("tuples_flat size mismatch");
        if (n_func == 0) throw std::runtime_error("set_species() must be called first");

        int tuples_per_cluster = 1;
        for (int p = 0; p < n_pts; ++p) tuples_per_cluster *= n_func;

        std::vector<int> ftuple(n_pts, 0);
        for (int c = 0; c < n_clusters; ++c) {
            const int* atoms = &tuples_flat[c * n_pts];
            int shell = shell_ids[c];
            std::fill(ftuple.begin(), ftuple.end(), 0);

            for (int t = 0; t < tuples_per_cluster; ++t) {
                // pack function tuple into a 64-bit key for the map lookup
                int64_t fkey = 0;
                for (int p = 0; p < n_pts; ++p) fkey = fkey * 16 + ftuple[p];

                auto key = std::make_tuple(shell, n_pts, fkey);
                auto it  = channel_lookup_.find(key);
                int  ch_idx;
                if (it == channel_lookup_.end()) {
                    ch_idx = (int)channels.size();
                    Channel ch;
                    ch.shell = shell;
                    ch.n_pts = n_pts;
                    for (int p = 0; p < n_pts; ++p) ch.funcs[p] = ftuple[p];
                    ch.n_instances = 0;
                    ch.target = 0.0;     // filled in finalize()
                    channels.push_back(ch);
                    channel_lookup_[key] = ch_idx;
                } else {
                    ch_idx = it->second;
                }

                Instance inst;
                inst.n_pts       = n_pts;
                inst.channel_idx = ch_idx;
                for (int p = 0; p < n_pts; ++p) inst.atoms[p] = atoms[p];
                instances.push_back(inst);
                channels[ch_idx].n_instances += 1;

                // increment function-tuple in mixed radix n_func
                for (int p = 0; p < n_pts; ++p) {
                    if (++ftuple[p] < n_func) break;
                    ftuple[p] = 0;
                }
            }
        }
    }

    // After all add_cluster_body() calls and once n_atoms is set.
    void finalize() {
        if (n_atoms <= 0) throw std::runtime_error("n_atoms must be set");
        if ((int)shell_diameter.size() == 0)
            throw std::runtime_error("shell_diameter must be set before finalize()");
        // compute targets and copy diameter onto each channel for locality
        max_n_pts = 2;
        for (auto& ch : channels) {
            double t = 1.0;
            for (int p = 0; p < ch.n_pts; ++p) t *= point_corr[ch.funcs[p]];
            ch.target   = t;
            ch.diameter = shell_diameter[ch.shell];
            if (ch.n_pts > max_n_pts) max_n_pts = ch.n_pts;
        }
        // build atom -> instance map (each atom records each instance once)
        atom2instances.assign(n_atoms, {});
        const int N = (int)instances.size();
        for (int i = 0; i < N; ++i) {
            const Instance& inst = instances[i];
            int prev[4] = {-1, -1, -1, -1};
            for (int p = 0; p < inst.n_pts; ++p) {
                int a = inst.atoms[p];
                if (a < 0 || a >= n_atoms)
                    throw std::runtime_error("atom index out of range in instance");
                bool seen = false;
                for (int q = 0; q < p; ++q) if (prev[q] == a) { seen = true; break; }
                if (!seen) atom2instances[a].push_back(i);
                prev[p] = a;
            }
        }
        // free temporary lookup
        channel_lookup_.clear();
    }

    // ----- evaluation ---------------------------------------------------------
    inline double instance_sigma(int inst_idx, const std::vector<int>& types) const {
        const Instance& inst  = instances[inst_idx];
        const Channel&  ch    = channels[inst.channel_idx];
        double s = 1.0;
        for (int p = 0; p < inst.n_pts; ++p) {
            s *= corrfunc[ch.funcs[p]][types[inst.atoms[p]]];
        }
        return s;
    }

    // Compute per-instance sigma and per-channel sum_sigma for given types.
    void compute_state(const std::vector<int>& types,
                       std::vector<double>& sigma_out,
                       std::vector<double>& sum_sigma_out) const {
        const int N = (int)instances.size();
        sigma_out.assign(N, 0.0);
        sum_sigma_out.assign(channels.size(), 0.0);
        for (int i = 0; i < N; ++i) {
            double s = instance_sigma(i, types);
            sigma_out[i] = s;
            sum_sigma_out[instances[i].channel_idx] += s;
        }
    }

    double objective_from_sumsigma(const std::vector<double>& sum_sigma) const {
        const int Nc = (int)channels.size();
        if (objective_mode == 0) {
            double obj = 0.0;
            for (int i = 0; i < Nc; ++i) {
                double pi = sum_sigma[i] / (double)channels[i].n_instances;
                obj += shell_weight[channels[i].shell] * std::fabs(pi - channels[i].target);
            }
            return obj;
        }
        // mode == 1: ATAT mcsqs formula with d1 perfect-match reward.
        // First pass: compute |Delta_pi|, find d_min and per-body maxdist.
        double d_min = std::numeric_limits<double>::infinity();
        for (int i = 0; i < Nc; ++i) {
            if (channels[i].diameter < d_min) d_min = channels[i].diameter;
        }
        if (!std::isfinite(d_min) || d_min <= 0.0) d_min = 1.0;

        const int n_body = max_n_pts - 1;   // bodies indexed by n_pts-2 = 0..max-2
        std::vector<double> maxdist(n_body, 0.0);
        // Initially: maxdist[p] = max diameter seen for that body (= upper bound).
        for (int i = 0; i < Nc; ++i) {
            int p = channels[i].n_pts - 2;
            if (channels[i].diameter > maxdist[p]) maxdist[p] = channels[i].diameter;
        }
        // Add d_min, then shrink to "longest perfectly-matched prefix".
        for (int p = 0; p < n_body; ++p) maxdist[p] += d_min;
        std::vector<double> dcorr(Nc);
        for (int i = 0; i < Nc; ++i) {
            double pi = sum_sigma[i] / (double)channels[i].n_instances;
            dcorr[i]  = std::fabs(pi - channels[i].target);
            int p     = channels[i].n_pts - 2;
            if (dcorr[i] > atat_tol && channels[i].diameter < maxdist[p]) {
                maxdist[p] = channels[i].diameter;
            }
        }
        // d1 = monotonic min across bodies
        double d1 = maxdist[0];
        for (int p = 1; p < n_body; ++p) {
            if (maxdist[p] < maxdist[p - 1]) maxdist[p] = maxdist[p];   // no-op, kept for parity
            else                              maxdist[p] = maxdist[p - 1];
            if (maxdist[p] < d1) d1 = maxdist[p];
        }
        // Second pass: weighted sum over channels with d >= d1.
        double objdev = 0.0, den = 0.0;
        for (int i = 0; i < Nc; ++i) {
            if (channels[i].diameter >= d1 - 1e-12) {
                double w = shell_weight[channels[i].shell]
                         * std::pow(atat_w_npts, channels[i].n_pts - 2);
                objdev += dcorr[i] * w;
                den    += w;
            }
        }
        objdev = (den > 0 ? objdev / den : 0.0);

        double obj = objdev;
        for (int p = 0; p < n_body; ++p) {
            obj -= atat_w_dist * std::pow(atat_w_npts, p) * maxdist[p] / d_min;
        }
        return obj;
    }

    // Diagnostic: per-channel (corr, |corr-target|), useful for is_sqs() judgment.
    std::vector<double> per_channel_delta(const std::vector<int>& types) const {
        std::vector<double> sigma, ssum;
        compute_state(types, sigma, ssum);
        std::vector<double> out(channels.size());
        for (size_t i = 0; i < channels.size(); ++i) {
            double pi = ssum[i] / (double)channels[i].n_instances;
            out[i] = std::fabs(pi - channels[i].target);
        }
        return out;
    }

    // Per-channel correlation values pi_alpha = sum/N for given types.
    std::vector<double> correlations(const std::vector<int>& types) const {
        std::vector<double> sigma, ssum;
        compute_state(types, sigma, ssum);
        std::vector<double> out(channels.size());
        for (size_t i = 0; i < channels.size(); ++i) {
            out[i] = ssum[i] / (double)channels[i].n_instances;
        }
        return out;
    }

    // ----- Monte Carlo --------------------------------------------------------
    // Runs n_replicas independent Markov chains via OpenMP and returns the
    // (best types, best objective, best correlations).
    std::tuple<std::vector<int>, double, std::vector<double>>
    run_mc(const std::vector<int>& init_types,
           int max_steps,
           double T,
           int n_replicas,
           uint64_t seed) {
        if ((int)init_types.size() != n_atoms)
            throw std::runtime_error("init_types size != n_atoms");

        struct Replica {
            std::vector<int>    types;
            std::vector<double> sigma;       // per instance
            std::vector<double> sum_sigma;   // per channel
            double              objective = 0.0;
        };

        std::vector<Replica> reps(n_replicas);
        for (int r = 0; r < n_replicas; ++r) {
            reps[r].types = init_types;
            std::mt19937_64 rng(seed + 7919ULL * (uint64_t)r);
            std::shuffle(reps[r].types.begin(), reps[r].types.end(), rng);
            compute_state(reps[r].types, reps[r].sigma, reps[r].sum_sigma);
            reps[r].objective = objective_from_sumsigma(reps[r].sum_sigma);
        }

        const int n_inst = (int)instances.size();

        #pragma omp parallel for schedule(dynamic, 1)
        for (int r = 0; r < n_replicas; ++r) {
            auto& rep = reps[r];
            std::mt19937_64 rng(seed + 1009ULL * (uint64_t)r + 1);
            std::uniform_real_distribution<double> uni01(0.0, 1.0);
            std::uniform_int_distribution<int>     pick(0, n_atoms - 1);

            std::vector<int>    touched_inst;  touched_inst.reserve(256);
            std::vector<char>   touched(n_inst, 0);
            std::vector<double> new_sigmas;    new_sigmas.reserve(256);
            std::vector<int>    ch_touched;    ch_touched.reserve(256);
            std::vector<double> ch_delta;      ch_delta.reserve(256);
            std::vector<char>   ch_seen(channels.size(), 0);

            for (int step = 0; step < max_steps; ++step) {
                int i = pick(rng);
                int j = pick(rng);
                if (i == j || rep.types[i] == rep.types[j]) continue;

                touched_inst.clear();
                for (int k : atom2instances[i]) {
                    if (!touched[k]) { touched[k] = 1; touched_inst.push_back(k); }
                }
                for (int k : atom2instances[j]) {
                    if (!touched[k]) { touched[k] = 1; touched_inst.push_back(k); }
                }

                // tentatively swap atoms
                std::swap(rep.types[i], rep.types[j]);

                // Compute per-instance new sigma + per-channel sigma delta
                ch_touched.clear();
                ch_delta.clear();
                new_sigmas.resize(touched_inst.size());
                for (size_t k = 0; k < touched_inst.size(); ++k) {
                    int idx = touched_inst[k];
                    double new_s = instance_sigma(idx, rep.types);
                    new_sigmas[k] = new_s;
                    double d = new_s - rep.sigma[idx];
                    int ch = instances[idx].channel_idx;
                    if (!ch_seen[ch]) {
                        ch_seen[ch] = 1;
                        ch_touched.push_back(ch);
                        ch_delta.push_back(d);
                    } else {
                        for (size_t m = 0; m < ch_touched.size(); ++m) {
                            if (ch_touched[m] == ch) { ch_delta[m] += d; break; }
                        }
                    }
                }

                // Apply deltas to sum_sigma, recompute objective globally
                // (cheap — O(N_channels)), then accept/reject.
                for (size_t m = 0; m < ch_touched.size(); ++m) {
                    rep.sum_sigma[ch_touched[m]] += ch_delta[m];
                }
                double new_obj = objective_from_sumsigma(rep.sum_sigma);
                double delta   = new_obj - rep.objective;

                bool accept = (delta <= 0.0) || (uni01(rng) < std::exp(-delta / T));

                if (accept) {
                    for (size_t k = 0; k < touched_inst.size(); ++k) {
                        rep.sigma[touched_inst[k]] = new_sigmas[k];
                    }
                    rep.objective = new_obj;
                } else {
                    // revert sum_sigma and types
                    for (size_t m = 0; m < ch_touched.size(); ++m) {
                        rep.sum_sigma[ch_touched[m]] -= ch_delta[m];
                    }
                    std::swap(rep.types[i], rep.types[j]);
                }

                for (int k : touched_inst) touched[k] = 0;
                for (int ch : ch_touched)  ch_seen[ch] = 0;
            }
        }

        int best = 0;
        for (int r = 1; r < n_replicas; ++r) {
            if (reps[r].objective < reps[best].objective) best = r;
        }
        std::vector<double> best_corr(channels.size());
        for (size_t i = 0; i < channels.size(); ++i) {
            best_corr[i] = reps[best].sum_sigma[i] / (double)channels[i].n_instances;
        }
        return std::make_tuple(reps[best].types, reps[best].objective, best_corr);
    }
};

} // namespace

NB_MODULE(_sqs, m) {
    nb::class_<SQSEngine>(m, "SQSEngine")
        .def(nb::init<>())
        .def_rw("n_atoms",         &SQSEngine::n_atoms)
        .def_ro("n_species",       &SQSEngine::n_species)
        .def_ro("n_func",          &SQSEngine::n_func)
        .def_ro("point_corr",      &SQSEngine::point_corr)
        .def_rw("shell_weight",    &SQSEngine::shell_weight)
        .def_rw("shell_diameter",  &SQSEngine::shell_diameter)
        .def_rw("objective_mode",  &SQSEngine::objective_mode)
        .def_rw("atat_tol",        &SQSEngine::atat_tol)
        .def_rw("atat_w_dist",     &SQSEngine::atat_w_dist)
        .def_rw("atat_w_npts",     &SQSEngine::atat_w_npts)
        .def("set_species",        &SQSEngine::set_species)
        .def("add_cluster_body",   &SQSEngine::add_cluster_body)
        .def("finalize",           &SQSEngine::finalize)
        .def("correlations",       &SQSEngine::correlations)
        .def("per_channel_delta",  &SQSEngine::per_channel_delta)
        .def("objective",
            [](SQSEngine& self, const std::vector<int>& types) {
                std::vector<double> sigma, ssum;
                self.compute_state(types, sigma, ssum);
                return self.objective_from_sumsigma(ssum);
            })
        .def("run_mc",             &SQSEngine::run_mc)
        .def("num_channels",
            [](SQSEngine& self) { return (int)self.channels.size(); })
        .def("num_instances",
            [](SQSEngine& self) { return (int)self.instances.size(); })
        .def("channel_info",
            [](SQSEngine& self, int i) {
                const Channel& c = self.channels[i];
                std::vector<int> funcs(c.funcs, c.funcs + c.n_pts);
                return std::make_tuple(c.n_pts, c.shell, funcs,
                                       c.n_instances, c.target);
            });
}
