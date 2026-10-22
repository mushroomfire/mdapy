// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "build/sqs.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>

namespace mdapy::build {
namespace {

// The ordered pair counts of each shell and their deviation from a random
// alloy, changed one atom at a time.
class Pairs {
public:
    Pairs(std::int64_t n, const std::int64_t* offsets, const index_t* neighbors,
          const std::int8_t* shell, int shells, int elements, std::int32_t* type,
          const double* weights)
        : offsets_(offsets), neighbors_(neighbors), shell_(shell), shells_(shells),
          elements_(elements), type_(type), weights_(weights),
          counts_(static_cast<std::size_t>(shells) * elements * elements, 0),
          totals_(static_cast<std::size_t>(shells), 0), fraction_(elements, 0.0) {
        for (std::int64_t i = 0; i < n; ++i) {
            fraction_[static_cast<std::size_t>(type[i])] += 1.0 / static_cast<double>(n);
            for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                ++count(shell[p], type[i], type[neighbors[p]]);
                ++totals_[static_cast<std::size_t>(shell[p])];
            }
        }
    }

    double objective() const {
        double sum = 0.0;
        for (int s = 0; s < shells_; ++s) {
            if (totals_[static_cast<std::size_t>(s)] == 0) {
                continue;
            }
            double part = 0.0;
            for (int a = 0; a < elements_; ++a) {
                for (int b = 0; b < elements_; ++b) {
                    const double d = static_cast<double>(count(s, a, b)) /
                                         static_cast<double>(totals_[static_cast<std::size_t>(s)]) -
                                     fraction_[static_cast<std::size_t>(a)] *
                                         fraction_[static_cast<std::size_t>(b)];
                    part += d * d;
                }
            }
            sum += weights_[s] * part;
        }
        return sum;
    }

    // Gives atom i the element to, updating the pairs it is part of: those
    // of its list and their reverses in the lists of the others; a pair of
    // the atom with its own image is reversed by another entry of its list.
    void change(std::int64_t i, std::int32_t to) {
        const std::int32_t from = type_[i];
        for (std::int64_t p = offsets_[i]; p < offsets_[i + 1]; ++p) {
            const std::int64_t j = neighbors_[p];
            const int s = shell_[p];
            if (j == i) {
                --count(s, from, from);
                ++count(s, to, to);
                continue;
            }
            const std::int32_t other = type_[j];
            --count(s, from, other);
            --count(s, other, from);
            ++count(s, to, other);
            ++count(s, other, to);
        }
        type_[i] = to;
    }

private:
    std::int64_t& count(int s, std::int32_t a, std::int32_t b) {
        return counts_[(static_cast<std::size_t>(s) * elements_ + static_cast<std::size_t>(a)) *
                           elements_ +
                       static_cast<std::size_t>(b)];
    }
    std::int64_t count(int s, std::int32_t a, std::int32_t b) const {
        return counts_[(static_cast<std::size_t>(s) * elements_ + static_cast<std::size_t>(a)) *
                           elements_ +
                       static_cast<std::size_t>(b)];
    }

    const std::int64_t* offsets_;
    const index_t* neighbors_;
    const std::int8_t* shell_;
    int shells_, elements_;
    std::int32_t* type_;
    const double* weights_;
    std::vector<std::int64_t> counts_, totals_;
    std::vector<double> fraction_;
};

void check(std::int64_t n, const std::int8_t* shell, std::int64_t pairs, int shells, int elements,
           const std::int32_t* type) {
    if (shells < 1 || elements < 1) {
        throw std::invalid_argument("shells and elements must be at least 1");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (type[i] < 0 || type[i] >= elements) {
            throw std::invalid_argument("types must be from 0 to elements - 1");
        }
    }
    for (std::int64_t p = 0; p < pairs; ++p) {
        if (shell[p] < 0 || shell[p] >= shells) {
            throw std::invalid_argument("shells must be from 0 to shells - 1");
        }
    }
}

}  // namespace

double sqs_objective(std::int64_t n, const std::int64_t* offsets, const index_t* neighbors,
                     const std::int8_t* shell, int shells, int elements, const std::int32_t* type,
                     const double* weights) {
    check(n, shell, offsets[n], shells, elements, type);
    std::vector<std::int32_t> copy(type, type + n);
    const Pairs pairs(n, offsets, neighbors, shell, shells, elements, copy.data(), weights);
    return pairs.objective();
}

SqsResult sqs(std::int64_t n, const std::int64_t* offsets, const index_t* neighbors,
              const std::int8_t* shell, int shells, int elements, std::int32_t* type,
              const double* weights, std::int64_t steps, double temperature, std::uint64_t seed,
              const Control& control) {
    check(n, shell, offsets[n], shells, elements, type);
    std::vector<std::int32_t> current(type, type + n);
    Pairs pairs(n, offsets, neighbors, shell, shells, elements, current.data(), weights);
    SqsResult result{pairs.objective(), pairs.objective(), 0};
    if (n < 2 || steps <= 0) {
        return result;
    }
    std::mt19937_64 random(seed);
    std::uniform_int_distribution<std::int64_t> atom(0, n - 1);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    // the typical change of a swap sets the scale of the temperature
    double scale = 0.0;
    int tried = 0;
    double energy = result.start;
    for (int k = 0; k < 1000 && tried < 100; ++k) {
        const std::int64_t i = atom(random), j = atom(random);
        const std::int32_t a = current[static_cast<std::size_t>(i)];
        const std::int32_t b = current[static_cast<std::size_t>(j)];
        if (a == b) {
            continue;
        }
        pairs.change(i, b);
        pairs.change(j, a);
        scale += std::fabs(pairs.objective() - energy);
        pairs.change(j, b);
        pairs.change(i, a);
        ++tried;
    }
    if (tried == 0) {
        return result;  // one element only
    }
    const double start_temperature = temperature * scale / tried;
    const std::int64_t chunk = (steps + 99) / 100;
    const std::int64_t parts = (steps + chunk - 1) / chunk;
    for (std::int64_t part = 0; part < parts; ++part) {
        // each part on the calling thread, for the progress and cancellation
        parallel_for(
            1, 1,
            [&](std::int64_t, std::int64_t) {
                const std::int64_t end = std::min(steps, (part + 1) * chunk);
                for (std::int64_t step = part * chunk; step < end; ++step) {
                    const std::int64_t i = atom(random), j = atom(random);
                    const std::int32_t a = current[static_cast<std::size_t>(i)];
                    const std::int32_t b = current[static_cast<std::size_t>(j)];
                    if (a == b) {
                        continue;
                    }
                    pairs.change(i, b);
                    pairs.change(j, a);
                    const double proposed = pairs.objective();
                    const double t = start_temperature * (1.0 - static_cast<double>(step) /
                                                                     static_cast<double>(steps));
                    const double delta = proposed - energy;
                    if (delta <= 0.0 || (t > 0.0 && uniform(random) < std::exp(-delta / t))) {
                        energy = proposed;
                        ++result.accepted;
                        if (energy < result.objective) {
                            result.objective = energy;
                            std::copy(current.begin(), current.end(), type);
                        }
                    } else {
                        pairs.change(j, b);
                        pairs.change(i, a);
                    }
                }
            },
            stage(control, part, parts));
    }
    return result;
}

}  // namespace mdapy::build
