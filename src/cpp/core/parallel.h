// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Thread pool used by every compiled kernel in mdapy. It does not use OpenMP,
// so mdapy never loads an OpenMP runtime and cannot clash with the ones
// shipped by PyTorch, NumPy or LAMMPS.

#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <vector>

namespace mdapy {

// Thrown when a parallel region stops because cancellation was requested.
class Cancelled : public std::exception {
public:
    const char* what() const noexcept override { return "operation cancelled"; }
};

// Flag that any thread can set to stop a running parallel region.
class CancelToken {
public:
    void cancel() noexcept { flag_.store(true, std::memory_order_relaxed); }
    void reset() noexcept { flag_.store(false, std::memory_order_relaxed); }
    bool cancelled() const noexcept { return flag_.load(std::memory_order_relaxed); }

private:
    std::atomic<bool> flag_{false};
};

// Called on the calling thread while a region runs, with the number of
// finished and total work items. Returning false requests cancellation.
using Monitor = std::function<bool(std::int64_t done, std::int64_t total)>;

struct Control {
    const CancelToken* cancel = nullptr;  // optional
    Monitor monitor;                      // optional
};

// The control of part `part` (from 0) of `parts` regions of equal size run
// one after the other: its monitor sees them as a single region.
inline Control stage(const Control& control, std::int64_t part, std::int64_t parts) {
    Control result;
    result.cancel = control.cancel;
    if (control.monitor) {
        result.monitor = [monitor = control.monitor, part, parts](std::int64_t done,
                                                                  std::int64_t total) {
            return monitor(part * total + done, parts * total);
        };
    }
    return result;
}

// Number of worker threads used by parallel regions (at least 1).
int get_num_threads();
void set_num_threads(int n);  // throws std::invalid_argument if n < 1
int hardware_threads();

// Chunk size used when grain <= 0. It depends only on n, never on the number
// of threads, so chunked reductions give identical results for any thread count.
std::int64_t default_grain(std::int64_t n);

// Calls body(begin, end) once for each chunk [begin, end) of [0, n), where
// chunk c covers [c * grain, min(n, (c + 1) * grain)). Chunks may run
// concurrently and in any order. Exceptions thrown by body are rethrown here;
// a cancelled region throws Cancelled. The monitor, if given, is called
// periodically on the calling thread and once more with done == total after
// a successful run.
void parallel_for(std::int64_t n, std::int64_t grain,
                  const std::function<void(std::int64_t, std::int64_t)>& body,
                  const Control& control = {});

// Maps each chunk to a value with map(begin, end) and folds the chunk values
// with combine in chunk order, starting from init. The result is the same for
// any number of threads.
template <class T, class Map, class Combine>
T parallel_reduce(std::int64_t n, std::int64_t grain, T init, Map map, Combine combine,
                  const Control& control = {}) {
    if (n <= 0) {
        return init;
    }
    if (grain <= 0) {
        grain = default_grain(n);
    }
    const std::int64_t nchunks = (n + grain - 1) / grain;
    std::vector<T> partial(static_cast<std::size_t>(nchunks), init);
    parallel_for(
        n, grain,
        [&](std::int64_t begin, std::int64_t end) {
            partial[static_cast<std::size_t>(begin / grain)] = map(begin, end);
        },
        control);
    T result = init;
    for (const T& value : partial) {
        result = combine(result, value);
    }
    return result;
}

}  // namespace mdapy
