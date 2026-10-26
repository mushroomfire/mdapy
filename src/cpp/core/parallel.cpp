// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "core/parallel.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <process.h>
#define MDAPY_GETPID _getpid
#else
#include <unistd.h>
#define MDAPY_GETPID getpid
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#elif defined(_M_ARM64)
#include <intrin.h>
#endif

namespace mdapy {
namespace {

using Clock = std::chrono::steady_clock;
constexpr auto kMonitorInterval = std::chrono::milliseconds(50);
constexpr auto kWaitSlice = std::chrono::milliseconds(10);
// How long an idle thread keeps checking for work before it sleeps: the
// regions of one kernel follow each other within microseconds, and waking
// sleeping threads costs far more.
constexpr auto kSpin = std::chrono::microseconds(200);

// A pause inside a busy wait.
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#elif defined(_M_ARM64)
    __yield();
#endif
}

// Calls done() until it is true, for at most kSpin, polling every so often;
// returns whether it became true.
template <class Done, class Poll>
bool spin_until(Done&& done, Poll&& poll) {
    const auto start = Clock::now();
    for (unsigned k = 1;; ++k) {
        if (done()) {
            return true;
        }
        cpu_relax();
        if (k % 64 == 0) {
            poll();
            if (Clock::now() - start > kSpin) {
                return done();
            }
        }
    }
}

// True on pool workers and on a thread that is inside a parallel region, so a
// nested parallel_for runs inline instead of waiting for busy workers.
thread_local bool tls_in_region = false;

class RegionFlag {
public:
    RegionFlag() : previous_(tls_in_region) { tls_in_region = true; }
    ~RegionFlag() { tls_in_region = previous_; }
    RegionFlag(const RegionFlag&) = delete;
    RegionFlag& operator=(const RegionFlag&) = delete;

private:
    bool previous_;
};

// The chunks [next, end) of one party of a job: its own at first, which it
// takes one at a time, and which the others take once theirs are done; and
// the items the party has finished, which only it writes.
struct alignas(64) Share {
    std::atomic<std::int64_t> next{0};
    std::int64_t end = 0;
    alignas(64) std::atomic<std::int64_t> done{0};
};

struct Job {
    // read by every chunk, written before the job starts
    const std::function<void(std::int64_t, std::int64_t)>* body = nullptr;
    const CancelToken* token = nullptr;
    std::int64_t n = 0;
    std::int64_t grain = 1;
    std::int64_t nchunks = 0;
    Share* shares = nullptr;
    int parties = 1;
    std::int64_t settled = 0;  // the items finished, once the parties have left
    // written while the job runs
    alignas(64) std::atomic<bool> stop{false};
    std::mutex error_mutex;
    std::exception_ptr error;

    // Splits the chunks into equal ranges, one for each of the parties.
    void share(Share* ranges, int count) {
        shares = ranges;
        parties = count;
        for (int p = 0; p < count; ++p) {
            ranges[p].next.store(nchunks * p / count, std::memory_order_relaxed);
            ranges[p].end = nchunks * (p + 1) / count;
            ranges[p].done.store(0, std::memory_order_relaxed);
        }
    }

    // The items finished so far.
    std::int64_t done() const {
        if (shares == nullptr) {
            return settled;
        }
        std::int64_t sum = 0;
        for (int p = 0; p < parties; ++p) {
            sum += shares[p].done.load(std::memory_order_relaxed);
        }
        return sum;
    }

    // Keeps the count of finished items once every party has left, so that
    // the shares can serve the next job.
    void settle() {
        settled = done();
        shares = nullptr;
    }

    // Runs the chunks of party own, then those left of the others, calling
    // after() after each; stops early when the job is stopped.
    template <class After>
    void run_chunks(int own, After&& after) {
        std::atomic<std::int64_t>& finished = shares[own].done;
        for (int k = 0; k < parties; ++k) {
            Share& share = shares[(own + k) % parties];
            // a plain read first, so that taking nothing from a share
            // others have finished does not write to it
            while (share.next.load(std::memory_order_relaxed) < share.end) {
                if (stop.load(std::memory_order_relaxed)) {
                    return;
                }
                if (token != nullptr && token->cancelled()) {
                    stop.store(true, std::memory_order_relaxed);
                    return;
                }
                const std::int64_t chunk = share.next.fetch_add(1, std::memory_order_relaxed);
                if (chunk >= share.end) {
                    break;
                }
                const std::int64_t begin = chunk * grain;
                const std::int64_t end = std::min(n, begin + grain);
                try {
                    (*body)(begin, end);
                } catch (...) {
                    fail(std::current_exception());
                    return;
                }
                finished.store(finished.load(std::memory_order_relaxed) + (end - begin),
                               std::memory_order_relaxed);
                after();
            }
        }
    }

    // Records the first error and stops the job.
    void fail(std::exception_ptr e) {
        {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) {
                error = std::move(e);
            }
        }
        stop.store(true, std::memory_order_relaxed);
    }

    bool finished() const {
        return done() >= n;
    }
};

// Calls the monitor at most once per kMonitorInterval and stops the job when
// it asks for cancellation.
class MonitorClock {
public:
    explicit MonitorClock(const Control& control) : control_(control) {}

    void poll(Job& job, bool force = false) {
        if (!control_.monitor) {
            return;
        }
        const auto now = Clock::now();
        if (!force && now - last_ < kMonitorInterval) {
            return;
        }
        last_ = now;
        // The monitor runs while workers may still use the job, so it must
        // never unwind through the pool: turn its exceptions into a job error.
        try {
            if (!control_.monitor(job.done(), job.n)) {
                job.stop.store(true, std::memory_order_relaxed);
            }
        } catch (...) {
            job.fail(std::current_exception());
        }
    }

private:
    const Control& control_;
    Clock::time_point last_ = Clock::now();
};

// Workers that help the calling thread with its regions. A region is
// published as a new generation of one atomic word, which also says how many
// workers take part; idle workers spin on it for a while, then each sleeps on
// a condition of its own, so that waking them never makes them queue for a
// lock.
class ThreadPool {
public:
    explicit ThreadPool(int nworkers)
        : sleepers_(new Sleeper[static_cast<std::size_t>(nworkers)]),
          shares_(new Share[static_cast<std::size_t>(nworkers) + 1]) {
        threads_.reserve(static_cast<std::size_t>(nworkers));
        for (int i = 0; i < nworkers; ++i) {
            threads_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~ThreadPool() {
        shutdown_.store(true);
        wake(size());
        for (auto& thread : threads_) {
            thread.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return static_cast<int>(threads_.size()); }

    // Runs the job on the calling thread and as many workers as it has
    // chunks beyond one, and returns once each of them has left the job. The
    // calling thread polls the monitor between its chunks and while it waits.
    void run(Job& job, MonitorClock& clock) {
        const auto helpers = static_cast<int>(std::min<std::int64_t>(size(), job.nchunks - 1));
        // the calling thread is the last party
        job.share(shares_.get(), helpers + 1);
        job_.store(&job, std::memory_order_relaxed);
        const std::uint64_t generation = (state_.load(std::memory_order_relaxed) >> kHelperBits) + 1;
        state_.store((generation << kHelperBits) | static_cast<std::uint64_t>(helpers));
        wake(helpers);
        job.run_chunks(helpers, [&] { clock.poll(job); });
        const auto finished = [&] {
            for (int i = 0; i < helpers; ++i) {
                if (sleepers_[static_cast<std::size_t>(i)].finished.load() != generation) {
                    return false;
                }
            }
            return true;
        };
        if (!spin_until(finished, [&] { clock.poll(job); })) {
            caller_waiting_.store(true);
            std::unique_lock<std::mutex> lock(done_mutex_);
            while (!finished()) {
                done_.wait_for(lock, kWaitSlice, finished);
                lock.unlock();
                clock.poll(job);
                lock.lock();
            }
            caller_waiting_.store(false);
        }
        job.settle();
    }

private:
    static constexpr int kHelperBits = 24;
    static constexpr std::uint64_t kHelpers = (std::uint64_t{1} << kHelperBits) - 1;

    struct alignas(64) Sleeper {
        std::mutex mutex;
        std::condition_variable wake;
        std::atomic<bool> asleep{false};
        // the last generation the worker has left, which only it writes
        alignas(64) std::atomic<std::uint64_t> finished{0};
    };

    // Wakes the first count workers where they sleep.
    void wake(int count) {
        if (sleepers_count_.load() == 0) {
            // none was asleep when the new state was published: each sees it
            return;
        }
        for (int i = 0; i < count; ++i) {
            Sleeper& s = sleepers_[static_cast<std::size_t>(i)];
            if (s.asleep.load()) {
                // taken between the new state and the notice, so that a
                // worker cannot miss both
                { std::lock_guard<std::mutex> lock(s.mutex); }
                s.wake.notify_one();
            }
        }
    }

    void worker_loop(int index) {
        tls_in_region = true;
        Sleeper& sleeper = sleepers_[static_cast<std::size_t>(index)];
        std::uint64_t seen = 0;
        const auto news = [&] {
            return shutdown_.load() || (state_.load() >> kHelperBits) != seen;
        };
        for (;;) {
            if (!spin_until(news, [] {})) {
                sleepers_count_.fetch_add(1);
                {
                    std::unique_lock<std::mutex> lock(sleeper.mutex);
                    sleeper.asleep.store(true);
                    sleeper.wake.wait(lock, news);
                    sleeper.asleep.store(false);
                }
                sleepers_count_.fetch_sub(1);
            }
            if (shutdown_.load()) {
                return;
            }
            const std::uint64_t state = state_.load();
            seen = state >> kHelperBits;
            if (static_cast<std::uint64_t>(index) < (state & kHelpers)) {
                job_.load(std::memory_order_relaxed)->run_chunks(index, [] {});
                sleeper.finished.store(seen);
                if (caller_waiting_.load()) {
                    std::lock_guard<std::mutex> lock(done_mutex_);
                    done_.notify_all();
                }
            }
        }
    }

    std::vector<std::thread> threads_;
    std::unique_ptr<Sleeper[]> sleepers_;
    std::unique_ptr<Share[]> shares_;  // of each worker, then of the calling thread
    // what idle workers watch: the generation of the region (high bits) and
    // its number of workers, and the job
    alignas(64) std::atomic<std::uint64_t> state_{0};
    std::atomic<Job*> job_{nullptr};
    std::atomic<bool> shutdown_{false};
    alignas(64) std::atomic<int> sleepers_count_{0};  // workers asleep or about to be
    alignas(64) std::atomic<bool> caller_waiting_{false};  // asleep until the workers leave
    std::mutex done_mutex_;
    std::condition_variable done_;
};

// Atomic rather than mutex-protected so a fork() can never leave it locked.
std::atomic<int> g_num_threads{0};  // 0 means "not set yet"
// Held by the thread that currently owns the pool.
std::mutex g_pool_mutex;
// Never deleted at exit: joining threads during process teardown can deadlock
// (for example under the Windows loader lock), and the OS reclaims them anyway.
ThreadPool* g_pool = nullptr;
long g_pool_pid = 0;

// Returns a pool with nworkers threads besides the calling one. Must be
// called with g_pool_mutex held.
ThreadPool* pool_with(int nworkers) {
    const long pid = static_cast<long>(MDAPY_GETPID());
    if (g_pool != nullptr && g_pool_pid != pid) {
        // Forked child: the parent's worker threads do not exist here. Drop
        // the stale pool without touching its threads.
        g_pool = nullptr;
    }
    if (g_pool != nullptr && g_pool->size() != nworkers) {
        delete g_pool;
        g_pool = nullptr;
    }
    if (g_pool == nullptr) {
        g_pool = new ThreadPool(nworkers);
        g_pool_pid = pid;
    }
    return g_pool;
}

// Runs every chunk on the calling thread, polling the monitor between chunks.
void run_inline(Job& job, MonitorClock& clock) {
    Share all;
    job.share(&all, 1);
    job.run_chunks(0, [&] { clock.poll(job); });
    job.settle();
}

}  // namespace

int hardware_threads() {
    const unsigned int n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : static_cast<int>(n);
}

int get_num_threads() {
    const int n = g_num_threads.load(std::memory_order_relaxed);
    return n > 0 ? n : hardware_threads();
}

void set_num_threads(int n) {
    if (n < 1) {
        throw std::invalid_argument("number of threads must be at least 1");
    }
    g_num_threads.store(n, std::memory_order_relaxed);
}

std::int64_t default_grain(std::int64_t n) {
    constexpr std::int64_t kTargetChunks = 512;
    return std::max<std::int64_t>(1, (n + kTargetChunks - 1) / kTargetChunks);
}

void prefix_sum(std::int64_t* values, std::int64_t count) {
    constexpr std::int64_t kBlock = std::int64_t{1} << 16;
    if (count <= 2 * kBlock) {
        for (std::int64_t i = 1; i < count; ++i) {
            values[i] += values[i - 1];
        }
        return;
    }
    // the sums within each block, then the sum before each block added
    const std::int64_t blocks = (count + kBlock - 1) / kBlock;
    std::vector<std::int64_t> before(static_cast<std::size_t>(blocks), 0);
    parallel_for(count, kBlock, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin + 1; i < end; ++i) {
            values[i] += values[i - 1];
        }
        before[static_cast<std::size_t>(begin / kBlock)] = values[end - 1];
    });
    std::int64_t sum = 0;
    for (auto& block : before) {
        const std::int64_t own = block;
        block = sum;
        sum += own;
    }
    parallel_for(count, kBlock, [&](std::int64_t begin, std::int64_t end) {
        const std::int64_t add = before[static_cast<std::size_t>(begin / kBlock)];
        for (std::int64_t i = begin; i < end; ++i) {
            values[i] += add;
        }
    });
}

void parallel_for(std::int64_t n, std::int64_t grain,
                  const std::function<void(std::int64_t, std::int64_t)>& body,
                  const Control& control) {
    MonitorClock clock(control);
    Job job;
    job.body = &body;
    job.token = control.cancel;
    job.n = std::max<std::int64_t>(n, 0);
    job.grain = grain > 0 ? grain : default_grain(job.n);
    job.nchunks = (job.n + job.grain - 1) / job.grain;

    if (job.n > 0) {
        const int nthreads = get_num_threads();
        const bool serial = tls_in_region || nthreads == 1 || job.nchunks == 1;
        RegionFlag region;
        if (serial) {
            run_inline(job, clock);
        } else {
            std::unique_lock<std::mutex> owner(g_pool_mutex, std::try_to_lock);
            if (owner.owns_lock()) {
                pool_with(nthreads - 1)->run(job, clock);
            } else {
                // Another thread is using the pool: run serially rather than wait.
                run_inline(job, clock);
            }
        }
    }

    if (job.error) {
        std::rethrow_exception(job.error);
    }
    if (!job.finished()) {
        throw Cancelled();
    }
    // Final report with done == total. A stop request at this point is moot:
    // the work is complete.
    clock.poll(job, /*force=*/true);
    if (job.error) {
        std::rethrow_exception(job.error);
    }
}

}  // namespace mdapy
