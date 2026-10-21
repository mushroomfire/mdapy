// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "core/parallel.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
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

namespace mdapy {
namespace {

using Clock = std::chrono::steady_clock;
constexpr auto kMonitorInterval = std::chrono::milliseconds(50);
constexpr auto kWaitSlice = std::chrono::milliseconds(10);

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

struct Job {
    const std::function<void(std::int64_t, std::int64_t)>* body = nullptr;
    const CancelToken* token = nullptr;
    std::int64_t n = 0;
    std::int64_t grain = 1;
    std::int64_t nchunks = 0;
    std::atomic<std::int64_t> next{0};
    std::atomic<std::int64_t> done{0};
    std::atomic<bool> stop{false};
    std::mutex error_mutex;
    std::exception_ptr error;

    // Claims and runs one chunk. Returns false when no chunk is left or the
    // job has been stopped.
    bool run_one() {
        if (stop.load(std::memory_order_relaxed)) {
            return false;
        }
        if (token != nullptr && token->cancelled()) {
            stop.store(true, std::memory_order_relaxed);
            return false;
        }
        const std::int64_t chunk = next.fetch_add(1, std::memory_order_relaxed);
        if (chunk >= nchunks) {
            return false;
        }
        const std::int64_t begin = chunk * grain;
        const std::int64_t end = std::min(n, begin + grain);
        try {
            (*body)(begin, end);
        } catch (...) {
            fail(std::current_exception());
            return false;
        }
        done.fetch_add(end - begin, std::memory_order_relaxed);
        return true;
    }

    void run_chunks() {
        while (run_one()) {
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
        return done.load(std::memory_order_relaxed) >= n;
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
            if (!control_.monitor(job.done.load(std::memory_order_relaxed), job.n)) {
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

class ThreadPool {
public:
    explicit ThreadPool(int nworkers) {
        threads_.reserve(static_cast<std::size_t>(nworkers));
        for (int i = 0; i < nworkers; ++i) {
            threads_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        start_.notify_all();
        for (auto& thread : threads_) {
            thread.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return static_cast<int>(threads_.size()); }

    // Runs the job on the workers. The calling thread only polls the monitor
    // and returns once every worker has left the job.
    void run(Job& job, MonitorClock& clock) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = &job;
            running_ = size();
            ++generation_;
        }
        start_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        while (running_ > 0) {
            finish_.wait_for(lock, kWaitSlice, [this] { return running_ == 0; });
            if (running_ == 0) {
                break;
            }
            lock.unlock();
            clock.poll(job);
            lock.lock();
        }
        job_ = nullptr;
    }

private:
    void worker_loop() {
        tls_in_region = true;
        std::uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            start_.wait(lock, [&] { return shutdown_ || generation_ != seen; });
            if (shutdown_) {
                return;
            }
            seen = generation_;
            Job* job = job_;
            lock.unlock();
            job->run_chunks();
            lock.lock();
            if (--running_ == 0) {
                finish_.notify_all();
            }
        }
    }

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable finish_;
    Job* job_ = nullptr;
    std::uint64_t generation_ = 0;
    int running_ = 0;
    bool shutdown_ = false;
};

// Atomic rather than mutex-protected so a fork() can never leave it locked.
std::atomic<int> g_num_threads{0};  // 0 means "not set yet"
// Held by the thread that currently owns the pool.
std::mutex g_pool_mutex;
// Never deleted at exit: joining threads during process teardown can deadlock
// (for example under the Windows loader lock), and the OS reclaims them anyway.
ThreadPool* g_pool = nullptr;
long g_pool_pid = 0;

// Returns a pool with nworkers threads. Must be called with g_pool_mutex held.
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
    while (job.run_one()) {
        clock.poll(job);
    }
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
                pool_with(nthreads)->run(job, clock);
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
