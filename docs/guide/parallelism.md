# Parallelism

mdapy runs its compiled kernels on a thread pool of its own. This page shows
how to choose the number of threads, how to follow and stop long
computations, and how mdapy behaves next to PyTorch, NumPy and LAMMPS.

```python
import mdapy

mdapy.get_num_threads()  # default: number of logical CPUs
mdapy.set_num_threads(8)  # for the rest of the process

with mdapy.num_threads(2):  # only inside this block
    ...
```

## Choosing the number of threads

| Name | Type | Default | Meaning |
|---|---|---|---|
| `set_num_threads(n)` | function | | Use `n` threads from now on. |
| `num_threads(n)` | context manager | | Use `n` threads inside a `with` block, then restore the previous value. |
| `get_num_threads()` | function | | The current number of threads. |
| `MDAPY_NUM_THREADS` | environment variable | number of logical CPUs | Initial number of threads, read once when mdapy is imported. |

`n` must be a positive integer; anything else raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter). An
invalid `MDAPY_NUM_THREADS` makes `import mdapy` fail with the same error,
rather than silently falling back to a default.

The setting is global to the process, not to the calling Python thread.

## Progress and cancellation

Computations that can take long accept two keyword-only arguments. (This is
the convention for every such computation; in 2.0.0.dev0 none of them is
available yet.)

| Name | Type | Default | Meaning |
|---|---|---|---|
| `progress` | callable or `None` | `None` | Called as `progress(done, total)` with the number of finished and total work items. |
| `cancel` | [`CancelToken`](../reference/api.md#mdapy.CancelToken) or `None` | `None` | Stops the computation once `token.cancel()` is called. |

`progress` is called on the thread that started the computation, about every
50 ms while it runs, and once more with `done == total` when it finishes.
`done` never decreases. If `progress` raises an exception, the computation
stops and that exception propagates to the caller.

A cancelled computation raises
[`Cancelled`](../reference/api.md#mdapy.errors.Cancelled). The token can be
cancelled from any thread, including from inside `progress`, and can be
reused after `token.reset()`.

```python
token = mdapy.CancelToken()
# in a GUI: connect a "Stop" button to token.cancel
try:
    result = some_analysis.compute(system, progress=report, cancel=token)
except mdapy.errors.Cancelled:
    ...
```

Pressing Ctrl-C stops a running kernel and raises `KeyboardInterrupt` as
usual.

## Reproducible results

Results do not depend on the number of threads. Work is split into chunks
whose size depends only on the problem size, and sums over chunks are always
added in the same order. Running the same computation with 1 or 64 threads
therefore gives bit-for-bit identical numbers.

## Working with PyTorch, NumPy and LAMMPS

mdapy does not use OpenMP and loads no OpenMP runtime. Two copies of an
OpenMP runtime in one process (for example mdapy's and PyTorch's) is a common
cause of crashes such as `OMP: Error #15` on macOS; this cannot happen with
mdapy, so no workaround such as `KMP_DUPLICATE_LIB_OK` is needed.

Other libraries keep their own thread settings:

- PyTorch: `torch.set_num_threads`.
- NumPy (BLAS): the `OMP_NUM_THREADS` or `OPENBLAS_NUM_THREADS` environment
  variables.
- polars: the `POLARS_MAX_THREADS` environment variable, which must be set
  before polars is first imported.

When several libraries compute at the same time, keep the total number of
threads close to the number of CPU cores.

## Calling mdapy from several threads

Compiled kernels release the Python global interpreter lock (GIL), so other
Python threads, for example a GUI event loop, keep running while mdapy
computes.

The pool serves one computation at a time. If a second Python thread starts a
computation while the pool is busy, that computation runs on its own thread
instead of waiting. It gives the same result, only without parallel speed-up.

Processes started with `multiprocessing` work with both the `fork` and
`spawn` start methods; a forked child creates its own pool on first use.

## Validation

The test suite (`tests/core/`) checks that:

- sums computed in parallel equal a sequential reference bit for bit, and are
  identical for 1, 2, 3, 4 and 8 threads;
- the configured number of threads is the maximum number used;
- nested parallel regions give exact results;
- exceptions raised inside a kernel or by `progress` reach the caller, and the
  pool keeps working afterwards;
- cancellation through a token (set in advance, from `progress`, or from
  another thread) and Ctrl-C stop a running kernel within a fraction of its
  run time;
- the GIL is released while kernels run, concurrent calls from several Python
  threads give correct results, and the pool works in a forked child process.

## Implementation notes

A computation over `n` items is split into chunks of a fixed size, by default
`ceil(n / 512)` items, so there are at most 512 chunks. Worker threads take
chunks one at a time. The calling thread does no chunk work itself: it waits,
reports progress and checks for cancellation and Ctrl-C. Nested parallel
regions, and regions started while the pool is busy, run on the calling
thread.
