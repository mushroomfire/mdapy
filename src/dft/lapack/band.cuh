// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The tridiagonalization in two stages, for the large matrices: part of
// dft/lapack/lapack.cu, which includes it among its kernels.
//
// 1. Dense to band. The columns below a band of half width b are reduced a
//    panel of b columns at a time by a QR factorization (one launch a column,
//    which touches the panel alone), and the trailing matrix takes the b
//    reflectors of the panel from both sides as products of Blas. The memory
//    of the whole matrix is passed three times a panel, not once a column.
// 2. Band to tridiagonal, by chasing bulges. Sweep s removes the column s below
//    its subdiagonal with a reflector of b rows, which fills a bulge below the
//    band; task k of the sweep moves it b rows down with the next reflector.
//    Task (s + 1, k) needs (s, k + 1) and no later one, so that all tasks with
//    the same 2 s + k are independent: a launch each, 2 n launches in all.
// Then the eigenvectors of the tridiagonal matrix take the reflectors of the
// chase (a kernel: a thread to each column, with the reflectors of a group of
// sweeps at one k) and those of the panels (products of Blas). The kernels of
// the chase are compiled for the half widths 8, 16 and 32.

// LAPACK's larfg: of the vector (alpha, x) with tail the sum of the squares of
// x, the reflector h = 1 - tau v v^H, v = (1, scale x), with h^H (alpha, x) =
// (beta, 0).
template <class S>
__device__ __forceinline__ void reflector_of(S alpha, typename RealOf<S>::type tail, S& tau, S& scale,
                                             typename RealOf<S>::type& beta) {
    using R = typename RealOf<S>::type;
    tau = from_real(S{}, static_cast<R>(0));
    scale = from_real(S{}, static_cast<R>(1));
    beta = re(alpha);
    if (tail != static_cast<R>(0) || im(alpha) != static_cast<R>(0)) {
        const R size = sqrt(re(alpha) * re(alpha) + im(alpha) * im(alpha) + tail);
        beta = re(alpha) >= static_cast<R>(0) ? -size : size;
        tau = mul(sub(from_real(S{}, beta), alpha), from_real(S{}, static_cast<R>(1) / beta));
        scale = inverse(sub(alpha, from_real(S{}, beta)));
    }
}

constexpr int kWide = 1024;  // the most threads of a block of band_column_kernel

// Column k of the QR factorization of the panel of the b columns from first,
// whose rows start at first + b: every block works out the reflector of the
// column (its pivot the row first + b + k) from the column as the launch
// before left it, and applies it to the column k + 1 + its index of the panel.
// The first block writes the reflector to reflectors, with tau and beta; the
// column itself is left as it was. The threads, a multiple of 32 up to kWide,
// share the rows, each asking for four numbers at a time.
template <class S>
__global__ void band_column_kernel(int n, int first, int b, int k, S* a, S* reflectors, S* tau,
                                   typename RealOf<S>::type* beta) {
    using R = typename RealOf<S>::type;
    __shared__ S partial[kWide + kWide / 32];
    const int t = static_cast<int>(threadIdx.x), step = static_cast<int>(blockDim.x), pivot = first + b + k;
    const S* x = a + static_cast<long long>(first + k) * n;
    const S zero = from_real(S{}, static_cast<R>(0)), one = from_real(S{}, static_cast<R>(1));
    R tail = static_cast<R>(0);
    for (int r = pivot + 1 + t; r < n; r += 4 * step) {
        S u[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) u[q] = r + q * step < n ? x[r + q * step] : zero;
#pragma unroll
        for (int q = 0; q < 4; ++q) tail += re(u[q]) * re(u[q]) + im(u[q]) * im(u[q]);
    }
    tail = re(wide_sum(partial, from_real(S{}, tail)));
    S reflection, scale;
    R size;
    reflector_of(x[pivot], tail, reflection, scale, size);
    if (blockIdx.x == 0) {
        S* v = reflectors + static_cast<long long>(first + k) * n;
        for (int r = pivot + t; r < n; r += step) v[r] = r == pivot ? one : mul(x[r], scale);
        if (t == 0) {
            tau[first + k] = reflection;
            beta[first + k] = size;
        }
    }
    const int j = k + 1 + static_cast<int>(blockIdx.x);
    if (j >= b) return;
    S* y = a + static_cast<long long>(first + j) * n;
    S sum = zero;
    for (int r = pivot + t; r < n; r += 4 * step) {
        S u[4], w[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int row = r + q * step;
            u[q] = row >= n ? zero : row == pivot ? one : mul(x[row], scale);
            w[q] = row >= n ? zero : y[row];
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) sum = add(sum, mul(cj(u[q]), w[q]));
    }
    const S factor = mul(cj(reflection), wide_sum(partial, sum));
    for (int r = pivot + t; r < n; r += 4 * step) {
        S u[4], w[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int row = r + q * step;
            u[q] = row >= n ? zero : row == pivot ? one : mul(x[row], scale);
            w[q] = row >= n ? zero : y[row];
        }
#pragma unroll
        for (int q = 0; q < 4; ++q)
            if (r + q * step < n) y[r + q * step] = sub(w[q], mul(factor, u[q]));
    }
}

// p = (v, x) and q = (x, v), width columns of each: of the product p q^H, the
// two halves of what a panel takes from the trailing matrix. v is stored with
// the rows of the matrix, x, p and q with theirs.
template <class S>
__global__ void pair_kernel(int rows, int width, int n, const S* __restrict__ v, const S* __restrict__ x,
                            S* __restrict__ p, S* __restrict__ q) {
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x, half = static_cast<long long>(rows) * width;
    if (e >= half) return;
    const S left = v[e % rows + (e / rows) * n], right = x[e];
    p[e] = left;
    p[half + e] = right;
    q[e] = right;
    q[half + e] = left;
}

// The band of a after its panels, band[i - j + j rows] = a_ij for
// 0 <= i - j <= b (the diagonal real) and zero below, where the chase needs
// room: the elements at b below the diagonal of the columns that got a
// reflector are its beta.
template <class S>
__global__ void band_kernel(int n, int b, int rows, const S* __restrict__ a,
                            const typename RealOf<S>::type* __restrict__ beta, S* __restrict__ band) {
    using R = typename RealOf<S>::type;
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e >= static_cast<long long>(n) * rows) return;
    const int j = static_cast<int>(e / rows), d = static_cast<int>(e % rows), i = j + d;
    S value = from_real(S{}, static_cast<R>(0));
    if (i < n && d <= b) {
        value = a[i + static_cast<long long>(j) * n];
        if (d == 0) value = from_real(S{}, re(value));
        // the panel of the column: it has reflectors while two rows or more are below it
        const int first = (j / b) * b, below = n - first - b;
        if (d == b && below >= 2 && j - first < below - 1) value = from_real(S{}, beta[j]);
    }
    band[e] = value;
}

// Task k = tick - 2 s of the sweep s = first + the index of the block, B
// threads (B the half width of the band). Its rows are the B from s + 1 + k B
// (fewer at the end of the matrix), with e the block of the band of those rows
// and the B columns before them, and d the diagonal block of the rows:
// - k > 0: the reflector of the task before of the sweep to e from the right,
//   which fills it;
// - the reflector of the first column of e (of the column s for k = 0), kept
//   as the rows of the column s of reflectors with its tau at k of the steps
//   of the sweep in taus; the column becomes (beta, 0);
// - the reflector to the other columns of e from the left, and to d from both
//   sides.
// A thread asks for the B numbers of its row or column at once and keeps them
// in registers: B is known to the compiler.
template <class S, int B>
__global__ void chase_kernel(int n, int rows, int tick, int first, int steps, S* band, S* reflectors, S* taus) {
    using R = typename RealOf<S>::type;
    __shared__ S v[B], p[B];
    const int i = static_cast<int>(threadIdx.x), s = first + static_cast<int>(blockIdx.x), k = tick - 2 * s;
    const int r0 = s + 1 + k * B, c0 = r0 - B, len = min(B, n - r0);
    const S zero = from_real(S{}, static_cast<R>(0));
    S* e = k > 0 ? band + static_cast<long long>(c0) * rows + B : band;  // e_ij at i - j + j rows
    S* d = band + static_cast<long long>(r0) * rows;                     // d_ij at i - j + j rows, i >= j
    S* column = k > 0 ? e : band + static_cast<long long>(s) * rows + 1;
    S x[B];
    if (k > 0) {
        if (i < B) p[i] = reflectors[static_cast<long long>(s) * n + c0 + i];
        if (i < len) {
#pragma unroll
            for (int j = 0; j < B; ++j) x[j] = e[i - j + static_cast<long long>(j) * rows];
        }
        __syncthreads();
        if (i < len) {
            S sum = zero;
#pragma unroll
            for (int j = 0; j < B; ++j) sum = add(sum, mul(x[j], p[j]));
            sum = mul(taus[static_cast<long long>(s) * steps + k - 1], sum);
#pragma unroll
            for (int j = 0; j < B; ++j) e[i - j + static_cast<long long>(j) * rows] = sub(x[j], mul(sum, cj(p[j])));
            v[i] = sub(x[0], mul(sum, cj(p[0])));
        }
    } else if (i < len) {
        v[i] = column[i];
    }
    __syncthreads();
    R tail = static_cast<R>(0);
#pragma unroll
    for (int j = 1; j < B; ++j)
        if (j < len) tail += re(v[j]) * re(v[j]) + im(v[j]) * im(v[j]);
    S tau, scale;
    R beta;
    reflector_of(v[0], tail, tau, scale, beta);
    const S mine = i == 0 ? from_real(S{}, static_cast<R>(1)) : i < len ? mul(v[i], scale) : zero;
    __syncthreads();
    if (i < len) {
        v[i] = mine;
        reflectors[static_cast<long long>(s) * n + r0 + i] = mine;
        column[i] = i == 0 ? from_real(S{}, beta) : zero;
    }
    if (i == 0) taus[static_cast<long long>(s) * steps + k] = tau;
    // the column i of e (a thread to each but the first), and the row i of d
    if (k > 0 && i > 0 && i < B) {
        const S* y = e + static_cast<long long>(i) * rows - i;
#pragma unroll
        for (int r = 0; r < B; ++r) x[r] = r < len ? y[r] : zero;
    }
    __syncthreads();
    if (k > 0 && i > 0 && i < B) {
        S* y = e + static_cast<long long>(i) * rows - i;
        S sum = zero;
#pragma unroll
        for (int r = 0; r < B; ++r)
            if (r < len) sum = add(sum, mul(cj(v[r]), x[r]));
        sum = mul(cj(tau), sum);
#pragma unroll
        for (int r = 0; r < B; ++r)
            if (r < len) y[r] = sub(x[r], mul(sum, v[r]));
    }
    // d - v w^H - w v^H, w = tau p - (|tau|^2 / 2) (v^H p) v and p = d v, of
    // which the lower triangle is kept: a thread to each row
    if (i < len) {
#pragma unroll
        for (int j = 0; j < B; ++j)
            x[j] = j >= len ? zero : j <= i ? d[i - j + static_cast<long long>(j) * rows]
                                             : cj(d[j - i + static_cast<long long>(i) * rows]);
        S sum = zero;
#pragma unroll
        for (int j = 0; j < B; ++j) sum = add(sum, mul(x[j], v[j]));
        p[i] = sum;
    }
    __syncthreads();
    R both = static_cast<R>(0);
#pragma unroll
    for (int j = 0; j < B; ++j)
        if (j < len) both += re(v[j]) * re(p[j]) + im(v[j]) * im(p[j]);
    both *= static_cast<R>(-0.5) * (re(tau) * re(tau) + im(tau) * im(tau));
    const S w = i < len ? add(mul(tau, p[i]), mul(from_real(S{}, both), v[i])) : zero;
    __syncthreads();
    if (i < len) p[i] = w;
    __syncthreads();
    if (i >= len) return;
#pragma unroll
    for (int j = 0; j < B; ++j)
        if (j <= i) {
            const S value = sub(x[j], add(mul(v[i], cj(p[j])), mul(w, cj(v[j]))));
            d[i - j + static_cast<long long>(j) * rows] = j == i ? from_real(S{}, re(value)) : value;
        }
}

// The diagonal and the subdiagonal of the band after the chase.
template <class S>
__global__ void band_ends_kernel(int n, int rows, const S* __restrict__ band, typename RealOf<S>::type* __restrict__ d,
                                 typename RealOf<S>::type* __restrict__ e) {
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= n) return;
    d[j] = re(band[static_cast<long long>(j) * rows]);
    e[j] = j + 1 < n ? re(band[static_cast<long long>(j) * rows + 1]) : 0;
}

constexpr int kChaseColumns = 64;  // of the eigenvectors, to a block of chase_back_kernel

// The reflectors of the chase applied to the eigenvectors z (n, any): a block
// to kChaseColumns columns (blockIdx.x) and to the reflectors of the task k of
// the group sweeps from (first + blockIdx.y) group, k = wave - (last - that
// group), which take B + group - 1 rows of z; a thread to each column.
// Reflectors of sweeps before others are applied after them: the thread holds
// the B rows of a reflector in registers, and from a sweep to the one before
// it gives the last of them back to z and takes the row above the first.
template <class S, int B>
__global__ void chase_back_kernel(int n, int group, int steps, int wave, int first, int last, int columns,
                                  const S* __restrict__ reflectors, const S* __restrict__ taus, S* z) {
    using R = typename RealOf<S>::type;
    const int g = first + static_cast<int>(blockIdx.y), k = wave - (last - g);
    const int s0 = g * group, row0 = s0 + 1 + k * B;
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (k < 0 || row0 >= n || c >= columns) return;
    const S zero = from_real(S{}, static_cast<R>(0));
    S* column = z + static_cast<long long>(c) * n + row0;
    const int sweeps = min(group, n - 1 - s0), rest = n - row0;  // the rows of z from row0
    S x[B];
#pragma unroll
    for (int r = 0; r < B; ++r) x[r] = sweeps - 1 + r < rest ? column[sweeps - 1 + r] : zero;
    for (int q = sweeps - 1; q >= 0; --q) {
        const int s = s0 + q, len = min(B, rest - q);
        if (len > 0) {
            const S* v = reflectors + static_cast<long long>(s) * n + row0 + q;
            S sum = zero;
            for (int r = 0; r < B; r += 8) {
                S u[8];
#pragma unroll
                for (int a = 0; a < 8; ++a) u[a] = r + a < len ? v[r + a] : zero;
#pragma unroll
                for (int a = 0; a < 8; ++a) sum = add(sum, mul(cj(u[a]), x[r + a]));
            }
            sum = mul(taus[static_cast<long long>(s) * steps + k], sum);
            for (int r = 0; r < B; r += 8) {
                S u[8];
#pragma unroll
                for (int a = 0; a < 8; ++a) u[a] = r + a < len ? v[r + a] : zero;
#pragma unroll
                for (int a = 0; a < 8; ++a) x[r + a] = sub(x[r + a], mul(sum, u[a]));
            }
        }
        if (q + B - 1 < rest) column[q + B - 1] = x[B - 1];
#pragma unroll
        for (int r = B - 1; r > 0; --r) x[r] = x[r - 1];
        if (q > 0) x[0] = column[q - 1];
    }
    // the rows 0 ... B - 2, which the shift left at 1 ... B - 1
#pragma unroll
    for (int r = 1; r < B; ++r)
        if (r - 1 < rest) column[r - 1] = x[r];
}

// What the two stages leave for the eigenvectors.
template <class T>
struct TwoStage {
    int b = 0, steps = 0;
    DeviceArray<T> reflectors;  // of the panels: the column j from the row j + b
    DeviceArray<T> factors;     // (b, b) of each panel, one after the other
    DeviceArray<T> taus;        // (steps) of each sweep of the chase
};

// a (n, n, both triangles) to the tridiagonal matrix of diagonal d and
// subdiagonal e (on the host), through a band of half width b. On return the
// columns of a hold the reflectors of the chase (those of the sweep s below its
// diagonal).
template <class T>
void two_stage_tridiagonal(Blas& blas, T* a, int n, int b, TwoStage<T>& kept, std::vector<double>& d,
                           std::vector<double>& e) {
    using S = std::remove_pointer_t<decltype(own(a))>;
    using R = typename RealOf<S>::type;
    const std::size_t count = static_cast<std::size_t>(n), width = static_cast<std::size_t>(b);
    const T one = unit(T{}, 1.0), zero = unit(T{}, 0.0), minus = unit(T{}, -1.0), half = unit(T{}, -0.5);
    const int panels = n > b + 1 ? (n - b - 2) / b + 1 : 0;  // those with two rows or more
    kept.b = b;
    kept.steps = (n - 2) / b + 1;
    kept.reflectors.resize(count * count);
    kept.reflectors.zero();
    kept.factors.resize(static_cast<std::size_t>(std::max(panels, 1)) * width * width);
    DeviceArray<T> tau(count), w(4 * count * width), y(count * width), gram(3 * width * width);
    DeviceArray<R> beta(count), ends(2 * count);
    {
        MDAPY_SPAN("band");
        for (int panel = 0; panel < panels; ++panel) {
            const int first = panel * b, from = first + b, below = n - from, made = std::min(b, below - 1);
            unsigned threads = 128;  // eight rows to each or fewer, up to kWide
            while (threads < kWide && static_cast<int>(threads) * 8 < below) threads *= 2;
            for (int k = 0; k < made; ++k)
                band_column_kernel<S><<<static_cast<unsigned>(std::max(b - 1 - k, 1)), threads>>>(
                    n, first, b, k, own(a), own(kept.reflectors.data()), own(tau.data()), beta.data());
            launched("a panel of the reduction to a band");
            // q = 1 - v t v^H of the panel; the trailing matrix becomes q^H a q =
            // a - v x^H - x v^H with x = y - v (t^H v^H y) / 2 and y = a v t
            T* v = kept.reflectors.data() + static_cast<std::size_t>(first) * count + from;
            T* t = kept.factors.data() + static_cast<std::size_t>(panel) * width * width;
            T* trailing = a + static_cast<std::size_t>(from) * count + from;
            T* inner = gram.data() + width * width;
            T* middle = inner + width * width;
            blas.gemm(Op::C, Op::N, made, made, below, one, v, n, v, n, zero, gram.data(), made);
            block_factor_kernel<S><<<1, kReflectors>>>(made, own(tau.data()) + first, own(gram.data()), own(t));
            launched("the factor of a panel");
            blas.gemm(Op::N, Op::N, below, made, below, one, trailing, n, v, n, zero, w.data(), below);
            blas.gemm(Op::N, Op::N, below, made, made, one, w.data(), below, t, made, zero, y.data(), below);
            blas.gemm(Op::C, Op::N, made, made, below, one, v, n, y.data(), below, zero, inner, made);
            blas.gemm(Op::C, Op::N, made, made, made, one, t, made, inner, made, zero, middle, made);
            blas.gemm(Op::N, Op::N, below, made, made, half, v, n, middle, made, one, y.data(), below);
            // v x^H + x v^H as one product, which passes the trailing matrix once
            pair_kernel<S><<<blocks(static_cast<long long>(below) * made), kThreads>>>(
                below, made, n, own(v), own(y.data()), own(w.data()), own(w.data()) + 2 * count * width);
            launched("the two halves of the update of a panel");
            blas.gemm(Op::N, Op::C, below, below, 2 * made, minus, w.data(), below, w.data() + 2 * count * width, below,
                      one, trailing, n);
        }
    }
    const int rows = 2 * b;
    DeviceArray<T> band(count * static_cast<std::size_t>(rows));
    {
        MDAPY_SPAN("chase");
        band_kernel<S><<<blocks(static_cast<long long>(n) * rows), kThreads>>>(n, b, rows, own(a), beta.data(),
                                                                               own(band.data()));
        launched("the band of a matrix");
        kept.taus.resize(count * static_cast<std::size_t>(kept.steps));
        kept.taus.zero();
        check_cuda(cudaMemset(a, 0, count * count * sizeof(T)), "cudaMemset");
        // the sweeps of a tick: those from 0 to n - 2 with a task tick - 2 s whose
        // first row s + 1 + (tick - 2 s) b is in the matrix
        for (int tick = 0; tick <= 2 * (n - 2); ++tick) {
            const int reach = tick * b - (n - 2);
            const int low = reach > 0 ? (reach + 2 * b - 2) / (2 * b - 1) : 0, high = std::min(tick / 2, n - 2);
            if (low > high) continue;
            const unsigned sweeps = static_cast<unsigned>(high - low + 1);
            S* reflectors = own(a);
#define MDAPY_CHASE(B)                                                                                        \
    case B:                                                                                                   \
        chase_kernel<S, B><<<sweeps, std::max(B, 32)>>>(n, rows, tick, low, kept.steps, own(band.data()),     \
                                                        reflectors, own(kept.taus.data()));                   \
        break;
            switch (b) {
                MDAPY_CHASE(8)
                MDAPY_CHASE(16)
                MDAPY_CHASE(32)
            }
#undef MDAPY_CHASE
        }
        band_ends_kernel<S><<<blocks(n), kThreads>>>(n, rows, own(band.data()), ends.data(), ends.data() + count);
        launched("the chase of the bulges");
    }
    std::vector<R> host(2 * count);
    ends.download(host.data(), host.size());
    d.assign(host.begin(), host.begin() + n);
    e.assign(host.begin() + n, host.begin() + 2 * n - 1);
}

// z (n, columns), the eigenvectors of the tridiagonal matrix, to those of the
// matrix: the reflectors of the chase (the columns of chased, as
// two_stage_tridiagonal leaves them), then those of the panels.
template <class T>
void two_stage_back(Blas& blas, T* chased, int n, TwoStage<T>& kept, T* z, int columns, int group) {
    using S = std::remove_pointer_t<decltype(own(z))>;
    const int b = kept.b;
    const std::size_t count = static_cast<std::size_t>(n), width = static_cast<std::size_t>(b);
    {
        MDAPY_SPAN("chased");
        // the groups of sweeps from the last, the tasks of each from the first:
        // the task k of a group needs its task k - 1 and the task k of the group
        // after, so that a wave takes the task wave - (last - g) of every group g
        const int last = (n - 2) / group;
        const unsigned tiles = blocks(columns, kChaseColumns);
        for (int wave = 0; wave < last + kept.steps; ++wave) {
            const int first = std::max(last - wave, 0);
            const dim3 grid(tiles, static_cast<unsigned>(last - first + 1));
#define MDAPY_CHASE(B)                                                                                        \
    case B:                                                                                                   \
        chase_back_kernel<S, B><<<grid, kChaseColumns>>>(n, group, kept.steps, wave, first, last, columns,    \
                                                         own(chased), own(kept.taus.data()), own(z));         \
        break;
            switch (b) {
                MDAPY_CHASE(8)
                MDAPY_CHASE(16)
                MDAPY_CHASE(32)
            }
#undef MDAPY_CHASE
        }
        launched("the reflectors of the chase");
    }
    MDAPY_SPAN("back");
    const T one = unit(T{}, 1.0), zero = unit(T{}, 0.0), minus = unit(T{}, -1.0);
    const int panels = n > b + 1 ? (n - b - 2) / b + 1 : 0;
    DeviceArray<T> w(static_cast<std::size_t>(columns) * width), tw(static_cast<std::size_t>(columns) * width);
    for (int panel = panels - 1; panel >= 0; --panel) {
        const int first = panel * b, from = first + b, below = n - from, made = std::min(b, below - 1);
        const T* v = kept.reflectors.data() + static_cast<std::size_t>(first) * count + from;
        const T* t = kept.factors.data() + static_cast<std::size_t>(panel) * width * width;
        blas.gemm(Op::C, Op::N, made, columns, below, one, v, n, z + from, n, zero, w.data(), made);
        blas.gemm(Op::N, Op::N, made, columns, made, one, t, made, w.data(), made, zero, tw.data(), made);
        blas.gemm(Op::N, Op::N, below, columns, made, minus, v, n, tw.data(), made, one, z + from, n);
    }
}
