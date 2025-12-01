#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "type.h"
#include <vector>
#include <cstring>
#include <omp.h>

namespace nb = nanobind;

/**
 * High-performance transform and filter operation.
 * 
 * Optimizations:
 * - Cache plane coefficients in local memory
 * - Vectorization-friendly memory access
 * - Reduced view overhead in hot loops
 * - Better memory allocation strategy
 */
auto transform_and_filter(
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD rotation_matrix,
    const ROneArrayD center,
    const ROneArrayD target_center,
    const RTwoArrayD coeffs
) {
    const size_t n_atoms = x_py.shape(0);
    const size_t n_faces = coeffs.shape(0);
    
    // Get views
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    auto R_view = rotation_matrix.view();
    auto center_view = center.view();
    auto target_view = target_center.view();
    auto coeffs_view = coeffs.view();
    
    // Cache scalar values
    const double cx = center_view(0);
    const double cy = center_view(1);
    const double cz = center_view(2);
    
    const double tx = target_view(0);
    const double ty = target_view(1);
    const double tz = target_view(2);
    
    // Cache rotation matrix transpose
    const double RT00 = R_view(0, 0), RT01 = R_view(1, 0), RT02 = R_view(2, 0);
    const double RT10 = R_view(0, 1), RT11 = R_view(1, 1), RT12 = R_view(2, 1);
    const double RT20 = R_view(0, 2), RT21 = R_view(1, 2), RT22 = R_view(2, 2);
    
    // Cache plane coefficients into contiguous memory for better cache locality
    std::vector<double> planes(n_faces * 4);
    for (size_t j = 0; j < n_faces; ++j) {
        planes[j * 4 + 0] = coeffs_view(j, 0);
        planes[j * 4 + 1] = coeffs_view(j, 1);
        planes[j * 4 + 2] = coeffs_view(j, 2);
        planes[j * 4 + 3] = coeffs_view(j, 3);
    }
    
    // First pass: filter and count
    std::vector<bool> inside_mask(n_atoms);
    size_t count = 0;
    
    #pragma omp parallel reduction(+:count)
    {
        // Thread-local copy of planes for even better cache performance
        const double* planes_local = planes.data();
        
        #pragma omp for schedule(static, 4096) nowait
        for (size_t i = 0; i < n_atoms; ++i) {
            // Read and center position
            const double dx = x(i) - cx;
            const double dy = y(i) - cy;
            const double dz = z(i) - cz;
            
            // Transform: (pos - center) @ R.T + target
            const double px_new = dx * RT00 + dy * RT10 + dz * RT20 + tx;
            const double py_new = dx * RT01 + dy * RT11 + dz * RT21 + ty;
            const double pz_new = dx * RT02 + dy * RT12 + dz * RT22 + tz;
            
            // Test against all planes with early exit
            bool inside = true;
            for (size_t j = 0; j < n_faces; ++j) {
                const double* plane = &planes_local[j * 4];
                const double val = px_new * plane[0] + 
                                 py_new * plane[1] + 
                                 pz_new * plane[2] + 
                                 plane[3];
                
                if (val >= 0.0) {
                    inside = false;
                    break;
                }
            }
            
            inside_mask[i] = inside;
            if (inside) ++count;
        }
    }
    
    // Allocate output array
    double* result_pos = new double[count * 3];
    nb::capsule pos_owner(result_pos, [](void *p) noexcept {
        delete[] static_cast<double*>(p);
    });
    
    // Second pass: copy filtered atoms
    // This is sequential but very fast (memory-bound, not compute-bound)
    size_t out_idx = 0;
    for (size_t i = 0; i < n_atoms; ++i) {
        if (inside_mask[i]) {
            const double dx = x(i) - cx;
            const double dy = y(i) - cy;
            const double dz = z(i) - cz;
            
            double* out = &result_pos[out_idx * 3];
            out[0] = dx * RT00 + dy * RT10 + dz * RT20 + tx;
            out[1] = dx * RT01 + dy * RT11 + dz * RT21 + ty;
            out[2] = dx * RT02 + dy * RT12 + dz * RT22 + tz;
            ++out_idx;
        }
    }
    
    return nb::ndarray<nb::numpy, double>(result_pos, {count, 3}, pos_owner);
}

NB_MODULE(_polycrystal, m) {
    m.doc() = "High-performance C++ accelerator for polycrystal generation";
    
    m.def("transform_and_filter", &transform_and_filter,
          nb::arg("x"), nb::arg("y"), nb::arg("z"),
          nb::arg("rotation_matrix"), nb::arg("center"), 
          nb::arg("target_center"), nb::arg("coeffs"),
          "Transform atom positions and filter by Voronoi cell boundaries.\n\n"
          "Performs: pos_new = (pos - center) @ R.T + target, then filters\n"
          "atoms inside the polyhedron defined by plane equations.");
}