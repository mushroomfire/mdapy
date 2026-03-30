#pragma once
/**
 * tachyon_optix_render.h  -  mdapy GPU renderer public interface
 *
 * IMPORTANT DESIGN CONSTRAINT
 * ---------------------------
 * This header is included by tachyon_render.cpp which is compiled by g++.
 * It must NEVER include TachyonOptiX.h, TachyonOptiXShaders.h, or any
 * CUDA / OptiX header.  Those headers contain:
 *   - CUDA device-only intrinsics (__fsqrt_rn, __saturatef, etc.) that g++
 *     does not know about.
 *   - Names (rt_texture, rt_directional_light, RT_FOG_NONE ...) that clash
 *     with macros / declarations in tachyon.h.
 *   - Class TachyonOptiX whose full definition requires OptiX headers.
 *
 * Solution: PIMPL (pointer-to-implementation).
 *   - This header exposes only TachyonOptiXRenderer with an opaque Impl ptr.
 *   - The actual implementation lives in tachyon_optix_impl.cu compiled by
 *     nvcc only.
 *
 * Compatible OptiX version: 7.4, 7.5, or 7.6  (NOT 7.7 or 8.x).
 * Use:  git clone --branch v7.6.0 --depth 1
 *             https://github.com/NVIDIA/optix-dev extern/optix
 */

#include <cstdint>
#include <vector>
#include <memory>

// tachyon_render.h must be included before this header.
// It provides: Vec3, CameraParams, RenderParams, ParticleData, BoxEdgeData.

namespace mdapy_tachyon {

/**
 * GPU ray-tracing renderer backed by TachyonOptiX (NVIDIA OptiX 7.4-7.6).
 * All CUDA and OptiX details are hidden behind the PIMPL pointer.
 */
class TachyonOptiXRenderer {
public:
    TachyonOptiXRenderer();   ///< throws if no CUDA GPU / OptiX init fails
    ~TachyonOptiXRenderer();

    TachyonOptiXRenderer(const TachyonOptiXRenderer &)            = delete;
    TachyonOptiXRenderer &operator=(const TachyonOptiXRenderer &) = delete;

    /**
     * Render one frame.
     * @return RGBA uint8 buffer, H*W*4 bytes, rows top-to-bottom.
     */
    std::vector<uint8_t> render(const RenderParams &rp,
                                const CameraParams &cp,
                                const ParticleData &pd,
                                const BoxEdgeData  *box = nullptr);

private:
    struct Impl;                    // defined in tachyon_optix_impl.cu only
    std::unique_ptr<Impl> impl_;
};

} // namespace mdapy_tachyon