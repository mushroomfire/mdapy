/*
 * TachyonOptiXShaders.cu - OptiX PTX shading and ray intersection routines 
 *
 * (C) Copyright 2013-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: TachyonOptiXShaders.cu,v 1.112 2022/04/20 03:30:38 johns Exp $
 *
 */

/**
 *  \file TachyonOptiXShaders.cu
 *  \brief Tachyon ray tracing engine core routines compiled to PTX for
 *         runtime JIT to build complete ray tracing pipelines.
 *         Written for NVIDIA OptiX 7 and later.
 */

//
// This is a second generation of the Tachyon implementation for OptiX.
// The new implementation favors the strengths of OptiX 7, and uses
// OptiX ray payload registers, direct CUDA interoperability and advanced
// CUDA features for both performance and maintainability.
//
// This software and its line of antecedants are described in:
//   "Multiscale modeling and cinematic visualization of photosynthetic 
//    energy conversion processes from electronic to cell scales"
//    M. Sener, S. Levy, J. E. Stone, AJ Christensen, B. Isralewitz, 
//    R. Patterson, K. Borkiewicz, J. Carpenter, C. N. Hunter, 
//    Z. Luthey-Schulten, D. Cox.
//    J. Parallel Computing, 102, pp. 102698, 2021.
//    https://doi.org/10.1016/j.parco.2020.102698
//
//   "Omnidirectional Stereoscopic Projections for VR"  
//    J. E. Stone.  In, William R. Sherman, editor,
//    VR Developer Gems, Taylor and Francis / CRC Press, Chapter 24, 2019.
//    https://www.taylorfrancis.com/chapters/edit/10.1201/b21598-24/omnidirectional-stereoscopic-projections-vr-john-stone
//
//   "Interactive Ray Tracing Techniques for
//    High-Fidelity Scientific Visualization"  
//    J. E. Stone. In, Eric Haines and Tomas Akenine-M�ller, editors,
//    Ray Tracing Gems, Apress, Chapter 27, pp. 493-515, 2019.
//    https://link.springer.com/book/10.1007/978-1-4842-4427-2
//
//   "A Planetarium Dome Master Camera"  
//    J. E. Stone.  In, Eric Haines and Tomas Akenine-M�ller, editors,
//    Ray Tracing Gems, Apress, Chapter 4, pp. 49-60, 2019.
//    https://link.springer.com/book/10.1007/978-1-4842-4427-2
//
//   "Immersive Molecular Visualization with Omnidirectional
//    Stereoscopic Ray Tracing and Remote Rendering"
//    J. E. Stone, W. R. Sherman, and K. Schulten.
//    High Performance Data Analysis and Visualization Workshop,
//    2016 IEEE International Parallel and Distributed Processing
//    Symposium Workshops (IPDPSW), pp. 1048-1057, 2016.
//    http://dx.doi.org/10.1109/IPDPSW.2016.121
//
//   "Atomic Detail Visualization of Photosynthetic Membranes with
//    GPU-Accelerated Ray Tracing"  
//    J. E. Stone, M. Sener, K. L. Vandivort, A. Barragan, A. Singharoy, 
//    I. Teo, J. V. Ribeiro, B. Isralewitz, B. Liu, B.-C. Goh, J. C. Phillips, 
//    C. MacGregor-Chatwin, M. P. Johnson, L. F. Kourkoutis, C. N. Hunter, 
//    K. Schulten
//    J. Parallel Computing, 55:17-27, 2016.
//    http://dx.doi.org/10.1016/j.parco.2015.10.015
//
//   "GPU-Accelerated Molecular Visualization on
//    Petascale Supercomputing Platforms"
//    J. E. Stone, K. L. Vandivort, and K. Schulten.
//    UltraVis'13: Proceedings of the 8th International Workshop on
//    Ultrascale Visualization, pp. 6:1-6:8, 2013.
//    http://dx.doi.org/10.1145/2535571.2535595
//
//    "An Efficient Library for Parallel Ray Tracing and Animation"
//    John E. Stone.  Master's Thesis, University of Missouri-Rolla,
//    Department of Computer Science, April 1998
//    https://scholarsmine.mst.edu/masters_theses/1747
//
//    "Rendering of Numerical Flow Simulations Using MPI"
//    J. Stone and M. Underwood.
//    Second MPI Developers Conference, pages 138-141, 1996.
//    http://dx.doi.org/10.1109/MPIDC.1996.534105
//


#include <optix.h>
//#include <optix_device.h>
#include <stdint.h>

#define TACHYON_INTERNAL 1
#include "TachyonOptiXShaders.h"

// Macros related to ray origin epsilon stepping to prevent
// self-intersections with the surface we're leaving
// This is a cheesy way of avoiding self-intersection
// but it ameliorates the problem.
// Since changing the scene epsilon even to large values does not
// always cure the problem, this workaround is still required.
#define TACHYON_USE_RAY_STEP       1
#define TACHYON_TRANS_USE_INCIDENT 1
#define TACHYON_RAY_STEP           N*rtLaunch.scene.epsilon*4.0f
#define TACHYON_RAY_STEP2          ray_direction*rtLaunch.scene.epsilon*4.0f

// reverse traversal of any-hit rays for shadows/AO
#define REVERSE_RAY_STEP       (scene_epsilon*10.0f)
#define REVERSE_RAY_LENGTH     3.0f

// Macros to enable particular ray-geometry intersection variants that
// optimize for speed, or some combination of speed and accuracy
#define TACHYON_USE_SPHERES_HEARNBAKER 1


/// Helper function to return 1-D framebuffer offset computed 
/// from the current thread's launch_index.
static __device__ __inline__ int tachyon1DLaunchIndex(void) {
  const uint3 launch_index = optixGetLaunchIndex();
  const uint3 launch_dim = optixGetLaunchDimensions();
  const int idx = launch_index.y*launch_dim.x + launch_index.x;
  return idx;
}

/// Helper function to return 1-D framebuffer offset computed 
/// from the current thread's launch_index.
static __device__ __inline__ int tachyon1DLaunchIndex(uint3 dim, uint3 index) {
  return index.y*dim.x + index.x;
}



//
// OptiX ray processing programs
//

/// launch parameters in constant memory, filled  by optixLaunch)
__constant__ tachyonLaunchParams rtLaunch;

static __forceinline__ __device__
void *unpackPointer( uint32_t i0, uint32_t i1 ) {
  const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
  void*           ptr = reinterpret_cast<void*>( uptr );
  return ptr;
}

static __forceinline__ __device__
void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 ) {
  const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD() {
  const uint32_t p0 = optixGetPayload_0();
  const uint32_t p1 = optixGetPayload_1();
  return reinterpret_cast<T*>( unpackPointer( p0, p1 ) );
}



//
// Per-ray data "PRD"
//

// radiance PRD data is used by closest-hit and miss programs 
struct PerRayData_radiance {
  float3 result;     // final shaded surface color
  float alpha;       // alpha value to back-propagate to framebuffer
  float importance;  // importance of recursive ray tree
  int depth;         // current recursion depth
  int transcnt;      // transmission ray surface count/depth
};

/// radiance PRD aasample count is stored in ray payload register 2
static __forceinline__ __device__ uint32_t getPayloadAAsample() {
  return optixGetPayload_2();
}


#if 0
// XXX we currently use ray payload registers for shadow PRD
//     but this is maintained for the inevitable future revisions
//     that bring more sophisticated shadow filtering
struct PerRayData_shadow {
  // attenuation is set to 0.0f at 100% shadow, and 1.0f no occlusion 
  float attenuation; // grayscale light filtered by transmissive surfaces
};
#endif

/// any-hit programs read-modify-update shadow attenuation value
/// carried in ray payload register 0
static __forceinline__ __device__ float getPayloadShadowAttenuation() {
  return __int_as_float(optixGetPayload_0());
}

/// any-hit programs read-modify-update shadow attenuation value
/// carried in ray payload register 0
static __forceinline__ __device__ void setPayloadShadowAttenuation(const float attenuation) {
  optixSetPayload_0(__float_as_int(attenuation));
}


static int __forceinline__ __device__ subframe_count() {
//  return (accumCount + progressiveSubframeIndex);
  return rtLaunch.frame.subframe_index; 
}



//
// Device functions for clipping rays by geometric primitives
//

// fade_start: onset of fading
//   fade_end: fully transparent, begin clipping of geometry
__device__ void sphere_fade_and_clip(const float3 &hit_point,
                                     const float3 &cam_pos,
                                     float fade_start, float fade_end,
                                     float &alpha) {
  float camdist = length(hit_point - cam_pos);

  // we can omit the distance test since alpha modulation value is clamped
  // if (1 || camdist < fade_start) {
    float fade_len = fade_start - fade_end;
    alpha *= __saturatef((camdist - fade_start) / fade_len);
  // }
}


__device__ void ray_sphere_clip_interval(float3 ray_origin,  
                                         float3 ray_direction, float3 center,
                                         float rad, float2 &tinterval) {
  float3 V = center - ray_origin;
  float b = dot(V, ray_direction);
  float disc = b*b + rad*rad - dot(V, V);

  // if the discriminant is positive, the ray hits...
  if (disc > 0.0f) {
    disc = sqrtf(disc);
    tinterval.x = b-disc;
    tinterval.y = b+disc;
  } else {
    tinterval.x = -RT_DEFAULT_MAX;
    tinterval.y =  RT_DEFAULT_MAX;
  }
}


__device__ void clip_ray_by_plane(float3 ray_origin,
                                  float3 ray_direction, 
                                  float &tmin, float &tmax,
                                  const float4 plane) {
  float3 n = make_float3(plane);
  float dt = dot(ray_direction, n);
  float t = (-plane.w - dot(n, ray_origin))/dt;
  if(t > tmin && t < tmax) {
    if (dt <= 0) {
      tmax = t;
    } else {
      tmin = t;
    }
  } else {
    // ray interval lies completely on one side of the plane.  Test one point.
    float3 p = ray_origin + tmin * ray_direction;
    if (dot(make_float4(p.x, p.y, p.z, 1.0f), plane) < 0) {
      tmin = tmax = RT_DEFAULT_MAX; // cull geometry
    }
  }
}



//
// Default Tachyon exception handling program
//   Any OptiX state on the stack will be gone post-exception, so if we 
//   want to store anything it would need to be written to a global 
//   memory allocation.
//   When printing exception info, all output should be emitted with a single
//   printf() call or equivalent to ensure correct output ordering.
//
extern "C" __global__ void __exception__all() {
  const int code = optixGetExceptionCode();
  const uint3 launch_index = optixGetLaunchIndex();

  switch (code) {
    case OPTIX_EXCEPTION_CODE_STACK_OVERFLOW:
      printf("TachyonOptiX) Stack overflow, launch idx (%u,%u)\n",
            launch_index.x, launch_index.y);
      break;

    case OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED:
      printf("TachyonOptiX) Max trace depth exceeded, launch idx (%u,%u)\n",
            launch_index.x, launch_index.y);
      break;

    case OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED:
      printf("TachyonOptiX) Max traversal depth exceeded, launch idx (%u,%u)\n",
            launch_index.x, launch_index.y);
      break;

    case OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_MISS_SBT:
      printf("TachyonOptiX) Invalid miss SBT record idx, launch idx (%u,%u)\n",
             launch_index.x, launch_index.y);
      // optixGetExceptionInvalidSbtOffset()
      break;

    case OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT:
      printf("TachyonOptiX) Invalid hit SBT record idx, launch idx (%u,%u)\n",
             launch_index.x, launch_index.y);
      // optixGetExceptionInvalidSbtOffset()
      break;

#if OPTIX_VERSION >= 70100
    case OPTIX_EXCEPTION_CODE_BUILTIN_IS_MISMATCH:
      printf("TachyonOptiX) Built-in IS mismatch, launch idx (%u,%u)\n",
             launch_index.x, launch_index.y);
      break;

    case OPTIX_EXCEPTION_CODE_INVALID_RAY:
      printf("TachyonOptiX) Trace call contains Inf/NaN, launch idx (%u,%u):\n"
             "TachyonOptiX)  @ %s\n", 
             launch_index.x, launch_index.y, optixGetExceptionLineInfo());
      // optixGetExceptionInvalidRay()
      break;

    case OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH:
      printf("TachyonOptiX) Callable param mismatch, launch idx (%d,%d)\n",
             launch_index.x, launch_index.y);
      // optixGetExceptionParameterMismatch()
      break;
#endif

    case OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_TRAVERSABLE:
    default:
      printf("TachyonOptiX) Caught exception 0x%X (%d) at launch idx (%u,%u)\n",
             code, code, launch_index.x, launch_index.y );
      break;
  }

  // and write to frame buffer ...
  const int idx = launch_index.x + launch_index.y*rtLaunch.frame.size.x;
  rtLaunch.frame.framebuffer[idx] = make_color_rgb4u(make_float3(0.f, 0.f, 0.f));
}


//
// Shadow ray programs
// 
// The shadow PRD attenuation factor represents what fraction of the light
// is visible.  An attenuation factor of 0 indicates full shadow, no light
// makes it to the surface.  An attenuation factor of 1.0 indicates a 
// complete lack of shadow.
//

extern "C" __global__ void __closesthit__shadow_nop() {
  // no-op
}


//
// Shadow miss program for any kind of geometry
//   Regardless what type of geometry we're rendering, if we end up running
//   the miss program, then we know we didn't hit an occlusion, so the 
//   light attenuation factor should be 1.0.
extern "C" __global__ void __miss__shadow_nop() {
  // For scenes with either opaque or transmissive objects, 
  // a "miss" always indicates that there was no (further) occlusion, 
  // and thus no shadow.

  // no-op
}


// Shadow AH program for purely opaque geometry
//   If we encounter an opaque object during an anyhit traversal,
//   it sets the light attentuation factor to 0 (full shadow) and
//   immediately terminates the shadow traversal.
extern "C" __global__ void __anyhit__shadow_opaque() {
  // this material is opaque, so it fully attenuates all shadow rays
  setPayloadShadowAttenuation(0.0f);

  // full shadow should cause us to early-terminate AH search
  optixTerminateRay();
}


// Shadow programs for scenes containing a mix of both opaque and transparent
//   objects.  In the case that a scene contains a mix of both fully opaque
//   and transparent objects, we have two different types of AH programs
//   for the two cases.  Prior to launching the shadow rays, the PRD attenuation
//   factor is set to 1.0 to indicate no shadowing, and it is subsequently
//   modified by the AH programs associated with the different objects in 
//   the scene.
//
//   To facilitate best performance in scenes that contain a mix of 
//   fully opaque and transparent geometry, we could run an anyhit against
//   opaque geometry first, and only if we had a miss, we could continue with
//   running anyhit traversal on just the transmissive geometry.
//

// Any hit program required for shadow filtering through transparent materials
extern "C" __global__ void __anyhit__shadow_transmission() {
#if defined(TACHYON_USE_GEOMFLAGS)
  const GeomSBTHG &sbtHG = *reinterpret_cast<const GeomSBTHG*>(optixGetSbtDataPointer());

  int geomflags = sbtHG.geomflags;
  if (!(geomflags & (RT_MAT_ALPHA | RT_MAT_TEXALPHA))) {
    // we hit something fully opaque
    setPayloadShadowAttenuation(0.0f); // 100% full shadow
    optixTerminateRay();
  } else {
    // use a VERY simple shadow filtering scheme based on opacity
    float opacity = rtLaunch.materials[sbtHG.materialindex].opacity;

#if 1
    // incorporate alpha cutout textures into any-hit if necessary
    if (geomflags & RT_MAT_TEXALPHA) {
      auto & tmesh = sbtHG.trimesh;
      if (tmesh.tex2d != nullptr) {
        const int primID = optixGetPrimitiveIndex();

        int3 index;
        if (tmesh.indices == NULL) {
          int idx3 = primID*3;
          index = make_int3(idx3, idx3+1, idx3+2);
        } else {
          index = tmesh.indices[primID];
        }

        const float2 barycentrics = optixGetTriangleBarycentrics();
 
        float2 txc0 = tmesh.tex2d[index.x];
        float2 txc1 = tmesh.tex2d[index.y];
        float2 txc2 = tmesh.tex2d[index.z];

        // interpolate tex coord from triangle barycentrics
        float2 texcoord = (txc0 * (1.0f - barycentrics.x - barycentrics.y) +
                           txc1 * barycentrics.x + txc2 * barycentrics.y);

        // XXX need to implement ray differentials for tex filtering
        int matidx = sbtHG.materialindex;
        const auto &mat = rtLaunch.materials[matidx];
        float4 tx = tex2D<float4>(mat.tex, texcoord.x, texcoord.y);

        opacity *= tx.w;
      }
    }
#endif

    // this material could be translucent, so it may attenuate shadow rays
    float attenuation = getPayloadShadowAttenuation(); 
    attenuation *= (1.0f - opacity);
    setPayloadShadowAttenuation(attenuation); 
    // ceck to see if we've hit 100% shadow or not
    if (attenuation < 0.001f) {
      optixTerminateRay();
    } else {
#if defined(TACHYON_RAYSTATS)
      const int idx = tachyon1DLaunchIndex();
      rtLaunch.frame.raystats2_buffer[idx].y++; // increment trans ray skip count
#endif
      optixIgnoreIntersection();
    }
  }

#else

  // use a VERY simple shadow filtering scheme based on opacity
  const GeomSBTHG &sbtHG = *reinterpret_cast<const GeomSBTHG*>(optixGetSbtDataPointer());
  float opacity = rtLaunch.materials[sbtHG.materialindex].opacity;

#if 0
  const uint3 launch_index = optixGetLaunchIndex();
  if (launch_index.x == 994) {
    printf("AH xy:%d %d mat[%d] diffuse: %g  opacity: %g  atten: %g\n", 
           launch_index.x, launch_index.y, 
           sbtHG.materialindex, mat.diffuse, mat.opacity,
           prd.attenuation);
  }
#endif

  // this material could be translucent, so it may attenuate shadow rays
  float attenuation = getPayloadShadowAttenuation(); 
  attenuation *= (1.0f - opacity);
  setPayloadShadowAttenuation(attenuation); 
  // check to see if we've hit 100% shadow or not
  if (attenuation < 0.001f) {
    optixTerminateRay();
  } else {
#if defined(TACHYON_RAYSTATS)
    const int idx = tachyon1DLaunchIndex();
    rtLaunch.frame.raystats2_buffer[idx].y++; // increment trans ray skip count
#endif
    optixIgnoreIntersection();
  }
#endif
}



// Any hit program required for shadow filtering when an
// HMD/camera fade-and-clip is active, through both
// solid and transparent materials
extern "C" __global__ void any_hit_shadow_clip_sphere() {
  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();
  const GeomSBTHG &sbtHG = *reinterpret_cast<const GeomSBTHG*>(optixGetSbtDataPointer());
  float opacity = rtLaunch.materials[sbtHG.materialindex].opacity;

  // compute world space hit point for use in evaluating fade/clip effect
  float3 hit_point = ray_origin + t_hit * ray_direction;

  // compute additional attenuation from clipping sphere if enabled
  float clipalpha = 1.0f;
  if (rtLaunch.clipview_mode == 2) {
    sphere_fade_and_clip(hit_point, rtLaunch.cam.pos, rtLaunch.clipview_start, 
                         rtLaunch.clipview_end, clipalpha);
  }


  // use a VERY simple shadow filtering scheme based on opacity
  // this material could be translucent, so it may attenuate shadow rays
  float attenuation = getPayloadShadowAttenuation();
  attenuation *= (1.0f - (clipalpha * opacity));
  setPayloadShadowAttenuation(attenuation);
  // check to see if we've hit 100% shadow or not
  if (attenuation < 0.001f) {
    optixTerminateRay();
  } else {
#if defined(TACHYON_RAYSTATS)
    const int idx = tachyon1DLaunchIndex();
    rtLaunch.frame.raystats2_buffer[idx].y++; // increment trans ray skip count
#endif
    optixIgnoreIntersection();
  }
}


// 
// OptiX anyhit program for radiance rays, a no-op
// 

extern "C" __global__ void __anyhit__radiance_nop() {
  // no-op
}


//
// OptiX miss programs for drawing the background color or
// background color gradient when no objects are hit
//

// Miss program for solid background
extern "C" __global__ void __miss__radiance_solid_bg() {
  // Fog overrides the background color if we're using
  // Tachyon radial fog, but not for OpenGL style fog.
  PerRayData_radiance &prd = *getPRD<PerRayData_radiance>();
  prd.result = rtLaunch.scene.bg_color;
  prd.alpha = 0.0f; // alpha of background is 0.0f;

#if defined(TACHYON_RAYSTATS)
  const int idx = tachyon1DLaunchIndex();
  rtLaunch.frame.raystats1_buffer[idx].w++; // increment miss counter
#endif
}


// Miss program for gradient background with perspective projection.
// Fog overrides the background color if we're using
// Tachyon radial fog, but not for OpenGL style fog.
extern "C" __global__ void __miss__radiance_gradient_bg_sky_sphere() {
  PerRayData_radiance &prd = *getPRD<PerRayData_radiance>();

  // project ray onto the gradient "up" direction, and compute the 
  // scalar color interpolation parameter
  float IdotG = dot(optixGetWorldRayDirection(), rtLaunch.scene.bg_grad_updir);
  float val = (IdotG - rtLaunch.scene.bg_grad_botval) * 
              rtLaunch.scene.bg_grad_invrange;

  // Compute and add random noise to the background gradient to 
  // avoid banding artifacts, particularly in compressed video.
  // Noise RNG depends only on pixel index, with no sample/subframe
  // contribution, so that dither pattern won't average out.
  const int idx = tachyon1DLaunchIndex();
#if 1
  float u = squares_rng<2>(idx, SQUARES_RNG_KEY1) * UINT32_RAND_MAX_INV;
#else
  float u = tea<4>(idx, idx) * UINT32_RAND_MAX_INV;
#endif
  float noise = rtLaunch.scene.bg_grad_noisemag * (u - 0.5f);
  val += noise; // add the noise to the interpolation parameter

  val = __saturatef(val); // clamp the interpolation param to [0:1]
  float3 col = val * rtLaunch.scene.bg_color_grad_top +
               (1.0f - val) * rtLaunch.scene.bg_color_grad_bot;
  prd.result = col;
  prd.alpha = 0.0f; // alpha of background is 0.0f;

#if defined(TACHYON_RAYSTATS)
  rtLaunch.frame.raystats1_buffer[idx].w++; // increment miss counter
#endif
}


// Miss program for gradient background with orthographic projection.
// Fog overrides the background color if we're using
// Tachyon radial fog, but not for OpenGL style fog.
extern "C" __global__ void __miss__radiance_gradient_bg_sky_plane() {
  PerRayData_radiance &prd = *getPRD<PerRayData_radiance>();

  // project ray onto the gradient "up" direction, and compute the 
  // scalar color interpolation parameter
  float IdotG = dot(optixGetWorldRayDirection(), rtLaunch.scene.bg_grad_updir);
  float val = (IdotG - rtLaunch.scene.bg_grad_botval) * 
              rtLaunch.scene.bg_grad_invrange;

  // Compute and add random noise to the background gradient to 
  // avoid banding artifacts, particularly in compressed video.
  // Noise RNG depends only on pixel index, with no sample/subframe
  // contribution, so that dither pattern won't average out.
  const int idx = tachyon1DLaunchIndex();
#if 1
  float u = squares_rng<2>(idx, SQUARES_RNG_KEY1) * UINT32_RAND_MAX_INV;
#else
  float u = tea<4>(idx, idx) * UINT32_RAND_MAX_INV;
#endif
  float noise = rtLaunch.scene.bg_grad_noisemag * (u - 0.5f);
  val += noise; // add the noise to the interpolation parameter

  val = __saturatef(val); // clamp the interpolation param to [0:1]
  float3 col = val * rtLaunch.scene.bg_color_grad_top +
               (1.0f - val) * rtLaunch.scene.bg_color_grad_bot;
  prd.result = col;
  prd.alpha = 0.0f; // alpha of background is 0.0f;

#if defined(TACHYON_RAYSTATS)
  rtLaunch.frame.raystats1_buffer[idx].w++; // increment miss counter
#endif
}



//
// Ray gen accumulation buffer helper routines
//
static void __inline__ __device__ accumulate_color(int idx, float4 colrgba4f) {
  // accumulate with existing contents except during a "clear"
  if (!rtLaunch.frame.fb_clearall) {
    float4 rgba = rtLaunch.frame.accum_buffer[idx];
    colrgba4f += rgba;
  }

  // always update the accumulation buffer
  rtLaunch.frame.accum_buffer[idx] = colrgba4f;

  // update the color buffer only when we're told to
  if (rtLaunch.frame.update_colorbuffer) {
    colrgba4f *= rtLaunch.frame.accum_normalize;

#if defined(TACHYON_OPTIXDENOISER)
    // When running on LDR inputs, AI denoiser consumes images in sRGB 
    // colorspace or at least using a gamma of 2.2, with floating point 
    // values range-clamped to [0,1].

    if (rtLaunch.frame.denoiser_enabled) {
      // pre-scale RGBA inputs to avoid excessive clamping before denoising
      colrgba4f *= 0.80f; // alpha is modified, but we revert it later

      // RGBA value range clamping can't be inverted, so LDR denoising 
      // has some flexibility downsides for subsequent steps.
      colrgba4f = clamp_float4(colrgba4f); // clamp values [0,1]

      // use a cheap gamma 2.0 approximation which is trivially inverted
      // to please the LDR denoiser input stage
      float4 sRGB_approx20 = linear_to_sRGB_approx_20(colrgba4f);
      rtLaunch.frame.denoiser_colorbuffer[idx] = sRGB_approx20;

      // When denoising, we early-exit here.  The remaining steps in 
      // the image pipeline are done within a separate CUDA kernel,
      // launched only after denoising has completed
      return;
    }
#endif

    //
    // The remaining steps here are only done when denoising is off,
    //

    // HDR tone mapping operators need to be applied after denoising
    // has been completed.  If we use tone mapping on an LDR input,
    // we may have to revert from sRGB to linear before applying the
    // TMO, and then convert back to sRGB.
    // Also performs color space conversion if required
    float4 tonedcol;
    tonedcol = tonemap_color(colrgba4f, 
                             rtLaunch.frame.tonemap_mode,
                             rtLaunch.frame.tonemap_exposure,
                             rtLaunch.frame.colorspace);

    if (rtLaunch.frame.colorspace == RT_COLORSPACE_sRGB)
      colrgba4f = linear_to_sRGB(tonedcol);
    else
      colrgba4f = tonedcol;

    // clamping is applied during conversion to uchar4
    rtLaunch.frame.framebuffer[idx] = make_color_rgb4u(colrgba4f);
  }
}


#if defined(TACHYON_RAYSTATS)
static void __inline__ __device__ raystats_clear_incr(unsigned int idx) {
  if (rtLaunch.frame.fb_clearall) {
    // assign ray stats immediately to cut register use later
    uint4 s=make_uint4(rtLaunch.aa_samples, 0, 0, 0); // set primary ray counter
    rtLaunch.frame.raystats1_buffer[idx]=s;
  } else {
    // increment ray stats immediately to cut register use later
    rtLaunch.frame.raystats1_buffer[idx].x+=rtLaunch.aa_samples; // increment primary ray counter
  }
}
#endif


//
// OptiX programs that implement the camera models and ray generation code
//


//
// CUDA device function for computing the new ray origin
// and ray direction, given the radius of the circle of confusion disc,
// and an orthonormal basis for each ray.
//
#if 1

static __device__ __inline__
void dof_ray(const float cam_dof_focal_dist, const float cam_dof_aperture_rad,
             const float3 &ray_origin_orig, float3 &ray_origin,
             const float3 &ray_direction_orig, float3 &ray_direction,
             unsigned int &randseed, const float3 &up, const float3 &right) {
  float3 focuspoint = ray_origin_orig + ray_direction_orig * cam_dof_focal_dist;
  float2 dofjxy;
  jitter_disc2f(randseed, dofjxy, cam_dof_aperture_rad);
  ray_origin = ray_origin_orig + dofjxy.x*right + dofjxy.y*up;
  ray_direction = normalize(focuspoint - ray_origin);
}

#else

// use low-discrepancy sequences for sampling the circle of confusion disc
static __device__ __inline__
void dof_ray(const float cam_dof_focal_dist, const float cam_dof_aperture_rad,
             const float3 &ray_origin_orig, float3 &ray_origin,
             const float3 &ray_direction_orig, float3 &ray_direction,
             float2 &qrnxy, const float3 &up, const float3 &right) {
  float3 focuspoint = ray_origin_orig + ray_direction_orig * cam_dof_focal_dist;
  float2 dofjxy;
  jitter_disc2f_qrn(qrnxy, dofjxy, cam_dof_aperture_rad);
  ray_origin = ray_origin_orig + dofjxy.x*right + dofjxy.y*up;
  ray_direction = normalize(focuspoint - ray_origin);
}

#endif


//
// Templated perspective camera ray generation code
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_perspective_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const int idx = tachyon1DLaunchIndex(launch_dim, launch_index);
#if defined(TACHYON_RAYSTATS)
  // clear/increment ray stats immediately to cut register use later
  raystats_clear_incr(idx);
#endif

  const auto &cam = rtLaunch.cam;

  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high
  // framebuffer, and the right eye into the lower half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // with simple pointer offset arithmetic.
  float3 eyepos;
  uint viewport_sz_y, viewport_idx_y;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // right image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyepos = cam.pos + cam.U * cam.stereo_eyesep * 0.5f;
    } else {
      // left image
      viewport_idx_y = launch_index.y;
      eyepos = cam.pos - cam.U * cam.stereo_eyesep * 0.5f;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyepos = cam.pos;
  }


  //
  // general primary ray calculations
  //
  float2 aspect = make_float2(float(launch_dim.x) / float(viewport_sz_y), 1.0f) * cam.zoom;
  float2 viewportscale = 1.0f / make_float2(launch_dim.x, viewport_sz_y);
  float2 d = make_float2(launch_index.x, viewport_idx_y) * viewportscale * aspect * 2.f - aspect; // center of pixel in image plane

  unsigned int randseed = tea<4>(idx, subframe_count());

  float3 col = make_float3(0.0f);
  float alpha = 0.0f;
  float3 ray_origin = eyepos;
  for (uint32_t s=0; s<rtLaunch.aa_samples; s++) {
    float2 jxy;
    jitter_offset2f(randseed, jxy);

    jxy = jxy * viewportscale * aspect * 2.f + d;
    float3 ray_direction = normalize(jxy.x*cam.U + jxy.y*cam.V + cam.W);

    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
              eyepos, ray_origin, ray_direction, ray_direction,
              randseed, cam.V, cam.U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.result = make_float3(0.0f);
    prd.alpha = 1.f;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = rtLaunch.max_trans;

    uint32_t p0, p1; // pack PRD pointer into p0,p1 payload regs
    packPointer(&prd, p0, p1);

    optixTrace(rtLaunch.traversable,
               ray_origin,
               ray_direction,
               0.0f,                          // tmin
               RT_DEFAULT_MAX,                // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
               RT_RAY_TYPE_RADIANCE,          // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_RADIANCE,          // missSBTIndex
               p0, p1,                        // PRD ptr in 2x uint32
               s);                            // use aasample in CH/MISS RNGs

    col += prd.result;
    alpha += prd.alpha;
  }

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(idx, make_float4(col, alpha));
#endif
}

extern "C" __global__ void __raygen__camera_perspective() {
  tachyon_camera_perspective_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_perspective_dof() {
  tachyon_camera_perspective_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_perspective_stereo() {
  tachyon_camera_perspective_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_perspective_stereo_dof() {
  tachyon_camera_perspective_general<1, 1>();
}




//
// Templated orthographic camera ray generation code
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_orthographic_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const int idx = tachyon1DLaunchIndex(launch_dim, launch_index);
#if defined(TACHYON_RAYSTATS)
  // clear/increment ray stats immediately to cut register use later
  raystats_clear_incr(idx);
#endif

  const auto &cam = rtLaunch.cam;

  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high
  // framebuffer, and the right eye into the lower half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // with simple pointer offset arithmetic.
  float3 eyepos;
  uint viewport_sz_y, viewport_idx_y;
  float3 view_direction;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // right image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyepos = cam.pos + cam.U * cam.stereo_eyesep * 0.5f;
    } else {
      // left image
      viewport_idx_y = launch_index.y;
      eyepos = cam.pos - cam.U * cam.stereo_eyesep * 0.5f;
    }
    view_direction = normalize(cam.pos-eyepos + normalize(cam.W) * cam.stereo_convergence_dist);
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyepos = cam.pos;
    view_direction = normalize(cam.W);
  }

  //
  // general primary ray calculations
  //
  float2 aspect = make_float2(float(launch_dim.x) / float(viewport_sz_y), 1.0f) * cam.zoom;
  float2 viewportscale = 1.0f / make_float2(launch_dim.x, viewport_sz_y);

  float2 d = make_float2(launch_index.x, viewport_idx_y) * viewportscale * aspect * 2.f - aspect; // center of pixel in image plane

  unsigned int randseed = tea<4>(idx, subframe_count());

  float3 col = make_float3(0.0f);
  float alpha = 0.0f;
  float3 ray_direction = view_direction;
  for (uint32_t s=0; s<rtLaunch.aa_samples; s++) {
    float2 jxy;
    jitter_offset2f(randseed, jxy);
    jxy = jxy * viewportscale * aspect * 2.f + d;
    float3 ray_origin = eyepos + jxy.x*cam.U + jxy.y*cam.V;

    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
              ray_origin, ray_origin, view_direction, ray_direction,
              randseed, cam.V, cam.U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.alpha = 1.f;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = rtLaunch.max_trans;

    uint32_t p0, p1; // pack PRD pointer into p0,p1 payload regs
    packPointer(&prd, p0, p1);

    optixTrace(rtLaunch.traversable,
               ray_origin,
               ray_direction,
               0.0f,                          // tmin
               RT_DEFAULT_MAX,                // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
               RT_RAY_TYPE_RADIANCE,          // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_RADIANCE,          // missSBTIndex
               p0, p1,                        // PRD ptr in 2x uint32
               s);                            // use aasample in CH/MISS RNGs

    col += prd.result;
    alpha += prd.alpha;
  }

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(idx, make_float4(col, alpha));
#endif
}

extern "C" __global__ void __raygen__camera_orthographic() {
  tachyon_camera_orthographic_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_orthographic_dof() {
  tachyon_camera_orthographic_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_orthographic_stereo() {
  tachyon_camera_orthographic_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_orthographic_stereo_dof() {
  tachyon_camera_orthographic_general<1, 1>();
}




//
// 360-degree stereoscopic cube map image format for use with
// Oculus, Google Cardboard, and similar VR headsets
//
// ORBX player format: 
//   all faces are left-right mirror images vs. what viewer sees from inside
//   top is also rotated 90 degrees right, bottom is rotated 90 degrees left
//   Faces are ordered Back, Front, Top, Bottom, Left, Right
//   Stereo layout has left eye images on the left, right eye images 
//   on the right, all within the same row
//   
// Unity cube map format:
//   https://docs.unity3d.com/Manual/class-Cubemap.html
//   https://vrandarchitecture.com/2016/07/19/stereoscopic-renders-in-unity3d-for-gearvr/
//   https://mheavers.medium.com/implementing-a-stereo-skybox-into-unity-for-virtual-reality-e427cf338b06
//   https://forum.unity.com/threads/camera-rendertocubemap-including-orientation.534469/
//   https://www.videopoetics.com/tutorials/capturing-stereoscopic-panoramas-unity/
//
// Unreal Engine cube map format:
//   https://docs.unrealengine.com/4.27/en-US/RenderingAndGraphics/Textures/Cubemaps/CreatingCubemaps/
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_cubemap_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const int idx = tachyon1DLaunchIndex(launch_dim, launch_index);
#if defined(TACHYON_RAYSTATS)
  // clear/increment ray stats immediately to cut register use later
  raystats_clear_incr(idx);
#endif

  const auto &cam = rtLaunch.cam;

  // compute which cubemap face we're drawing by the X index.
  uint facesz = launch_dim.y; // square cube faces, equal to image height
  uint face = (launch_index.x / facesz) % 6;
  uint2 face_idx = make_uint2(launch_index.x % facesz, launch_index.y);

  // For the OTOY ORBX viewer, Oculus VR software, and some of the
  // related apps, the cubemap image is stored with the X axis oriented
  // such that when viewed as a 2-D image, they are all mirror images.
  // The mirrored left-right orientation used here corresponds to what is
  // seen standing outside the cube, whereas the ray tracer shoots
  // rays from the inside, so we flip the X-axis pixel storage order.
  // The top face of the cubemap has both the left-right and top-bottom
  // orientation flipped also.
  // Set per-face orthonormal basis for camera
  float3 face_U, face_V, face_W;
  switch (face) {
    case 0: // back face, left-right mirror
      face_U =  cam.U;
      face_V =  cam.V;
      face_W = -cam.W;
      break;

    case 1: // front face, left-right mirror
      face_U =  -cam.U;
      face_V =  cam.V;
      face_W =  cam.W;
      break;

    case 2: // top face, left-right mirrored, rotated 90 degrees right
      face_U = -cam.W;
      face_V =  cam.U;
      face_W =  cam.V;
      break;

    case 3: // bottom face, left-right mirrored, rotated 90 degrees left
      face_U = -cam.W;
      face_V = -cam.U;
      face_W = -cam.V;
      break;

    case 4: // left face, left-right mirrored
      face_U = -cam.W;
      face_V =  cam.V;
      face_W = -cam.U;
      break;

    case 5: // right face, left-right mirrored
      face_U =  cam.W;
      face_V =  cam.V;
      face_W =  cam.U;
      break;
  }

  // Stereoscopic rendering is provided by rendering in a side-by-side
  // format with the left eye image into the left half of a double-wide
  // framebuffer, and the right eye into the right half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // into an efficient cubemap texture.
  uint viewport_sz_x; // , viewport_idx_x;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-wide framebuffer when stereo is enabled
    viewport_sz_x = launch_dim.x >> 1;
    if (launch_index.x >= viewport_sz_x) {
      // right image
//      viewport_idx_x = launch_index.x - viewport_sz_x;
      eyeshift =  0.5f * cam.stereo_eyesep;
    } else {
      // left image
//      viewport_idx_x = launch_index.x;
      eyeshift = -0.5f * cam.stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_x = launch_dim.x;
//    viewport_idx_x = launch_index.x;
    eyeshift = 0.0f;
  }

  //
  // general primary ray calculations, locked to 90-degree FoV per face...
  //
  float facescale = 1.0f / facesz;
  float2 d = make_float2(face_idx.x, face_idx.y) * facescale * 2.f - 1.0f; // center of pixel in image plane

  unsigned int randseed = tea<4>(idx, subframe_count());

  float3 col = make_float3(0.0f);
  float alpha = 0.0f;
  for (uint32_t s=0; s<rtLaunch.aa_samples; s++) {
    float2 jxy;
    jitter_offset2f(randseed, jxy);
    jxy = jxy * facescale * 2.f + d;
    float3 ray_direction = normalize(jxy.x*face_U + jxy.y*face_V + face_W);

    float3 ray_origin = cam.pos;
    if (STEREO_ON) {
      ray_origin += eyeshift * cross(ray_direction, cam.V);
    }

    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
              ray_origin, ray_origin, ray_direction, ray_direction,
              randseed, face_V, face_U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.alpha = 1.f;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = rtLaunch.max_trans;

    uint32_t p0, p1; // pack PRD pointer into p0,p1 payload regs
    packPointer(&prd, p0, p1);

    optixTrace(rtLaunch.traversable,
               ray_origin,
               ray_direction,
               0.0f,                          // tmin
               RT_DEFAULT_MAX,                // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
               RT_RAY_TYPE_RADIANCE,          // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_RADIANCE,          // missSBTIndex
               p0, p1,                        // PRD ptr in 2x uint32
               s);                            // use aasample in CH/MISS RNGs

    col += prd.result;
    alpha += prd.alpha;
  }

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(idx, make_float4(col, alpha));
#endif
}


extern "C" __global__ void __raygen__camera_cubemap() {
  tachyon_camera_cubemap_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_cubemap_dof() {
  tachyon_camera_cubemap_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_cubemap_stereo() {
  tachyon_camera_cubemap_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_cubemap_stereo_dof() {
  tachyon_camera_cubemap_general<1, 1>();
}




//
// Camera ray generation code for planetarium dome display
// Generates a fisheye style frame with ~180 degree FoV
//
// A variation of this implementation is described here:
//   A Planetarium Dome Master Camera.  John E. Stone.
//   In, Eric Haines and Tomas Akenine-M�ller, editors, 
//   Ray Tracing Gems, Apress, Chapter 4, pp. 49-60, 2019.
//   https://doi.org/10.1007/978-1-4842-4427-2_4
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_dome_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const int idx = tachyon1DLaunchIndex(launch_dim, launch_index);
#if defined(TACHYON_RAYSTATS)
  // clear/increment ray stats immediately to cut register use later
  raystats_clear_incr(idx);
#endif

  const auto &cam = rtLaunch.cam;

  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high
  // framebuffer, and the right eye into the lower half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // with simple pointer offset arithmetic.
  uint viewport_sz_y, viewport_idx_y;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // left image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyeshift = -0.5f * cam.stereo_eyesep;
    } else {
      // right image
      viewport_idx_y = launch_index.y;
      eyeshift =  0.5f * cam.stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyeshift = 0.0f;
  }

  float fov = M_PIf; // dome FoV in radians

  // half FoV in radians, pixels beyond this distance are outside
  // of the field of view of the projection, and are set black
  float thetamax = 0.5 * fov;

  // The dome angle from center of the projection is proportional
  // to the image-space distance from the center of the viewport.
  // viewport_sz contains the viewport size, radperpix contains the
  // radians/pixel scaling factors in X/Y, and viewport_mid contains
  // the midpoint coordinate of the viewpoint used to compute the
  // distance from center.
  float2 viewport_sz = make_float2(launch_dim.x, viewport_sz_y);
  float2 radperpix = fov / viewport_sz;
  float2 viewport_mid = viewport_sz * 0.5f;

  unsigned int randseed = tea<4>(idx, subframe_count());

  float3 col = make_float3(0.0f);
  float alpha = 0.0f;
  for (uint32_t s=0; s<rtLaunch.aa_samples; s++) {
    // compute the jittered image plane sample coordinate
    float2 jxy;
    jitter_offset2f(randseed, jxy);
    float2 viewport_idx = make_float2(launch_index.x, viewport_idx_y) + jxy;

    // compute the ray angles in X/Y and total angular distance from center
    float2 p = (viewport_idx - viewport_mid) * radperpix;
    float theta = hypotf(p.x, p.y);

    // pixels outside the dome FoV are treated as black by not
    // contributing to the color accumulator
    if (theta < thetamax) {
      float3 ray_direction;
      float3 ray_origin = cam.pos;

      if (theta == 0) {
        // handle center of dome where azimuth is undefined by
        // setting the ray direction to the zenith
        ray_direction = cam.W;
      } else {
        float sintheta, costheta;
        sincosf(theta, &sintheta, &costheta);
        float rsin = sintheta / theta; // normalize component
        ray_direction = cam.U*rsin*p.x + cam.V*rsin*p.y + cam.W*costheta;
        if (STEREO_ON) {
          // assumes a flat dome, where cam.W also points in the
          // audience "up" direction
          ray_origin += eyeshift * cross(ray_direction, cam.W);
        }

        if (DOF_ON) {
          float rcos = costheta / theta; // normalize component
          float3 ray_up    = -cam.U*rcos*p.x  -cam.V*rcos*p.y + cam.W*sintheta;
          float3 ray_right =  cam.U*(p.y/theta) + cam.V*(-p.x/theta);
          dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
                  ray_origin, ray_origin, ray_direction, ray_direction,
                  randseed, ray_up, ray_right);
        }
      }

      // trace the new ray...
      PerRayData_radiance prd;
      prd.alpha = 1.f;
      prd.importance = 1.f;
      prd.depth = 0;
      prd.transcnt = rtLaunch.max_trans;

      uint32_t p0, p1; // pack PRD pointer into p0,p1 payload regs
      packPointer(&prd, p0, p1);

      optixTrace(rtLaunch.traversable,
                 ray_origin,
                 ray_direction,
                 0.0f,                          // tmin
                 RT_DEFAULT_MAX,                // tmax
                 0.0f,                          // ray time
                 OptixVisibilityMask( 255 ),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
                 RT_RAY_TYPE_RADIANCE,          // SBT offset
                 RT_RAY_TYPE_COUNT,             // SBT stride
                 RT_RAY_TYPE_RADIANCE,          // missSBTIndex
                 p0, p1,                        // PRD ptr in 2x uint32
                 s);                            // use aasample in CH/MISS RNGs

      col += prd.result;
      alpha += prd.alpha;
    }
  }

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(idx, make_float4(col, alpha));
#endif
}


extern "C" __global__ void __raygen__camera_dome_master() {
  tachyon_camera_dome_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_dome_master_dof() {
  tachyon_camera_dome_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_dome_master_stereo() {
  tachyon_camera_dome_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_dome_master_stereo_dof() {
  tachyon_camera_dome_general<1, 1>();
}


//
// Camera ray generation code for 360 degre FoV
// equirectangular (lat/long) projection suitable
// for use a texture map for a sphere, e.g. for
// immersive VR HMDs, other spheremap-based projections.
//
// A variation of this implementation is described here:
//   Omnidirectional Stereoscopic Projections for VR.  John E. Stone.
//   In, William R. Sherman, editor, VR Developer Gems, 
//   Taylor and Francis / CRC Press, Chapter 24, pp. 423-436, 2019. 
//   https://www.taylorfrancis.com/chapters/edit/10.1201/b21598-24/omnidirectional-stereoscopic-projections-vr-john-stone
//
// Paul Bourke's page:
//   http://paulbourke.net/stereographics/ODSPmaths/
//
// Google:
//   https://developers.google.com/vr/jump/rendering-ods-content.pdf
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_equirectangular_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const int idx = tachyon1DLaunchIndex(launch_dim, launch_index);
#if defined(TACHYON_RAYSTATS)
  // clear/increment ray stats immediately to cut register use later
  raystats_clear_incr(idx);
#endif

  const auto &cam = rtLaunch.cam;

  // The Samsung GearVR OTOY ORBX players have the left eye image on top,
  // and the right eye image on the bottom.
  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high
  // framebuffer, and the right eye into the lower half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // with simple pointer offset arithmetic.
  uint viewport_sz_y, viewport_idx_y;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // left image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyeshift = -0.5f * cam.stereo_eyesep;
    } else {
      // right image
      viewport_idx_y = launch_index.y;
      eyeshift =  0.5f * cam.stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyeshift = 0.0f;
  }

  float2 viewport_sz = make_float2(launch_dim.x, viewport_sz_y);
  float2 radperpix = M_PIf / viewport_sz * make_float2(2.0f, 1.0f);
  float2 viewport_mid = viewport_sz * 0.5f;

  unsigned int randseed = tea<4>(idx, subframe_count());

  float3 col = make_float3(0.0f);
  float alpha = 0.0f;
  for (uint32_t s=0; s<rtLaunch.aa_samples; s++) {
    float2 jxy;
    jitter_offset2f(randseed, jxy);

    float2 viewport_idx = make_float2(launch_index.x, viewport_idx_y) + jxy;
    float2 rangle = (viewport_idx - viewport_mid) * radperpix;

    float sin_ax, cos_ax, sin_ay, cos_ay;
    sincosf(rangle.x, &sin_ax, &cos_ax);
    sincosf(rangle.y, &sin_ay, &cos_ay);

    float3 ray_direction = normalize(cos_ay * (cos_ax * cam.W + sin_ax * cam.U) + sin_ay * cam.V);

    float3 ray_origin = cam.pos;
    if (STEREO_ON) {
      ray_origin += eyeshift * cross(ray_direction, cam.V);
    }

    // compute new ray origin and ray direction
    if (DOF_ON) {
      float3 ray_right = normalize(cos_ay * (-sin_ax * cam.W - cos_ax * cam.U) + sin_ay * cam.V);
      float3 ray_up = cross(ray_direction, ray_right);
      dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
              ray_origin, ray_origin, ray_direction, ray_direction,
              randseed, ray_up, ray_right);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.alpha = 1.f;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = rtLaunch.max_trans;

    uint32_t p0, p1; // pack PRD pointer into p0,p1 payload regs
    packPointer(&prd, p0, p1);

    optixTrace(rtLaunch.traversable,
               ray_origin,
               ray_direction,
               0.0f,                          // tmin
               RT_DEFAULT_MAX,                // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
               RT_RAY_TYPE_RADIANCE,          // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_RADIANCE,          // missSBTIndex
               p0, p1,                        // PRD ptr in 2x uint32
               s);                            // use aasample in CH/MISS RNGs

    col += prd.result;
    alpha += prd.alpha;
  }

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(idx, make_float4(col, alpha));
#endif
}

extern "C" __global__ void __raygen__camera_equirectangular() {
  tachyon_camera_equirectangular_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_equirectangular_dof() {
  tachyon_camera_equirectangular_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_equirectangular_stereo() {
  tachyon_camera_equirectangular_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_equirectangular_stereo_dof() {
  tachyon_camera_equirectangular_general<1, 1>();
}



//
// Octohedral panoramic camera, defined for a square image:
//   Essential Ray Generation Shaders.  Morgan McGuire and Zander Majercik,
//   In Adam Marrs, Peter Shirley, Ingo Wald, editors,
//   Ray Tracing Gems II, Apress, Chapter 3, pp. 40-64, 2021.
//   https://link.springer.com/content/pdf/10.1007%2F978-1-4842-7185-8.pdf
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_octahedral_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const int idx = tachyon1DLaunchIndex(launch_dim, launch_index);
#if defined(TACHYON_RAYSTATS)
  // clear/increment ray stats immediately to cut register use later
  raystats_clear_incr(idx);
#endif

  const auto &cam = rtLaunch.cam;

  // The Samsung GearVR OTOY ORBX players have the left eye image on top,
  // and the right eye image on the bottom.
  // Stereoscopic rendering is provided by rendering in an over/under
  // format with the left eye image into the top half of a double-high
  // framebuffer, and the right eye into the lower half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // with simple pointer offset arithmetic.
  uint viewport_sz_y, viewport_idx_y;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-high framebuffer when stereo is enabled
    viewport_sz_y = launch_dim.y >> 1;
    if (launch_index.y >= viewport_sz_y) {
      // left image
      viewport_idx_y = launch_index.y - viewport_sz_y;
      eyeshift = -0.5f * cam.stereo_eyesep;
    } else {
      // right image
      viewport_idx_y = launch_index.y;
      eyeshift =  0.5f * cam.stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_y = launch_dim.y;
    viewport_idx_y = launch_index.y;
    eyeshift = 0.0f;
  }

  float2 viewport_sz = make_float2(launch_dim.x, viewport_sz_y);
  float2 viewport_sz_inv = make_float2(1.0f / launch_dim.x,
                                       1.0f /  viewport_sz_y);

  unsigned int randseed = tea<4>(idx, subframe_count());

  float3 col = make_float3(0.0f);
  float alpha = 0.0f;
  for (uint32_t s=0; s<rtLaunch.aa_samples; s++) {
    float2 jxy;
    jitter_offset2f(randseed, jxy);

    float2 viewport_idx = make_float2(launch_index.x, viewport_idx_y) + jxy;

    float2 px = viewport_idx * viewport_sz_inv;
    px = (px - make_float2(0.5f, 0.5f)) * 2.0f;

    // convert planar pixel coordinate to a spherical direction
    float3 ray_direction = OctDecode<1>(px);

    float3 ray_origin = cam.pos;
    if (STEREO_ON) {
      ray_origin += eyeshift * cross(ray_direction, cam.V);
    }

    // compute new ray origin and ray direction
    if (DOF_ON) {
      float3 ray_right = normalize(cross(ray_direction, cam.V));
      float3 ray_up = cross(ray_direction, ray_right);
      dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
              ray_origin, ray_origin, ray_direction, ray_direction,
              randseed, ray_up, ray_right);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.alpha = 1.f;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = rtLaunch.max_trans;

    uint32_t p0, p1; // pack PRD pointer into p0,p1 payload regs
    packPointer(&prd, p0, p1);

    optixTrace(rtLaunch.traversable,
               ray_origin,
               ray_direction,
               0.0f,                          // tmin
               RT_DEFAULT_MAX,                // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
               RT_RAY_TYPE_RADIANCE,          // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_RADIANCE,          // missSBTIndex
               p0, p1,                        // PRD ptr in 2x uint32
               s);                            // use aasample in CH/MISS RNGs

    col += prd.result;
    alpha += prd.alpha;
  }

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(idx, make_float4(col, alpha));
#endif
}

extern "C" __global__ void __raygen__camera_octahedral() {
  tachyon_camera_octahedral_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_octahedral_dof() {
  tachyon_camera_octahedral_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_octahedral_stereo() {
  tachyon_camera_octahedral_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_octahedral_stereo_dof() {
  tachyon_camera_octahedral_general<1, 1>();
}



//
// Templated Oculus Rift perspective camera ray generation code
//
template<int STEREO_ON, int DOF_ON>
static __device__ __inline__
void tachyon_camera_oculus_rift_general() {
#if defined(TACHYON_TIME_COLORING)
  clock_t t0 = clock(); // start per-pixel RT timer
#endif

  const uint3 launch_dim   = optixGetLaunchDimensions();
  const uint3 launch_index = optixGetLaunchIndex();
  const int idx = tachyon1DLaunchIndex(launch_dim, launch_index);
#if defined(TACHYON_RAYSTATS)
  // clear/increment ray stats immediately to cut register use later
  raystats_clear_incr(idx);
#endif

  const auto &cam = rtLaunch.cam;

  // Stereoscopic rendering is provided by rendering in a side-by-side
  // format with the left eye image in the left half of a double-wide
  // framebuffer, and the right eye in the right half.  The subsequent
  // OpenGL drawing code can trivially unpack and draw the two images
  // with simple pointer offset arithmetic.
  uint viewport_sz_x, viewport_idx_x;
  float eyeshift;
  if (STEREO_ON) {
    // render into a double-wide framebuffer when stereo is enabled
    viewport_sz_x = launch_dim.x >> 1;
    if (launch_index.x >= viewport_sz_x) {
      // right image
      viewport_idx_x = launch_index.x - viewport_sz_x;
      eyeshift =  0.5f * cam.stereo_eyesep;
    } else {
      // left image
      viewport_idx_x = launch_index.x;
      eyeshift = -0.5f * cam.stereo_eyesep;
    }
  } else {
    // render into a normal size framebuffer if stereo is not enabled
    viewport_sz_x = launch_dim.x;
    viewport_idx_x = launch_index.x;
    eyeshift = 0.0f;
  }

  //
  // general primary ray calculations
  //
  float2 aspect = make_float2(float(viewport_sz_x) / float(launch_dim.y), 1.0f) * cam.zoom;
  float2 viewportscale = 1.0f / make_float2(viewport_sz_x, launch_dim.y);
  float2 d = make_float2(viewport_idx_x, launch_index.y) * viewportscale * aspect * 2.f - aspect; // center of pixel in image plane


  // Compute barrel distortion required to correct for the pincushion inherent
  // in the plano-convex optics in the Oculus Rift, Google Cardboard, etc.
  // Barrel distortion involves computing distance of the pixel from the
  // center of the eye viewport, and then scaling this distance by a factor
  // based on the original distance:
  //   rnew = 0.24 * r^4 + 0.22 * r^2 + 1.0
  // Since we are only using even powers of r, we can use efficient
  // squared distances everywhere.
  // The current implementation doesn't discard rays that would have fallen
  // outside of the original viewport FoV like most OpenGL implementations do.
  // The current implementation computes the distortion for the initial ray
  // but doesn't apply these same corrections to antialiasing jitter, to
  // depth-of-field jitter, etc, so this leaves something to be desired if
  // we want best quality, but this raygen code is really intended for
  // interactive display on an Oculus Rift or Google Cardboard type viewer,
  // so I err on the side of simplicity/speed for now.
  float2 cp = make_float2(viewport_sz_x >> 1, launch_dim.y >> 1) * viewportscale * aspect * 2.f - aspect;;
  float2 dr = d - cp;
  float r2 = dr.x*dr.x + dr.y*dr.y;
  float r = 0.24f*r2*r2 + 0.22f*r2 + 1.0f;
  d = r * dr;

  int subframecount = subframe_count();
  unsigned int randseed = tea<4>(idx, subframecount);

  float3 eyepos = cam.pos;
  if (STEREO_ON) {
    eyepos += eyeshift * cam.U;
  }

  float3 ray_origin = eyepos;
  float3 col = make_float3(0.0f);
  float alpha = 0.0f;
  for (uint32_t s=0; s<rtLaunch.aa_samples; s++) {
    float2 jxy;
    jitter_offset2f(randseed, jxy);

    // don't jitter the first sample, since when using an HMD we often run
    // with only one sample per pixel unless the user wants higher fidelity
    jxy *= (subframecount > 0 || s > 0);

    jxy = jxy * viewportscale * aspect * 2.f + d;
    float3 ray_direction = normalize(jxy.x*cam.U + jxy.y*cam.V + cam.W);

    // compute new ray origin and ray direction
    if (DOF_ON) {
      dof_ray(cam.dof_focal_dist, cam.dof_aperture_rad,
              eyepos, ray_origin, ray_direction, ray_direction,
              randseed, cam.V, cam.U);
    }

    // trace the new ray...
    PerRayData_radiance prd;
    prd.alpha = 1.f;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.transcnt = rtLaunch.max_trans;

    uint32_t p0, p1; // pack PRD pointer into p0,p1 payload regs
    packPointer(&prd, p0, p1);

    optixTrace(rtLaunch.traversable,
               ray_origin,
               ray_direction,
               0.0f,                          // tmin
               RT_DEFAULT_MAX,                // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
               RT_RAY_TYPE_RADIANCE,          // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_RADIANCE,          // missSBTIndex
               p0, p1,                        // PRD ptr in 2x uint32
               s);                            // use aasample in CH/MISS RNGs

    col += prd.result;
    alpha += prd.alpha;
  }

#if defined(TACHYON_TIME_COLORING)
  accumulate_time_coloring(col, t0);
#else
  accumulate_color(idx, make_float4(col, alpha));
#endif
}

extern "C" __global__ void __raygen__camera_oculus_rift() {
  tachyon_camera_oculus_rift_general<0, 0>();
}

extern "C" __global__ void __raygen__camera_oculus_rift_dof() {
  tachyon_camera_oculus_rift_general<0, 1>();
}

extern "C" __global__ void __raygen__camera_oculus_rift_stereo() {
  tachyon_camera_oculus_rift_general<1, 0>();
}

extern "C" __global__ void __raygen__camera_oculus_rift_stereo_dof() {
  tachyon_camera_oculus_rift_general<1, 1>();
}


//
// Existing versions of Nsight Systems don't show the correct 
// OptiX raygen program name in the CUDA API trace line.
// Rather than showing the correct raygen program name, the traces
// show the name of the last raygen program defined in the loaded PTX code.
// We add this here for the time being so that any traces we capture
// with the existing broken behavior clearly indicate that we don't really
// know which raygen program was actually run.
//
extern "C" __global__ void __raygen__UNKNOWN() {
  printf("This should never happen!\n");
}


//
// Shared utility functions needed by custom geometry intersection or
// shading helper functions.
//

// normal calc routine needed only to simplify the macro to produce the
// complete combinatorial expansion of template-specialized
// closest hit radiance functions
static __inline__ __device__ 
float3 calc_ffworld_normal(const float3 &Nshading, const float3 &Ngeometric) {
  float3 world_shading_normal = normalize(optixTransformNormalFromObjectToWorldSpace(Nshading));
  float3 world_geometric_normal = normalize(optixTransformNormalFromObjectToWorldSpace(Ngeometric));
  const float3 ray_dir = optixGetWorldRayDirection();
  return faceforward(world_shading_normal, -ray_dir, world_geometric_normal);
}



//
// Object and/or vertex/color/normal buffers...
//


//
// Color-per-cone array primitive
//
extern "C" __global__ void __intersection__cone_array_color() {
  const GeomSBTHG &sbtHG = *reinterpret_cast<const GeomSBTHG*>(optixGetSbtDataPointer());
  const float3 ray_origin = optixGetObjectRayOrigin();
  const float3 obj_ray_direction = optixGetObjectRayDirection();
  const int primID = optixGetPrimitiveIndex();

  float3 base = sbtHG.cone.base[primID];
  float3 apex = sbtHG.cone.apex[primID];
  float baserad = sbtHG.cone.baserad[primID];
  float apexrad = sbtHG.cone.apexrad[primID];

  float3 axis = (apex - base);
  float3 obase = ray_origin - base;
  float3 oapex = ray_origin - apex;
  float m0 = dot(axis, axis);
  float m1 = dot(obase, axis);
  float m2 = dot(obj_ray_direction, axis);
  float m3 = dot(obj_ray_direction, obase);
  float m5 = dot(obase, obase);
  float m9 = dot(oapex, axis);

  // caps...

  float rr = baserad - apexrad;
  float hy = m0 + rr*rr;
  float k2 = m0*m0    - m2*m2*hy;
  float k1 = m0*m0*m3 - m1*m2*hy + m0*baserad*(rr*m2*1.0f             );
  float k0 = m0*m0*m5 - m1*m1*hy + m0*baserad*(rr*m1*2.0f - m0*baserad);
  float h = k1*k1 - k2*k0;
  if (h < 0.0f) 
    return; // no intersection

  float t = (-k1-sqrt(h))/k2;
  float y = m1 + t*m2;
  if (y < 0.0f || y > m0) 
    return; // no intersection
  
  optixReportIntersection(t, RT_HIT_CONE);
}


static __host__ __device__ __inline__
void get_shadevars_cone_array(const GeomSBTHG &sbtHG, float3 &shading_normal) {
  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();
  const int primID = optixGetPrimitiveIndex();

  // compute geometric and shading normals:

  float3 base = sbtHG.cone.base[primID];
  float3 apex = sbtHG.cone.apex[primID];
  float baserad = sbtHG.cone.baserad[primID];
  float apexrad = sbtHG.cone.apexrad[primID];
  
  float3 axis = (apex - base);
  float3 obase = ray_origin - base;
  float3 oapex = ray_origin - apex;
  float m0 = dot(axis, axis);
  float m1 = dot(obase, axis);
  float m2 = dot(ray_direction, axis);
  float m3 = dot(ray_direction, obase);
  float m5 = dot(obase, obase);
  float m9 = dot(oapex, axis);

  // caps...

  float rr = baserad - apexrad;
  float hy = m0 + rr*rr;
  float k2 = m0*m0    - m2*m2*hy;
  float k1 = m0*m0*m3 - m1*m2*hy + m0*baserad*(rr*m2*1.0f             );
  float k0 = m0*m0*m5 - m1*m1*hy + m0*baserad*(rr*m1*2.0f - m0*baserad);
  float h = k1*k1 - k2*k0;
//  if (h < 0.0f) 
//    return; // no intersection
 
  float t = (-k1-sqrt(h))/k2;
  float y = m1 + t*m2;
//  if (y < 0.0f || y > m0) 
//    return; // no intersection

  float3 hit = t * ray_direction; 
  float3 Ng = normalize(m0*(m0*(obase + hit) + rr*axis*baserad) - axis*hy*y);

  shading_normal = calc_ffworld_normal(Ng, Ng);
}



//
// Color-per-cylinder array primitive
//
// XXX not yet handling Obj vs. World coordinate xforms
extern "C" __global__ void __intersection__cylinder_array_color() {
  const GeomSBTHG &sbtHG = *reinterpret_cast<const GeomSBTHG*>(optixGetSbtDataPointer());
  const float3 ray_origin = optixGetObjectRayOrigin();
  const float3 obj_ray_direction = optixGetObjectRayDirection();
  const int primID = optixGetPrimitiveIndex();

  float3 start = sbtHG.cyl.start[primID];
  float3 end = sbtHG.cyl.end[primID];
  float radius = sbtHG.cyl.radius[primID];

  float3 axis = (end - start);
  float3 rc = ray_origin - start;
  float3 n = cross(obj_ray_direction, axis);
  float lnsq = dot(n, n);

  // check if ray is parallel to cylinder
  if (lnsq == 0.0f) {
    return; // ray is parallel, we missed or went through the "hole"
  }
  float invln = rsqrtf(lnsq);
  n *= invln;
  float d = fabsf(dot(rc, n));

  // check for cylinder intersection
  if (d <= radius) {
    float3 O = cross(rc, axis);
    float t = -dot(O, n) * invln;
    O = cross(n, axis);
    O = normalize(O);
    float s = dot(obj_ray_direction, O);
    s = fabs(sqrtf(radius*radius - d*d) / s);
    float axlen = length(axis);
    float3 axis_u = normalize(axis);

    // test hit point against cylinder ends
    float tin = t - s;
    float3 hit = ray_origin + obj_ray_direction * tin;
    float3 tmp2 = hit - start;
    float tmp = dot(tmp2, axis_u);
    if ((tmp > 0.0f) && (tmp < axlen)) {
      optixReportIntersection(tin, RT_HIT_CYLINDER);
    }

    // continue with second test...
    float tout = t + s;
    hit = ray_origin + obj_ray_direction * tout;
    tmp2 = hit - start;
    tmp = dot(tmp2, axis_u);
    if ((tmp > 0.0f) && (tmp < axlen)) {
      optixReportIntersection(tout, RT_HIT_CYLINDER);
    }
  }
}


static __host__ __device__ __inline__
void get_shadevars_cylinder_array(const GeomSBTHG &sbtHG, float3 &shading_normal) {
  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();
  const int primID = optixGetPrimitiveIndex();

  // compute geometric and shading normals:
  float3 start = sbtHG.cyl.start[primID];
  float3 end = sbtHG.cyl.end[primID];
  float3 axis_u = normalize(end-start);
  float3 hit = ray_origin + ray_direction * t_hit;
  float3 tmp2 = hit - start;
  float tmp = dot(tmp2, axis_u);
  float3 Ng = normalize(hit - (tmp * axis_u + start));
  shading_normal = calc_ffworld_normal(Ng, Ng);
}



#if 0

extern "C" __global__ void cylinder_array_color_bounds(int primIdx, float result[6]) {
  const float3 start = cylinder_buffer[primIdx].start;
  const float3 end = start + cylinder_buffer[primIdx].axis;
  const float3 rad = make_float3(cylinder_buffer[primIdx].radius);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = fminf(start - rad, end - rad);
    aabb->m_max = fmaxf(start + rad, end + rad);
  } else {
    aabb->invalidate();
  }
}

#endif


//
// Quadrilateral mesh primitive
//
// Based on the ray-quad approach by Ares Lagae and Philip Dutr�,
// "An efficient ray-quadrilateral intersection test"
// Journal of graphics tools, 10(4):23-32, 2005
//   https://graphics.cs.kuleuven.be/publications/LD05ERQIT/LD05ERQIT_paper.pdf
//   https://github.com/erich666/jgt-code/blob/master/Volume_10/Number_4/Lagae2005/erqit.cpp
//
// Note: The 2-D projection scheme used by Inigo Quilez is probably also worthy 
// of a look, since the later stages work in 2-D and might therefore involve
// fewer FLOPS and registers:
//   https://www.shadertoy.com/view/XtlBDs
//

// #define QUAD_VERTEX_REORDERING 1

extern "C" __global__ void __intersection__quadmesh() {
  const GeomSBTHG &sbtHG = *reinterpret_cast<const GeomSBTHG*>(optixGetSbtDataPointer());
  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const int primID = optixGetPrimitiveIndex();
  const float quadepsilon = rtLaunch.scene.epsilon * 1.0e-2f;

  auto & qmesh = sbtHG.quadmesh;

  int4 index;
  if (qmesh.indices == NULL) {
    int idx4 = primID*4;
    index = make_int4(idx4, idx4+1, idx4+2, idx4+3);
  } else {
    index = qmesh.indices[primID];
  }

  // use key variable names as per Lagae and Dutr� paper
  const float3 &v00 = qmesh.vertices[index.x];
  const float3 &v10 = qmesh.vertices[index.y];
  const float3 &v11 = qmesh.vertices[index.z];
  const float3 &v01 = qmesh.vertices[index.w];

  float3 e01 = v10 - v00;
  float3 e03 = v01 - v00;
  float3 P = cross(ray_direction, e03);
  float det = dot(e01, P);
  if (fabsf(det) < quadepsilon)
    return;

#if 0
  float inv_det = __frcp_rn(det);
#else
  float inv_det = 1.0f / det;
#endif
  float3 T = ray_origin - v00;
  float alpha = dot(T, P) * inv_det;
  if (alpha < 0.0f)
    return;
#if defined(QUAD_VERTEX_REORDERING)
  if (alpha > 1.0f) 
    return; // uncomment if vertex reordering is used
#endif
  float3 Q = cross(T, e01);
  float beta = dot(ray_direction, Q) * inv_det;
  if (beta < 0.0f)
    return;  
#if defined(QUAD_VERTEX_REORDERING)
  if (beta > 1.0f)
    return;  
#endif
  
  if ((alpha + beta) > 1.0f) {
    // reject rays that intersect plane Q 
    // to the left of v11v10 or to the right of v11v01
    float3 e23 = v01 - v11;
    float3 e21 = v10 - v11;
    float3 P_prime = cross(ray_direction, e21);
    float det_prime = dot(e23, P_prime);
    if (fabsf(det_prime) < quadepsilon)
      return;
#if 0
    float inv_det_prime = __frcp_rn(det_prime);
#else
    float inv_det_prime = 1.0f / det_prime;
#endif
    float3 T_prime = ray_origin - v11;
    float alpha_prime = dot(T_prime, P_prime) * inv_det_prime;
    if (alpha_prime < 0.0f)
      return;
    float3 Q_prime = cross(T_prime, e23);
    float beta_prime = dot(ray_direction, Q_prime) * inv_det_prime;
    if (beta_prime < 0.0f)
      return;
  }

  float t = dot(e03, Q) * inv_det;

  // do we really need to screen this for positive t-values here?
  if (t > 0.0f) { 
    // report intersection t value, and the alpha/beta values needed to
    // interpolate colors and normals during shading
    optixReportIntersection(t, RT_HIT_QUAD,
                            __float_as_int(alpha), // report alpha as attrib 0
                            __float_as_int(beta)); // report beta as attrib 1
  }
}


// calculate barycentrics for vertex v11 -- can be precomputed and stored
static __host__ __device__ __inline__
void quad_calc_barycentrics_v11(const GeomSBTHG &sbtHG,
                                float &alpha11, float &beta11) {
  const int primID = optixGetPrimitiveIndex();

  auto & qmesh = sbtHG.quadmesh;

  const int4 index = qmesh.indices[primID];

  // use key variable names and vertex order per Lagae and Dutr� paper
  // vertices are listed in counterclockwise order (v00, v10, v11, v01)
  const float3 &v00 = qmesh.vertices[index.x];
  const float3 &v10 = qmesh.vertices[index.y];
  const float3 &v11 = qmesh.vertices[index.z];
  const float3 &v01 = qmesh.vertices[index.w];

  float3 e01 = v10 - v00; // also calced during isect tests
  float3 e02 = v11 - v00; 
  float3 e03 = v01 - v00; // also calced during isect tests
  float3 n = cross(e01, e03);

  float3 absn = make_float3(fabsf(n.x), fabsf(n.y), fabsf(n.z));
  if ((absn.x >= absn.y)  && (absn.x >= absn.z)) {
    alpha11 = ((e02.y * e03.z) - (e02.z * e03.y)) / n.x;
    beta11 = ((e01.y * e02.z) - (e01.z * e02.y)) / n.x;
  } else if ((absn.y >= absn.x) && (absn.y >= absn.z)) {
    alpha11 = ((e02.z * e03.x) - (e02.x * e03.z)) / n.y;
    beta11 = ((e01.z * e02.x) - (e01.x * e02.z)) / n.y;
  } else {
    alpha11 = ((e02.x * e03.y) - (e02.y * e03.x)) / n.z;
    beta11 = ((e01.x * e02.y) - (e01.y * e02.x)) / n.z;
  }
}


// calculate bilinear interpolation parameters u and v given the
// barycentric coordinates alpha/beta obtained during intersection,
// and the barycentric coords for vertex v11, alpha11/beta11, which
// can either be computed on-demand or stored in advance
static __host__ __device__ __inline__
void quad_calc_bilinear_coords(const GeomSBTHG &sbtHG,
                               const float alpha, const float beta,
                               const float &alpha11, const float &beta11,
                               float &u, float &v) {
  const float quadepsilon = rtLaunch.scene.epsilon * 1.0e-2f;
  const float alpha11minus1 = alpha11 - 1.0f;
  const float beta11minus1  = beta11 - 1.0f;

  if (fabsf(alpha11minus1) < quadepsilon) {
    // quad is a trapezium 
    u = alpha;
    if (fabsf(beta11minus1) < quadepsilon) {
      v = beta; // quad is a parallelogram
    } else { 
      v = beta / ((u * beta11minus1) + 1.0f); // quad is a trapezium
    }
  } else if (fabsf(beta11minus1) < quadepsilon) {
    // quad is a trapezium
    v = beta;
    u = alpha / ((v * alpha11minus1) + 1.0f);
  } else {
    float A = -beta11minus1;
    float B = (alpha * beta11minus1) - (beta * alpha11minus1) - 1.0f;
    float C = alpha;
    float D = (B * B) - (4.0f * A * C);
    float Q = -0.5f * (B + (((B < 0.0f) ? -1.0f : 1.0f) * sqrtf(D)));
    u = Q / A;
    if ((u < 0.0f) || (u > 1.0f)) 
      u = C / Q;
    v = beta / ((u * beta11minus1) + 1.0f);
  }
}


static __host__ __device__ __inline__
void get_shadevars_quadmesh(const GeomSBTHG &sbtHG, float3 &hit_color,
                            float3 &shading_normal) {
  const int primID = optixGetPrimitiveIndex();

  auto & qmesh = sbtHG.quadmesh;

  int4 index;
  if (qmesh.indices == NULL) {
    int idx4 = primID*4;
    index = make_int4(idx4, idx4+1, idx4+2, idx4+3);
  } else {
    index = qmesh.indices[primID];
  }

  float alpha = __int_as_float(optixGetAttribute_0());
  float beta = __int_as_float(optixGetAttribute_1());

  // calc barycentric coords of vertex v11 
  // XXX could be precomputed and stored
  float alpha11, beta11;
  quad_calc_barycentrics_v11(sbtHG, alpha11, beta11);

  // calc bilinear interpolation parameters u and v
  float u, v;  
  quad_calc_bilinear_coords(sbtHG, alpha, beta, alpha11, beta11, u, v);

#if 0
  // in practice we can get u/v values beyond 1.0 with not-quite-planar
  // quads, which will lead to interpolation problems, 
  // so we must clamp them to the range 0->1 before use
  if (u > 1.0f || v > 1.0f)
  printf("quad: u:%.2f v:%.2f a:%.2f b:%.2f a11:%.2f b11:%.2f\n",
         u, v, alpha, beta, alpha11, beta11);
#endif

  u = __saturatef(u);
  v = __saturatef(v);

  // compute geometric and shading normals:
  float3 Ng, Ns;
  if (qmesh.normals != nullptr) {
    const float3 &v00 = qmesh.vertices[index.x];
    const float3 &v10 = qmesh.vertices[index.y];
//    const float3 &v11 = qmesh.vertices[index.z];
    const float3 &v01 = qmesh.vertices[index.w];
    Ng = normalize(cross(v01-v00, v10-v00)); ///< XXX fix for quads 

    const float3& n00 = qmesh.normals[index.x];
    const float3& n10 = qmesh.normals[index.y];
    const float3& n11 = qmesh.normals[index.z];
    const float3& n01 = qmesh.normals[index.w];

    // interpolate quad normal using bilinear params u and v
    Ns = normalize(((n00 * (1.0f - u)) + (n10 * u)) * (1.0f - v) +
                   ((n01 * (1.0f - u)) + (n11 * u)) * v);
  } else {
    const float3 &v00 = qmesh.vertices[index.x];
    const float3 &v10 = qmesh.vertices[index.y];
//    const float3 &v11 = qmesh.vertices[index.z];
    const float3 &v01 = qmesh.vertices[index.w];
    Ns = Ng = normalize(cross(v01-v00, v10-v00)); ///< XXX fix for quads 
  }
  shading_normal = calc_ffworld_normal(Ns, Ng);

  // Assign vertex-interpolated, per-primitive or uniform color
  if (qmesh.vertcolors3f != nullptr) {
    const float3 c00 = qmesh.vertcolors3f[index.x];
    const float3 c10 = qmesh.vertcolors3f[index.y];
    const float3 c11 = qmesh.vertcolors3f[index.z];
    const float3 c01 = qmesh.vertcolors3f[index.w];

    // interpolate quad color using bilinear params u and v
    hit_color = ((c00 * (1.0f - u)) + (c10 * u)) * (1.0f - v) +
                ((c01 * (1.0f - u)) + (c11 * u)) * v;
  } else if (qmesh.vertcolors4u != nullptr) {
    const float ci2f = 1.0f / 255.0f;
    const float3 c00 = qmesh.vertcolors4u[index.x] * ci2f;
    const float3 c10 = qmesh.vertcolors4u[index.y] * ci2f;
    const float3 c11 = qmesh.vertcolors4u[index.z] * ci2f;
    const float3 c01 = qmesh.vertcolors4u[index.w] * ci2f;

    // interpolate quad color using bilinear params u and v
    hit_color = ((c00 * (1.0f - u)) + (c10 * u)) * (1.0f - v) +
                ((c01 * (1.0f - u)) + (c11 * u)) * v;
  } else if (sbtHG.prim_color != nullptr) {
    hit_color = sbtHG.prim_color[primID];
  } else {
    hit_color = sbtHG.uniform_color;
  }
}



//
// Ring array primitive
//
extern "C" __global__ void __intersection__ring_array() {
  const GeomSBTHG &sbtHG = *reinterpret_cast<const GeomSBTHG*>(optixGetSbtDataPointer());
  const float3 obj_ray_origin = optixGetObjectRayOrigin();
  const float3 obj_ray_direction = optixGetObjectRayDirection();
  const int primID = optixGetPrimitiveIndex();

  const float3 center = sbtHG.ring.center[primID];
  const float3 norm = sbtHG.ring.norm[primID];
  const float inrad = sbtHG.ring.inrad[primID];
  const float outrad = sbtHG.ring.outrad[primID];

  float d = -dot(center, norm);
  float t = -(d + dot(norm, obj_ray_origin));
  float td = dot(norm, obj_ray_direction);
  if (td != 0.0f) {
    t /= td;
    if (t >= 0.0f) {
      float3 hit = obj_ray_origin + t * obj_ray_direction;
      float rd = length(hit - center);
      if ((rd > inrad) && (rd < outrad)) {
        optixReportIntersection(t, RT_HIT_RING);
      }
    }
  }
}


static __host__ __device__ __inline__
void get_shadevars_ring_array(const GeomSBTHG &sbtHG, float3 &shading_normal) {
  const int primID = optixGetPrimitiveIndex();

  // compute geometric and shading normals:
  float3 Ng = sbtHG.ring.norm[primID];
  shading_normal = calc_ffworld_normal(Ng, Ng);
}



#if 0

extern "C" __global__ void ring_array_color_bounds(int primIdx, float result[6]) {
  const float3 center = ring_buffer[primIdx].center;
  const float3 rad = make_float3(ring_buffer[primIdx].outrad);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = center - rad;
    aabb->m_max = center + rad;
  } else {
    aabb->invalidate();
  }
}

#endif



#if defined(TACHYON_USE_SPHERES_HEARNBAKER)

// Ray-sphere intersection method with improved floating point precision
// for cases where the sphere size is small relative to the distance
// from the camera to the sphere.  This implementation is based on
// Eq. 10-72, p.603 of "Computer Graphics with OpenGL", 3rd Ed., by 
// Donald Hearn and Pauline Baker, 2004, Eq. 10, p.639 in the 4th edition 
// (Hearn, Baker, Carithers), and in Ray Tracing Gems, 
// Precision Improvements for Ray/Sphere Intersection, pp. 87-94, 2019.
static __host__ __device__ __inline__
void sphere_intersect_hearn_baker(float3 center, float rad) {
  const float3 ray_origin = optixGetObjectRayOrigin();
  const float3 obj_ray_direction = optixGetObjectRayDirection();

  // if scaling xform was been applied, the ray length won't be normalized, 
  // so we have to scale the resulting t hitpoints to world coords
  float ray_invlen;
  const float3 ray_direction = normalize_len(obj_ray_direction, ray_invlen);

  float3 deltap = center - ray_origin;
  float ddp = dot(ray_direction, deltap);
  float3 remedyTerm = deltap - ddp * ray_direction;
  float disc = rad*rad - dot(remedyTerm, remedyTerm);
  if (disc >= 0.0f) {
    float disc_root = sqrtf(disc);

#if 0 && defined(FASTONESIDEDSPHERES)
    float t1 = ddp - disc_root;
    t1 *= ray_invlen; // transform t value back to world coordinates
    optixReportIntersection(t1, RT_HIT_SPHERE);
#else
    float t1 = ddp - disc_root;
    t1 *= ray_invlen; // transform t value back to world coordinates
    optixReportIntersection(t1, RT_HIT_SPHERE);

    float t2 = ddp + disc_root;
    t2 *= ray_invlen; // transform t value back to world coordinates
    optixReportIntersection(t2, RT_HIT_SPHERE);
#endif
  }
}

#else

//
// Ray-sphere intersection using standard geometric solution approach
//
static __host__ __device__ __inline__
void sphere_intersect_classic(float3 center, float rad) {
  const float3 ray_origin = optixGetObjectRayOrigin();
  const float3 obj_ray_direction = optixGetObjectRayDirection();

  // if scaling xform was been applied, the ray length won't be normalized, 
  // so we have to scale the resulting t hitpoints to world coords
  float ray_invlen;
  const float3 ray_direction = normalize_len(obj_ray_direction, ray_invlen);

  float3 deltap = center - ray_origin;
  float ddp = dot(ray_direction, deltap);
  float disc = ddp*ddp + rad*rad - dot(deltap, deltap);
  if (disc > 0.0f) {
    float disc_root = sqrtf(disc);

#if 0 && defined(FASTONESIDEDSPHERES)
    // only calculate the nearest intersection, for speed
    float t1 = ddp - disc_root;
    t1 *= ray_invlen; // transform t value back to world coordinates
    optixReportIntersection(t1, RT_HIT_SPHERE);
#else
    float t2 = ddp + disc_root;
    t2 *= ray_invlen; // transform t value back to world coordinates
    optixReportIntersection(t2, RT_HIT_SPHERE);

    float t1 = ddp - disc_root;
    t1 *= ray_invlen; // transform t value back to world coordinates
    optixReportIntersection(t1, RT_HIT_SPHERE);
#endif
  }
}

#endif


//
// Sphere array primitive
//
extern "C" __global__ void __intersection__sphere_array() {
  const GeomSBTHG &sbtHG = *reinterpret_cast<const GeomSBTHG*>(optixGetSbtDataPointer());
  const int primID = optixGetPrimitiveIndex();
  float4 xyzr = sbtHG.sphere.PosRadius[primID];
  float3 center = make_float3(xyzr);
  float radius = xyzr.w;

#if defined(TACHYON_USE_SPHERES_HEARNBAKER)
  sphere_intersect_hearn_baker(center, radius);
#else
  sphere_intersect_classic(center, radius);
#endif
}


static __host__ __device__ __inline__
void get_shadevars_sphere_array(const GeomSBTHG &sbtHG, float3 &shading_normal) {
  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();
  const int primID = optixGetPrimitiveIndex();

  // compute geometric and shading normals:
  float4 xyzr = sbtHG.sphere.PosRadius[primID];
  float3 center = make_float3(xyzr);
  float radius = xyzr.w;
  float3 deltap = center - ray_origin;
  float3 Ng = (t_hit * ray_direction - deltap) * (1.0f / radius);
  shading_normal = calc_ffworld_normal(Ng, Ng);
}


#if 0
extern "C" __global__ void sphere_array_bounds(int primIdx, float result[6]) {
  const float3 cen = sphere_buffer[primIdx].center;
  const float3 rad = make_float3(sphere_buffer[primIdx].radius);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = cen - rad;
    aabb->m_max = cen + rad;
  } else {
    aabb->invalidate();
  }
}
#endif


#if 0
// OptiX 6.x bounds code
extern "C" __global__ void sphere_array_color_bounds(int primIdx, float result[6]) {
  const float3 cen = sphere_color_buffer[primIdx].center;
  const float3 rad = make_float3(sphere_color_buffer[primIdx].radius);
  optix::Aabb* aabb = (optix::Aabb*)result;

  if (rad.x > 0.0f && !isinf(rad.x)) {
    aabb->m_min = cen - rad;
    aabb->m_max = cen + rad;
  } else {
    aabb->invalidate();
  }
}
#endif



//
// Curve array primitives
//
#if OPTIX_VERSION >= 70100
static __host__ __device__ __inline__
void get_shadevars_curves_linear(const GeomSBTHG &sbtHG, float3 &hit_color,
                                 float3 &shading_normal) {
  const int primID = optixGetPrimitiveIndex();

  // XXX unfinished
  hit_color = sbtHG.uniform_color;
  shading_normal = make_float3(0.0f, 0.0f, -1.0f); 
}
#endif

#if OPTIX_VERSION >= 70400
static __host__ __device__ __inline__
void get_shadevars_curves_catmullrom(const GeomSBTHG &sbtHG, float3 &hit_color,
                                     float3 &shading_normal) {
  const int primID = optixGetPrimitiveIndex();

  // XXX unfinished
  hit_color = sbtHG.uniform_color;
  shading_normal = make_float3(0.0f, 0.0f, -1.0f); 
}
#endif


//
// Triangle mesh/array primitives
//

static __host__ __device__ __inline__
void get_shadevars_trimesh(const GeomSBTHG &sbtHG, float3 &hit_color,
                           float &hit_alpha, float3 &shading_normal) {
  const int primID = optixGetPrimitiveIndex();

  auto & tmesh = sbtHG.trimesh;

  int3 index;
  if (tmesh.indices == NULL) {
    int idx3 = primID*3;
    index = make_int3(idx3, idx3+1, idx3+2);
  } else {
    index = tmesh.indices[primID];
  }

  const float2 barycentrics = optixGetTriangleBarycentrics();

  // compute geometric and shading normals:
  float3 Ng, Ns;
  if (tmesh.packednormals != nullptr) {

#if 1
    // XXX packed normals currently only work for implicit indexed buffers
    Ng = unpackNormal(tmesh.packednormals[primID].x);
    const float3& n0 = unpackNormal(tmesh.packednormals[primID].y);
    const float3& n1 = unpackNormal(tmesh.packednormals[primID].z);
    const float3& n2 = unpackNormal(tmesh.packednormals[primID].w);
#else
    // XXX we can't use indexing for uint4 packed normals
    Ng = unpackNormal(tmesh.packednormals[index].x);
    const float3& n0 = unpackNormal(tmesh.packednormals[index.x].y);
    const float3& n1 = unpackNormal(tmesh.packednormals[index.y].z);
    const float3& n2 = unpackNormal(tmesh.packednormals[index.z].w);
#endif

    // interpolate triangle normal from barycentrics
    Ns = normalize(n0 * (1.0f - barycentrics.x - barycentrics.y) +
                   n1 * barycentrics.x + n2 * barycentrics.y);
  } else if (tmesh.normals != nullptr) {
    const float3 &A = tmesh.vertices[index.x];
    const float3 &B = tmesh.vertices[index.y];
    const float3 &C = tmesh.vertices[index.z];
    Ng = normalize(cross(B-A, C-A));

    const float3& n0 = tmesh.normals[index.x];
    const float3& n1 = tmesh.normals[index.y];
    const float3& n2 = tmesh.normals[index.z];

    // interpolate triangle normal from barycentrics
    Ns = normalize(n0 * (1.0f - barycentrics.x - barycentrics.y) +
                   n1 * barycentrics.x + n2 * barycentrics.y);
  } else {
    const float3 &A = tmesh.vertices[index.x];
    const float3 &B = tmesh.vertices[index.y];
    const float3 &C = tmesh.vertices[index.z];
    Ns = Ng = normalize(cross(B-A, C-A));
  }
  shading_normal = calc_ffworld_normal(Ns, Ng);

  // Assign texture, vertex-interpolated, per-primitive or uniform color
  if (tmesh.vertcolors4u != nullptr) {
    const float ci2f = 1.0f / 255.0f;
    const float3 c0 = tmesh.vertcolors4u[index.x] * ci2f;
    const float3 c1 = tmesh.vertcolors4u[index.y] * ci2f;
    const float3 c2 = tmesh.vertcolors4u[index.z] * ci2f;

    // interpolate triangle color from barycentrics
    hit_color = (c0 * (1.0f - barycentrics.x - barycentrics.y) +
                 c1 * barycentrics.x + c2 * barycentrics.y);
  } else if (tmesh.vertcolors3f != nullptr) {
    const float3 c0 = tmesh.vertcolors3f[index.x];
    const float3 c1 = tmesh.vertcolors3f[index.y];
    const float3 c2 = tmesh.vertcolors3f[index.z];

    // interpolate triangle color from barycentrics
    hit_color = (c0 * (1.0f - barycentrics.x - barycentrics.y) +
                 c1 * barycentrics.x + c2 * barycentrics.y);
  } else if (sbtHG.prim_color != nullptr) {
    hit_color = sbtHG.prim_color[primID];
  } else if (tmesh.tex2d != nullptr) {
    float2 txc0 = tmesh.tex2d[index.x];
    float2 txc1 = tmesh.tex2d[index.y];
    float2 txc2 = tmesh.tex2d[index.z];

    // interpolate tex coord from triangle barycentrics
    float2 texcoord = (txc0 * (1.0f - barycentrics.x - barycentrics.y) +
                       txc1 * barycentrics.x + txc2 * barycentrics.y);

    // XXX need to implement ray differentials for tex filtering
    int matidx = sbtHG.materialindex;
    const auto &mat = rtLaunch.materials[matidx];
    float4 tx = tex2D<float4>(mat.tex, texcoord.x, texcoord.y);
    hit_color = make_float3(tx);
    hit_alpha = tx.w; // overwrite hit_alpha when available
  } else if (tmesh.tex3d != nullptr) {
    float3 txc0 = tmesh.tex3d[index.x];
    float3 txc1 = tmesh.tex3d[index.y];
    float3 txc2 = tmesh.tex3d[index.z];

    // interpolate tex coord from triangle barycentrics
    float3 texcoord = (txc0 * (1.0f - barycentrics.x - barycentrics.y) +
                       txc1 * barycentrics.x + txc2 * barycentrics.y);

    // XXX need to implement ray differentials for tex filtering
    int matidx = sbtHG.materialindex;
    const auto &mat = rtLaunch.materials[matidx];
    float4 tx = tex3D<float4>(mat.tex, texcoord.x, texcoord.y, texcoord.z);
    hit_color = make_float3(tx);
    hit_alpha = tx.w; // overwrite hit_alpha when available
  } else {
    hit_color = sbtHG.uniform_color;
  }
}



#if 0

// inline device function for computing triangle bounding boxes
__device__ __inline__ void generic_tri_bounds(optix::Aabb *aabb,
                                              float3 v0, float3 v1, float3 v2) {
#if 1
  // conventional paranoid implementation that culls degenerate triangles
  float area = length(cross(v1-v0, v2-v0));
  if (area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf(fminf(v0, v1), v2);
    aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
  } else {
    aabb->invalidate();
  }
#else
  // don't cull any triangles, even if they might be degenerate
  aabb->m_min = fminf(fminf(v0, v1), v2);
  aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
#endif
}


//
// triangle mesh with vertices, geometric normal, uniform color
//
extern "C" __global__ void ort_tri_intersect(int primIdx) {
  float3 v0 = stri_buffer[primIdx].v0;
  float3 v1 = tri_buffer[primIdx].v1;
  float3 v2 = tri_buffer[primIdx].v2;

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      shading_normal = geometric_normal = normalize(n);

      // uniform color for the entire object
      prim_color = uniform_color;
      rtReportIntersection(0);
    }
  }
}

extern "C" __global__ void ort_tri_bounds(int primIdx, float result[6]) {
  float3 v0 = tri_buffer[primIdx].v0;
  float3 v1 = tri_buffer[primIdx].v1;
  float3 v2 = tri_buffer[primIdx].v2;

  optix::Aabb *aabb = (optix::Aabb*)result;
  generic_tri_bounds(aabb, v0, v1, v2);
}


//
// triangle mesh with vertices, smoothed normals, uniform color
//
extern "C" __global__ void ort_stri_intersect(int primIdx) {
  float3 v0 = stri_buffer[primIdx].v0;
  float3 v1 = stri_buffer[primIdx].v1;
  float3 v2 = stri_buffer[primIdx].v2;

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      float3 n0 = stri_buffer[primIdx].n0;
      float3 n1 = stri_buffer[primIdx].n1;
      float3 n2 = stri_buffer[primIdx].n2;
      shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f-beta-gamma));
      geometric_normal = normalize(n);

      // uniform color for the entire object
      prim_color = uniform_color;
      rtReportIntersection(0);
    }
  }
}

extern "C" __global__ void ort_stri_bounds(int primIdx, float result[6]) {
  float3 v0 = stri_buffer[primIdx].v0;
  float3 v1 = stri_buffer[primIdx].v1;
  float3 v2 = stri_buffer[primIdx].v2;

  optix::Aabb *aabb = (optix::Aabb*)result;
  generic_tri_bounds(aabb, v0, v1, v2);
}


//
// triangle mesh with vertices, smoothed normals, colors
//
extern "C" __global__ void ort_vcstri_intersect(int primIdx) {
  float3 v0 = vcstri_buffer[primIdx].v0;
  float3 v1 = vcstri_buffer[primIdx].v1;
  float3 v2 = vcstri_buffer[primIdx].v2;

  // Intersect ray with triangle
  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma)) {
    if (rtPotentialIntersection(t)) {
      float3 n0 = vcstri_buffer[primIdx].n0;
      float3 n1 = vcstri_buffer[primIdx].n1;
      float3 n2 = vcstri_buffer[primIdx].n2;
      shading_normal = normalize(n1*beta + n2*gamma + n0*(1.0f-beta-gamma));
      geometric_normal = normalize(n);

      float3 c0 = vcstri_buffer[primIdx].c0;
      float3 c1 = vcstri_buffer[primIdx].c1;
      float3 c2 = vcstri_buffer[primIdx].c2;
      prim_color = c1*beta + c2*gamma + c0*(1.0f-beta-gamma);
      rtReportIntersection(0);
    }
  }
}

extern "C" __global__ void ort_vcstri_bounds(int primIdx, float result[6]) {
  float3 v0 = vcstri_buffer[primIdx].v0;
  float3 v1 = vcstri_buffer[primIdx].v1;
  float3 v2 = vcstri_buffer[primIdx].v2;

  optix::Aabb *aabb = (optix::Aabb*)result;
  generic_tri_bounds(aabb, v0, v1, v2);
}

#endif



//
// Support functions for closest hit and any hit programs for radiance rays
//  

// Fog implementation
static __device__ __forceinline__ float fog_coord(float3 hit_point) {
  // Compute planar fog (e.g. to match OpenGL) by projecting t value onto
  // the camera view direction vector to yield a planar a depth value.
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();
  const auto &scene = rtLaunch.scene;

  float r = dot(ray_direction, rtLaunch.cam.W) * t_hit;
  float f=1.0f;
  float v;

  switch (scene.fog_mode) {
    case 1: // RT_FOG_LINEAR
      f = (scene.fog_end - r) / (scene.fog_end - scene.fog_start);
      break;

    case 2: // RT_FOG_EXP
      // XXX Tachyon needs to allow fog_start to be non-zero for 
      //     exponential fog, but fixed-function OpenGL does not...
      // float v = fog_density * (r - fog_start);
      v = scene.fog_density * r;
      f = expf(-v);
      break;

    case 3: // RT_FOG_EXP2
      // XXX Tachyon needs to allow fog_start to be non-zero for 
      //     exponential fog, but fixed-function OpenGL does not...
      // float v = fog_density * (r - fog_start);
      v = scene.fog_density * r;
      f = expf(-v*v);
      break;

    case 0: // RT_FOG_NONE
    default:
      break;
  }
  return __saturatef(f);
}


static __device__ __forceinline__ float3 fog_color(float fogmod, float3 hit_col) {
  float3 col = (fogmod * hit_col) + ((1.0f - fogmod) * rtLaunch.scene.bg_color);
  return col;
}



//
// trivial ambient occlusion implementation
//
static __device__ float shade_ambient_occlusion(float3 hit, float3 N, float aoimportance) {
  float inten = 0.0f;

  // Improve AO RNG seed generation when more than one AA sample is run
  // per rendering pass.  The classic OptiX 6 formulation doesn't work
  // as well now that we do our own subframe counting, and with RTX hardware
  // we often want multiple AA samples per pass now unlike before.
  unsigned int aas = 1+getPayloadAAsample(); // add one to prevent a zero
  int teabits1 = aas * subframe_count() * 313331337;
  unsigned int randseed = tea<2>(teabits1, teabits1);

  // do all the samples requested, with no observance of importance
  for (int s=0; s<rtLaunch.lights.ao_samples; s++) {
    float3 dir;
    jitter_sphere3f(randseed, dir);
    float ndotambl = dot(N, dir);

    // flip the ray so it's in the same hemisphere as the surface normal
    if (ndotambl < 0.0f) {
      ndotambl = -ndotambl;
      dir = -dir;
    }

    float3 aoray_origin, aoray_direction;
    float tmax=rtLaunch.lights.ao_maxdist;
#ifdef USE_REVERSE_SHADOW_RAYS
    if (shadows_enabled == RT_SHADOWS_ON_REVERSE) {
      // reverse any-hit ray traversal direction for increased perf
      // XXX We currently hard-code REVERSE_RAY_LENGTH in such a way that
      //     it works well for scenes that fall within the view volume,
      //     given the relationship between the model and camera coordinate
      //     systems, but this would be best computed by the diagonal of the
      //     AABB for the full scene, and then scaled into camera coordinates.
      //     The REVERSE_RAY_STEP size is computed to avoid self intersection
      //     with the surface we're shading.
      tmax = REVERSE_RAY_LENGTH - REVERSE_RAY_STEP;
      aoray_origin = hit + dir * REVERSE_RAY_LENGTH;
      aoray_direction = -dir;
    } else
#endif
    {
#if defined(TACHYON_USE_RAY_STEP)
      aoray_origin = hit + TACHYON_RAY_STEP;
#else
      aoray_origin = hit;
#endif
      aoray_direction = dir;
    }

    // initialize per-ray shadow attenuation as "no-occlusion"
    uint32_t p0 = __float_as_int(1.0f);

    optixTrace(rtLaunch.traversable,
               aoray_origin,
               aoray_direction,
               0.0f,                          // tmin
               tmax,                          // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
#if 1
               OPTIX_RAY_FLAG_NONE,
#elif 1
               // Hard shadows only, no opacity filtering.
               // For shadow rays skip any/closest hit and terminate 
               // on first intersection with anything.
               OPTIX_RAY_FLAG_DISABLE_ANYHIT
               | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
               | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
#endif
               RT_RAY_TYPE_SHADOW,            // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_SHADOW,            // missSBTIndex
               p0);                           // send attention
  
    inten += ndotambl * __int_as_float(p0); // fetch attenuation from p0
  }

  // unweighted non-importance-sampled scaling factor
  return inten * rtLaunch.lights.ao_lightscale;
}



template<int SHADOWS_ON>       /// scene-wide shading property
static __device__ __inline__ void shade_light(float3 &result,
                                              float3 &hit_point,
                                              float3 &N, float3 &L,
                                              float p_Kd,
                                              float p_Ks,
                                              float p_phong_exp,
                                              float3 &col,
                                              float3 &phongcol,
                                              float shadow_tmax) {
  float inten = dot(N, L);

  // cast shadow ray
  float light_attenuation = static_cast<float>(inten > 0.0f);
  if (SHADOWS_ON && rtLaunch.lights.shadows_enabled && inten > 0.0f) {

    float3 shadowray_origin, shadowray_direction;
    float tmax=shadow_tmax;
#ifdef USE_REVERSE_SHADOW_RAYS
    if (shadows_enabled == RT_SHADOWS_ON_REVERSE) {
      // reverse any-hit ray traversal direction for increased perf
      // XXX We currently hard-code REVERSE_RAY_LENGTH in such a way that
      //     it works well for scenes that fall within the view volume,
      //     given the relationship between the model and camera coordinate
      //     systems, but this would be best computed by the diagonal of the
      //     AABB for the full scene, and then scaled into camera coordinates.
      //     The REVERSE_RAY_STEP size is computed to avoid self intersection
      //     with the surface we're shading.
      tmax = REVERSE_RAY_LENGTH - REVERSE_RAY_STEP;
      shadowray_origin = hit_point + L * REVERSE_RAY_LENGTH;
      shadowray_direction = -L
      tmax = fminf(tmax, shadow_tmax));
    }
    else
#endif
    {
      shadowray_origin = hit_point + TACHYON_RAY_STEP;
      shadowray_direction = L;
    }

    // initialize per-ray shadow attenuation as "no-occlusion"
    uint32_t p0 = __float_as_int(1.0f);

    optixTrace(rtLaunch.traversable,
               shadowray_origin,
               shadowray_direction,
               0.0f,                          // tmin
               tmax,                          // tmax
               0.0f,                          // ray time
               OptixVisibilityMask( 255 ),
#if 1
               OPTIX_RAY_FLAG_NONE,
#else
               // Hard shadows only, no opacity filtering.
               // For shadow rays skip any/closest hit and terminate 
               // on first intersection with anything.
               OPTIX_RAY_FLAG_DISABLE_ANYHIT
               | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
               | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
#endif
               RT_RAY_TYPE_SHADOW,            // SBT offset
               RT_RAY_TYPE_COUNT,             // SBT stride
               RT_RAY_TYPE_SHADOW,            // missSBTIndex
               p0);                           // send attention
    light_attenuation = __int_as_float(p0); // get attenuation from p0

#if defined(TACHYON_RAYSTATS)
    const int idx = tachyon1DLaunchIndex();
    rtLaunch.frame.raystats1_buffer[idx].y++; // increment shadow ray counter
#endif
  }

  // If not completely shadowed, light the hit point.
  // When shadows are disabled, the light can't possibly be attenuated.
  if (!SHADOWS_ON || light_attenuation > 0.0f) {
    result += col * p_Kd * inten * light_attenuation;

    // add specular hightlight using Blinn's halfway vector approach
    const float3 ray_direction = optixGetWorldRayDirection();
    float3 H = normalize(L - ray_direction);
    float nDh = dot(N, H);
    if (nDh > 0) {
      float power = powf(nDh, p_phong_exp);
      phongcol += make_float3(p_Ks) * power * light_attenuation;
    }
  }
}




//
// Partial re-implementation of the key portions of Tachyon's "full" shader.
//
// This shader has been written to be expanded into a large set of
// fully specialized shaders generated through combinatorial expansion
// of each of the major shader features associated with scene-wide or
// material-specific shading properties.
// At present, there are three scene-wide properties (fog, shadows, AO),
// and three material-specific properties (outline, reflection, transmission).
// There can be a performance cost for OptiX work scheduling of disparate
// materials if too many unique materials are used in a scene.
// Although there are 8 combinations of scene-wide parameters and
// 8 combinations of material-specific parameters (64 in total),
// the scene-wide parameters are uniform for the whole scene.
// We will therefore only have at most 8 different shader variants
// in use in a given scene, due to the 8 possible combinations
// of material-specific (outline, reflection, transmission) properties.
//
// The macros that generate the full set of 64 possible shader variants
// are at the very end of this source file.
//
template<int CLIP_VIEW_ON,     /// scene-wide shading property
         int HEADLIGHT_ON,     /// scene-wide shading property
         int FOG_ON,           /// scene-wide shading property
         int SHADOWS_ON,       /// scene-wide shading property
         int AO_ON,            /// scene-wide shading property
         int OUTLINE_ON,       /// material-specific shading property
         int REFLECTION_ON,    /// material-specific shading property
         int TRANSMISSION_ON>  /// material-specific shading property
static __device__ void shader_template(float3 prim_color, float3 N,
                                       float p_Ka, float p_Kd, float p_Ks,
                                       float p_phong_exp, float p_reflectivity,
                                       float p_opacity,
                                       float p_outline, float p_outlinewidth,
                                       int p_transmode) {
  PerRayData_radiance &prd = *getPRD<PerRayData_radiance>();
  const float3 ray_origin = optixGetWorldRayOrigin();
  const float3 ray_direction = optixGetWorldRayDirection();
  const float t_hit = optixGetRayTmax();

  float3 hit_point = ray_origin + t_hit * ray_direction;
  float3 result = make_float3(0.0f);
  float3 phongcol = make_float3(0.0f);

  // add depth cueing / fog if enabled
  // use fog coordinate to modulate importance for AO rays, etc.
  float fogmod = 1.0f;
  if (FOG_ON && rtLaunch.scene.fog_mode != 0) {
    fogmod = fog_coord(hit_point);
  }

#if defined(TACHYON_RAYSTATS)
  // compute and reuse 1-D pixel index
  const int idx = tachyon1DLaunchIndex();
#endif

#if 1
  // don't render transparent surfaces if we've reached the max count
  // this implements the same logic as the -trans_max_surfaces argument
  // in the CPU version of Tachyon, and is described in:
  //   Interactive Ray Tracing Techniques for High-Fidelity 
  //   Scientific Visualization.  John E. Stone.
  //   In, Eric Haines and Tomas Akenine-M�ller, editors, 
  //   Ray Tracing Gems, Apress, Chapter 27, pp. 493-515, 2019.
  //   https://doi.org/10.1007/978-1-4842-4427-2_27
  // 
  if ((p_opacity < 1.0f) && (prd.transcnt < 1)) {
    // shoot a transmission ray
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * (1.0f - p_opacity);
    new_prd.alpha = 1.0f;
    new_prd.result = rtLaunch.scene.bg_color;

    // For correct operation with the RTX runtime strategy and its
    // associated stack management scheme, we MUST increment the
    // ray recursion depth counter when performing transparent surface
    // peeling, otherwise we could go beyond the max recursion depth
    // that we previously requested from OptiX.  This will work less well
    // than the former approach in terms of visual outcomes, but we presently
    // have no alternative and must avoid issues with runtime stack overruns.
    new_prd.depth = prd.depth + 1;
    new_prd.transcnt = 0; // don't decrement further since unsigned int type

    if (new_prd.importance >= 0.001f && new_prd.depth < rtLaunch.max_depth) {
      float3 transray_direction = ray_direction;
      float3 transray_origin;
#if defined(TACHYON_USE_RAY_STEP)
#if defined(TACHYON_TRANS_USE_INCIDENT)
      // step the ray in the incident ray direction
      transray_origin = hit_point + TACHYON_RAY_STEP2;
#else
      // step the ray in the direction opposite the surface normal (going in)
      // rather than out, for transmission rays...
      transray_origin = hit_point - TACHYON_RAY_STEP;
#endif
#else
      transray_origin = hit_point;
#endif

      // the values we store the PRD pointer in:
      uint32_t p0, p1;
      packPointer( &new_prd, p0, p1 );
      uint32_t s = getPayloadAAsample(); // use aasample in CH/MISS RNGs

      optixTrace(rtLaunch.traversable,
                 transray_origin,
                 transray_direction,
                 0.0f,                          // tmin
                 RT_DEFAULT_MAX,                // tmax
                 0.0f,                          // ray time
                 OptixVisibilityMask( 255 ),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
                 RT_RAY_TYPE_RADIANCE,          // SBT offset
                 RT_RAY_TYPE_COUNT,             // SBT stride
                 RT_RAY_TYPE_RADIANCE,          // missSBTIndex
                 p0, p1,                        // PRD ptr in 2x uint32
                 s);                            // use aasample in CH/MISS RNGs

#if defined(TACHYON_RAYSTATS)
      rtLaunch.frame.raystats2_buffer[idx].x++; // increment trans ray counter
#endif
    }
    prd.result = new_prd.result;
    return; // early-exit
  }
#endif

  // execute the object's texture function
  float3 col = prim_color; // XXX no texturing implemented yet

  // compute lighting from directional lights
  for (int i=0; i < rtLaunch.lights.num_dir_lights; i++) {
    float3 L = rtLaunch.lights.dir_lights[i];
    shade_light<SHADOWS_ON>(result, hit_point, N, L, p_Kd, p_Ks, p_phong_exp,
                            col, phongcol, RT_DEFAULT_MAX);
  }

  // compute lighting from positional lights
  for (int i=0; i < rtLaunch.lights.num_pos_lights; i++) {
    float3 Lpos = rtLaunch.lights.pos_lights[i];
    float3 L = Lpos - hit_point;
    float shadow_tmax;
    L = normalize_len(L, shadow_tmax); // normalize and compute shadow tmax
    shade_light<SHADOWS_ON>(result, hit_point, N, L, p_Kd, p_Ks, p_phong_exp,
                            col, phongcol, shadow_tmax);
  }

  // add point light for camera headlight need for Oculus Rift HMDs,
  // equirectangular panorama images, and planetarium dome master images
  if (HEADLIGHT_ON && (rtLaunch.lights.headlight_mode != 0)) {
    float3 L = rtLaunch.cam.pos - hit_point;
    float shadow_tmax;
    L = normalize_len(L, shadow_tmax); // normalize and compute shadow tmax
    shade_light<SHADOWS_ON>(result, hit_point, N, L, p_Kd, p_Ks, p_phong_exp,
                            col, phongcol, shadow_tmax);
  }

  // add ambient occlusion diffuse lighting, if enabled
  if (AO_ON && rtLaunch.lights.ao_samples > 0) {
    result *= rtLaunch.lights.ao_direct;
    // Compute AO shade factor first, then combine with colored ambient in a
    // single col multiply.  This keeps col in registers and avoids the spill/
    // reload that would occur if col * p_Ka were added after the outline block.
    // Matches CPU formula: col * (Kd * ao_ambient * shade_ao + Ka)
    float ao_shade = shade_ambient_occlusion(hit_point, N, fogmod * p_opacity);
    result += col * (rtLaunch.lights.ao_ambient * p_Kd * ao_shade + p_Ka);

#if defined(TACHYON_RAYSTATS)
    rtLaunch.frame.raystats1_buffer[idx].z+=rtLaunch.lights.ao_samples; // increment AO shadow ray counter
#endif
  }

  // add edge shading if applicable
  if (OUTLINE_ON && p_outline > 0.0f) {
    float edgefactor = dot(N, ray_direction);
    edgefactor *= edgefactor;
    edgefactor = 1.0f - edgefactor;
    edgefactor = 1.0f - powf(edgefactor, (1.0f - p_outlinewidth) * 32.0f);
    float outlinefactor = __saturatef((1.0f - p_outline) + (edgefactor * p_outline));
    result *= outlinefactor;
  }

  // When AO is enabled, colored ambient was already included in the AO block
  // above.  For non-AO paths (AO_ON == 0) add white ambient here.
  // IMPORTANT: do NOT reference `col` here — keeping col live past the AO
  // block forces the register allocator to spill it during the AO ray loop,
  // causing a severe (up to 8x) performance regression.
  // AO_ON is a compile-time template int: the ternary resolves at compile time
  // to make_float3(0.0f) (no-op, eliminated) for AO_ON=1, or make_float3(p_Ka)
  // (white ambient) for AO_ON=0.
  result += make_float3(AO_ON ? 0.0f : p_Ka);
  result += phongcol;          // add phong highlights

  //
  // spawn reflection rays if necessary
  //
  if (REFLECTION_ON && p_reflectivity > 0.0f) {
    // ray tree attenuation
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * p_reflectivity;
    new_prd.depth = prd.depth + 1;
    new_prd.transcnt = prd.transcnt;

    // shoot a reflection ray
    if (new_prd.importance >= 0.001f && new_prd.depth <= rtLaunch.max_depth) {
      float3 reflray_direction = reflect(ray_direction, N);

      float3 reflray_origin;
#if defined(TACHYON_USE_RAY_STEP)
      reflray_origin = hit_point + TACHYON_RAY_STEP;
#else
      reflray_origin = hit_point;
#endif

      // the values we store the PRD pointer in:
      uint32_t p0, p1;
      packPointer( &new_prd, p0, p1 );
      uint32_t s = getPayloadAAsample(); // use aasample in CH/MISS RNGs

      optixTrace(rtLaunch.traversable,
                 reflray_origin,
                 reflray_direction,
                 0.0f,                          // tmin
                 RT_DEFAULT_MAX,                // tmax
                 0.0f,                          // ray time
                 OptixVisibilityMask( 255 ),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
                 RT_RAY_TYPE_RADIANCE,          // SBT offset
                 RT_RAY_TYPE_COUNT,             // SBT stride
                 RT_RAY_TYPE_RADIANCE,          // missSBTIndex
                 p0, p1,                        // PRD ptr in 2x uint32
                 s);                            // use aasample in CH/MISS RNGs

#if defined(TACHYON_RAYSTATS)
      rtLaunch.frame.raystats2_buffer[idx].w++; // increment refl ray counter
#endif
      result += p_reflectivity * new_prd.result;
    }
  }

  //
  // spawn transmission rays if necessary
  //
  float alpha = p_opacity;
#if 1
  if (CLIP_VIEW_ON && (rtLaunch.clipview_mode == 2))
    sphere_fade_and_clip(hit_point, rtLaunch.cam.pos,
                         rtLaunch.clipview_start, rtLaunch.clipview_end, alpha);
#else
  if (CLIP_VIEW_ON && (rtLaunch.clipview_mode == 2)) {
    // draft implementation of a smooth "fade-out-and-clip sphere"
    float fade_start = 1.00f; // onset of fading
    float fade_end   = 0.20f; // fully transparent
    float camdist = length(hit_point - rtLaunch.cam.pos);

    // XXX we can omit the distance test since alpha modulation value is clamped
    // if (1 || camdist < fade_start) {
      float fade_len = fade_start - fade_end;
      alpha *= __saturatef((camdist - fade_start) / fade_len);
    // }
  }
#endif

#if 1
  // TRANSMISSION_ON: handles transparent surface shading, test is only
  // performed when the geometry has a known-transparent material
  // CLIP_VIEW_ON: forces check of alpha value for all geom as per transparent
  // material, since all geometry may become tranparent with the
  // fade+clip sphere active
  if ((TRANSMISSION_ON || CLIP_VIEW_ON) && alpha < 0.999f ) {
    // Emulate Tachyon/Raster3D's angle-dependent surface opacity if enabled
    if (p_transmode) {
      alpha = 1.0f + cosf(3.1415926f * (1.0f-alpha) * dot(N, ray_direction));
      alpha = alpha*alpha * 0.25f;
    }

    result *= alpha; // scale down current lighting by opacity

    // shoot a transmission ray
    PerRayData_radiance new_prd;
    new_prd.importance = prd.importance * (1.0f - alpha);
    new_prd.alpha = 1.0f;
    new_prd.result = rtLaunch.scene.bg_color;
    new_prd.depth = prd.depth + 1;
    new_prd.transcnt = max(1, prd.transcnt) - 1; // prevent uint wraparound
    if (new_prd.importance >= 0.001f && new_prd.depth <= rtLaunch.max_depth) {
      float3 transray_direction = ray_direction;
      float3 transray_origin;
#if defined(TACHYON_USE_RAY_STEP)
#if defined(TACHYON_TRANS_USE_INCIDENT)
      // step the ray in the incident ray direction
      transray_origin = hit_point + TACHYON_RAY_STEP2;
#else
      // step the ray in the direction opposite the surface normal (going in)
      // rather than out, for transmission rays...
      transray_origin = hit_point - TACHYON_RAY_STEP;
#endif
#else
      transray_origin = hit_point;
#endif

      // the values we store the PRD pointer in:
      uint32_t p0, p1;
      packPointer( &new_prd, p0, p1 );
      uint32_t s = getPayloadAAsample(); // use aasample in CH/MISS RNGs

      optixTrace(rtLaunch.traversable,
                 transray_origin,
                 transray_direction,
                 0.0f,                          // tmin
                 RT_DEFAULT_MAX,                // tmax
                 0.0f,                          // ray time
                 OptixVisibilityMask( 255 ),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT, // Only want CH
                 RT_RAY_TYPE_RADIANCE,          // SBT offset
                 RT_RAY_TYPE_COUNT,             // SBT stride
                 RT_RAY_TYPE_RADIANCE,          // missSBTIndex
                 p0, p1,                        // PRD ptr in 2x uint32
                 s);                            // use aasample in CH/MISS RNGs

#if defined(TACHYON_RAYSTATS)
      rtLaunch.frame.raystats2_buffer[idx].x++; // increment trans ray counter
#endif
    }
    result += (1.0f - alpha) * new_prd.result;
    prd.alpha = alpha + (1.0f - alpha) * new_prd.alpha;
  }
#endif

  // add depth cueing / fog if enabled
  if (FOG_ON && fogmod < 1.0f) {
    result = fog_color(fogmod, result);
  }

  prd.result = result; // pass the color back up the tree
}



//
// OptiX closest hit and anyhit programs for radiance rays
//  
//#define TACHYON_FLATTEN_CLOSESTHIT_DISPATCH 1
//#define TACHYON_MERGED_CLOSESTHIT_DISPATCH 1

// general-purpose any-hit program, with all template options enabled,
// intended for shader debugging and comparison with the original
// Tachyon full_shade() code.
extern "C" __global__ void __closesthit__radiance_general() {
  const GeomSBTHG &sbtHG = *reinterpret_cast<const GeomSBTHG*>(optixGetSbtDataPointer());

  // shading variables that need to be computed/set by primitive-specific code
  float3 shading_normal;
  float3 hit_color;
  float  hit_alpha=1.0f; // tex alpha|cutout transparency, mult w/ mat opacity
  int vertexcolorset=0;


#if defined(TACHYON_MERGED_CLOSESTHIT_DISPATCH)
  unsigned int hit_kind = optixGetHitKind();
  unsigned int hit_prim_type = 0;
#if OPTIX_VERSION >= 70100
  hit_prim_type = optixGetPrimitiveType(hit_kind);
#endif

  // merge kind+type by left shifting OptiX "0x25XX" value range so it can
  // be directly bitwise-ORed with the hit kind from our custom prims, or
  // OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE or OPTIX_HIT_KIND_TRIANGLE_BACK_FACE.
  // If we strip off the low bit from hit kind, and ensure that our 
  // custom prim types start with indices >= 2, we can handle the merged
  // triangle front/back faces as a single case (they are coded as 
  // 0xFF and 0xFE in the OptiX headers)
  unsigned int mergeprimtype = (hit_prim_type << 16) | (0xFE & hit_kind);
  switch (mergeprimtype) {
    case RT_PRM_CONE:
      get_shadevars_cone_array(sbtHG, shading_normal);
      break;

    case RT_PRM_CYLINDER:
      get_shadevars_cylinder_array(sbtHG, shading_normal);
      break;

    case RT_PRM_QUAD:
      get_shadevars_quadmesh(sbtHG, hit_color, shading_normal);
      vertexcolorset=1;
      break;

    case RT_PRM_RING:
      get_shadevars_ring_array(sbtHG, shading_normal);
      break;

    case RT_PRM_SPHERE:
      get_shadevars_sphere_array(sbtHG, shading_normal);
      break;

    case RT_PRM_TRIANGLE:
      get_shadevars_trimesh(sbtHG, hit_color, hit_alpha, shading_normal);
      vertexcolorset=1;
      break;

#if OPTIX_VERSION >= 70400
    case RT_PRM_CATMULLROM:
      get_shadevars_curves_catmullrom(sbtHG, hit_color, shading_normal);
      break;
#endif
#if OPTIX_VERSION >= 70200
    case RT_PRM_LINEAR:
      get_shadevars_curves_linear(sbtHG, hit_color, shading_normal);
      break;
#endif

#if 0
    default:
      printf("Unrecognized merged prim: %08x\n", mergeprimtype);
      break;
#endif
  }   
  {

#else // !defined(TACHYON_MERGED_CLOSESTHIT_DISPATCH)

  // Handle normal and color computations according to primitive type
  unsigned int hit_kind = optixGetHitKind();

#if !defined(TACHYON_FLATTEN_CLOSESTHIT_DISPATCH)
#if OPTIX_VERSION >= 70100
//  OptixPrimitiveType hit_prim_type = optixGetPrimitiveType(hit_kind);
  unsigned int hit_prim_type = optixGetPrimitiveType(hit_kind);

  // XXX It would be more desirable to have a full switch block 
  // for triangles/curves/custom rather than chained if/else 
  if (hit_prim_type == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
    get_shadevars_trimesh(sbtHG, hit_color, hit_alpha, shading_normal);
  } else if (hit_prim_type == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR) {
    get_shadevars_curves_linear(sbtHG, hit_color, shading_normal);
#if OPTIX_VERSION >= 70400
  } else if (hit_prim_type == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM) {
    get_shadevars_curves_catmullrom(sbtHG, hit_color, shading_normal);
#endif
  } else 
#endif
#endif // TACHYON_FLATTEN_CLOSESTHIT_DISPATCH
  {
    // For OPTIX_PRIMITIVE_TYPE_CUSTOM we check the lowest 7 bits of 
    // hit_kind to determine our user-defined primitive type.
    // For peak traversal performance, calculation of surface normals 
    // colors, etc is deferred until CH/AH shading herein. 
    switch (hit_kind) {
#if (OPTIX_VERSION < 70100) || defined(TACHYON_FLATTEN_CLOSESTHIT_DISPATCH)
      // For OptiX 7.0.0 we handle triangle hits here too since
      // it lacked optixGetPrimitiveType() etc...
      // Built-in primitive hit == (hit_kind & 0x80)
      case OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE:  // front-face of triangle hit
      case OPTIX_HIT_KIND_TRIANGLE_BACK_FACE:  // back-face of triangle hit
        get_shadevars_trimesh(sbtHG, hit_color, hit_alpha, shading_normal);
        vertexcolorset=1;
        break;
#endif

      case RT_HIT_CONE: 
        get_shadevars_cone_array(sbtHG, shading_normal);
        break;

      case RT_HIT_CYLINDER: 
        get_shadevars_cylinder_array(sbtHG, shading_normal);
        break;

      case RT_HIT_QUAD:
        get_shadevars_quadmesh(sbtHG, hit_color, shading_normal);
        vertexcolorset=1;
        break;

      case RT_HIT_RING: 
        get_shadevars_ring_array(sbtHG, shading_normal);
        break;

      case RT_HIT_SPHERE: 
        get_shadevars_sphere_array(sbtHG, shading_normal);
        break;

#if defined(TACHYON_FLATTEN_CLOSESTHIT_DISPATCH)
#if OPTIX_VERSION >= 70100
      default:
        {
          OptixPrimitiveType hit_prim_type = optixGetPrimitiveType(hit_kind);
          // At this point we know it must be a curve or one of the
          // other built-in types
          if (hit_prim_type == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR) {
            get_shadevars_curves_linear(sbtHG, hit_color, shading_normal);
#if OPTIX_VERSION >= 70400
          } else if (hit_prim_type == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM) {
            get_shadevars_curves_catmullrom(sbtHG, hit_color, shading_normal);
#endif
          }
        }
        break;
#endif
#endif

    }
#endif // !defined(TACHYON_MERGED_CLOSESTHIT_DISPATCH)

    // Assign either per-primitive or uniform color
    if (!vertexcolorset) {
      if (sbtHG.prim_color != nullptr) {
        const int primID = optixGetPrimitiveIndex();
        hit_color = sbtHG.prim_color[primID];
      } else {
        hit_color = sbtHG.uniform_color;
      }
    }
  }

  // VR CLIP_VIEW and HEADLIGHT modes are locked out for normal renderings
  // for the time being, so they don't harm performance measurements for
  // the more typical non-VR use cases.
  int matidx = sbtHG.materialindex; // common for all geometry
  const auto &mat = rtLaunch.materials[matidx];
  shader_template<0, 0, 1, 1, 1, 1, 1, 1>(hit_color, shading_normal,
                                          mat.ambient, mat.diffuse, 
                                          mat.specular, mat.shininess,
                                          mat.reflectivity, 
                                          mat.opacity * hit_alpha,
                                          mat.outline, mat.outlinewidth,
                                          mat.transmode);
}




