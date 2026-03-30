/*
 * TachyonOptiX.h - OptiX host-side RT engine APIs and data structures
 * 
 * (C) Copyright 2013-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: TachyonOptiX.h,v 1.57 2022/04/19 02:54:24 johns Exp $
 *
 */

/**
 *  \file TachyonOptiX.h
 *  \brief Tachyon ray tracing host side routines and internal APIs that 
 *         provide the core ray OptiX-based RTX-accelerated tracing engine.
 *         The major responsibilities of the core engine are to manage
 *         runtime RT pipeline construction and JIT-linked shaders 
 *         to build complete ray tracing pipelines, management of 
 *         RT engine state, and managing associated GPU hardware.
 *         Written for NVIDIA OptiX 7 and later.
 *
 *         The declarations and prototypes needed to drive
 *         the raytracer.  Third party driver code should only use the 
 *         functions in this header file to interface with the rendering engine.
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

#ifndef TACHYONOPTIX_H
#define TACHYONOPTIX_H

#if 0
/// Enable fully conservative fall-back code paths that force
/// cudaStreamSynchronize() after every async CUDA API for debugging
#define TACHYON_CUMEMBUF_FORCE_SYNCHRONOUS 1

/// Enable fully conservative fall-back code paths that force
/// full memory frees and size zeroing even for the persistent memory APIs,
/// to support debugging. 
#define TACHYON_CUMEMBUF_FORCE_FREE        1


/// Enable forced host-side allocation of pinned memory buffers to fully
/// enable asynchronous APIs and almost 2x the DMA transfer performance.
#define TACHYON_USEPINNEDMEMORY            1

#endif

#if defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)
// OptiX headers require NOMINMAX be defined for Windows builds
#define NOMINMAX 1
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

// When compiling using the classic Tachyon-internal timer API,
// we use the non-renamed rt_xxx API struct and function names.
#if defined(TACHYONINTERNAL)
#define wkf_timerhandle      rt_timerhandle
#define wkf_timer_create     rt_timer_create
#define wkf_timer_start      rt_timer_start
#define wkf_timer_stop       rt_timer_stop
#define wkf_timer_time       rt_timer_time
#define wkf_timer_timenow    rt_timer_timenow
#define wkf_timer_destroy    rt_timer_destroy
#include "util.h"
#else
#include "WKFUtils.h"
#endif
#include "TachyonOptiXShaders.h"


#if defined(TACHYON_USEPINNEDMEMORY)
//
// Include NVIDIA Thrust lib for special pinned allocator support
//
#include <cuda_runtime.h>

// THRUST_VERSION >= 100904
// Thrust 1.9.4 adds new thrust::universal_host_pinned_memory_resource
// for getting pinned memory, but this isn't a drop-in replacement for
// what we've been doing with the experimental pinned allocator.
// For now we continue using the existing interface.
#include <thrust/system/cuda/experimental/pinned_allocator.h>

/// Macro to simplify conditional compilation supporting host pinned memory
#define THRUSTPINALLOC thrust::system::cuda::experimental::pinned_allocator
/// Macro to simplify conditional compilation supporting host pinned memory
#define PINALLOCS(memtype) ,thrust::system::cuda::experimental::pinned_allocator<memtype>

#else
/// use the standard allocator if pinned memory isn't enabled at compile-time
#define PINALLOCS(memtype)
#endif


// Tachyon profiling colors
#define RTPROF_GENERAL  PROFILE_BLUE    ///< trace color for general operations
#define RTPROF_ACCEL    PROFILE_RED     ///< trace color for RT AS builds
#define RTPROF_SBT      PROFILE_TEAL    ///< trace color for SBT construction
#define RTPROF_RENDER   PROFILE_ORANGE  ///< trace color for overall rendering
#define RTPROF_RENDERRT PROFILE_YELLOW  ///< trace color for specifically for RT
#define RTPROF_TRANSFER PROFILE_RED     ///< trace color for host-GPU DMA
#define RTPROF_GEOM     PROFILE_PURPLE  ///< trace color for geometry processing

struct ConeArray {
  std::vector<float3 PINALLOCS(float3)> base;
  std::vector<float3 PINALLOCS(float3)> apex;
  std::vector<float  PINALLOCS(float)>  baserad;
  std::vector<float  PINALLOCS(float)>  apexrad;
  std::vector<float3 PINALLOCS(float3)> primcolors3f;
  float3 uniform_color;
  int materialindex;

  void addCone(const float3 &cbase, const float3 &capex, 
               const float cbaserad, const float capexrad) {
    base.push_back(cbase);
    apex.push_back(capex);
    baserad.push_back(cbaserad);
    apexrad.push_back(capexrad);
  }
};


struct CylinderArray {
  std::vector<float3 PINALLOCS(float3)> start;
  std::vector<float3 PINALLOCS(float3)> end;
  std::vector<float  PINALLOCS(float)>  radius;
  std::vector<float3 PINALLOCS(float3)> primcolors3f;
  float3 uniform_color;
  int materialindex;

  void addCylinder(const float3 &cstart, const float3 &cend, const float crad) {
    start.push_back(cstart);
    end.push_back(cend);
    radius.push_back(crad);
  }
};


struct QuadMesh {
  std::vector<float3 PINALLOCS(float3)> vertices;
  std::vector<int4   PINALLOCS(int4)>   indices;
  std::vector<float3 PINALLOCS(float3)> normals;
  std::vector<uint4  PINALLOCS(uint4)>  packednormals;
  std::vector<float3 PINALLOCS(float3)> vertcolors3f;
  std::vector<uchar4 PINALLOCS(uchar4)> vertcolors4u;
  std::vector<float3 PINALLOCS(float3)> primcolors3f;
  float3 uniform_color;
  int materialindex;
};


struct RingArray {
  std::vector<float3 PINALLOCS(float3)> center;
  std::vector<float3 PINALLOCS(float3)> normal;
  std::vector<float  PINALLOCS(float)>  inrad;
  std::vector<float  PINALLOCS(float)>  outrad;
  std::vector<float3 PINALLOCS(float3)> primcolors3f;
  float3 uniform_color;
  int materialindex;

  void addRing(const float3 &ricenter, const float3 &rinorm,
               const float riinrad, const float rioutrad) {
    center.push_back(ricenter);
    normal.push_back(rinorm);
    inrad.push_back(riinrad);
    outrad.push_back(rioutrad);
  }
};


struct SphereArray {
  std::vector<float3 PINALLOCS(float3)> center;
  std::vector<float  PINALLOCS(float)> radius;
  std::vector<float3 PINALLOCS(float3)> primcolors3f;
  float3 uniform_color;
  int materialindex;

  void addSphere(const float3 &spcenter, const float &spradius) {
    center.push_back(spcenter);
    radius.push_back(spradius);
  }
};


struct CurveArray {
  std::vector<float3 PINALLOCS(float3)> vertices;
  std::vector<float  PINALLOCS(float)> vertradii;
  std::vector<int    PINALLOCS(int)> segindices;
  // std::vector<float3 PINALLOCS(float3)> normals; // XXX not implemented in Optix <= 7.4
  std::vector<float3 PINALLOCS(float3)> primcolors3f;
  float3 uniform_color;
  int materialindex;

  void addCurve(const float *verts, const int numverts,
                const float *rads,  const int numradii,
                const int *sidxs, const int numindices) {
    for (int i=0; i<numverts; i++) {
      vertices.push_back(make_float3(verts[3*i    ],
                                     verts[3*i + 1],
                                     verts[3*i + 2]));
    }
    for (int i=0; i<numradii; i++) {
      vertradii.push_back(rads[i]);
    }
    for (int i=0; i<numindices; i++) {
      segindices.push_back(sidxs[i]);
    }
  }
};


struct TriangleMesh {
  std::vector<float3 PINALLOCS(float3) > vertices;
  std::vector<int3 PINALLOCS(int3) >   indices;
  std::vector<float3 PINALLOCS(float3) > normals;
  std::vector<uint4 PINALLOCS(uint4) >  packednormals;
  std::vector<float3 PINALLOCS(float3) > vertcolors3f;
  std::vector<uchar4 PINALLOCS(uchar4) > vertcolors4u;
  std::vector<float3 PINALLOCS(float3) > primcolors3f;
  std::vector<float2 PINALLOCS(float2) > tex2d;
  std::vector<float3 PINALLOCS(float3) > tex3d;
  float3 uniform_color;
  int materialindex;

  void addCube(const float3 &center, const float3 &s /* size */) {
    int firstVertexID = (int)vertices.size();
    vertices.push_back(center + make_float3(s.x*0.f, s.y*0.f, s.z*0.f));
    vertices.push_back(center + make_float3(s.x*1.f, s.y*0.f, s.z*0.f));
    vertices.push_back(center + make_float3(s.x*0.f, s.y*1.f, s.z*0.f));
    vertices.push_back(center + make_float3(s.x*1.f, s.y*1.f, s.z*0.f));
    vertices.push_back(center + make_float3(s.x*0.f, s.y*0.f, s.z*1.f));
    vertices.push_back(center + make_float3(s.x*1.f, s.y*0.f, s.z*1.f));
    vertices.push_back(center + make_float3(s.x*0.f, s.y*1.f, s.z*1.f));
    vertices.push_back(center + make_float3(s.x*1.f, s.y*1.f, s.z*1.f));
    int ind[] = {0,1,3, 2,3,0,
                 5,7,6, 5,6,4,
                 0,4,5, 0,5,1,
                 2,3,7, 2,7,6,
                 1,5,7, 1,7,3,
                 4,0,2, 4,2,6
                 };
    for (int i=0; i<12; i++)
      indices.push_back(make_int3(ind[3*i+0] + firstVertexID,
                                  ind[3*i+1] + firstVertexID,
                                  ind[3*i+2] + firstVertexID));
  }
};


/// Several OptiX APIs make use of CUDA driver API pointer types
/// (CUdevicepointer) so it becomes worthwhile to manage these 
/// in a templated class supporting easy memory management for
/// vectors of special template types, and simple copies to and
/// from the associated CUDA device.
struct CUMemBuf {
  size_t sz     { 0 };        ///< device memory buffer usage size in bytes
  size_t dmemsz { 0 };        ///< device memory buffer allocated size in bytes
  void  *d_ptr  { nullptr };  ///< pointer to device memory buffer

  inline void * dptr() const { 
    // if we've kept GPU memory allocations persistent but the current 
    // stored size is zero, we return a NULL device pointer if a caller
    // asks for it, to indicate that there's nothing here.  This emulates
    // the previous behavior of the non-persistent allocation scheme.
    return (sz > 0) ? d_ptr : 0;  
  }

  inline CUdeviceptr cu_dptr() const { 
    // if we've kept GPU memory allocations persistent but the current 
    // stored size is zero, we return a NULL device pointer if a caller
    // asks for it, to indicate that there's nothing here.  This emulates
    // the previous behavior of the non-persistent allocation scheme.
    return CUdeviceptr((sz > 0) ? d_ptr : 0); 
  }

  /// query current buffer size in bytes
  size_t get_size(void) {
    return sz;
  }    

  /// (re)allocate buffer of requested size
  void set_size(size_t newsize) {
    // don't reallocate unless it is strictly necessary since 
    // CUDA memory management operations are very costly
    if (newsize > dmemsz) {
      if (d_ptr) 
        this->free();

      sz = newsize;
      dmemsz = newsize;
      cudaMalloc((void**) &d_ptr, dmemsz);
    } else {
      // use the existing CUDA memory allocation, but change the stored size
      sz = newsize;
    }
  }

  /// (re)allocate buffer of requested size, asynchronously
  void set_size(size_t newsize, cudaStream_t stream) {
#if !defined(TACHYON_CUMEMBUF_FORCE_FREE)
    // don't reallocate unless it is strictly necessary since 
    // CUDA memory management operations are very costly
    if (newsize != dmemsz) {
      if (d_ptr) 
        this->free(stream);
#if defined(TACHYON_CUMEMBUF_FORCE_SYNCHRONOUS)
      cudaStreamSynchronize(stream);
#endif

      sz = newsize;
      dmemsz = newsize;
#if CUDA_VERSION >= 11200
      cudaMallocAsync((void**) &d_ptr, dmemsz, stream);
#else
      cudaMalloc((void**) &d_ptr, dmemsz);
#endif
#if defined(TACHYON_CUMEMBUF_FORCE_SYNCHRONOUS)
      cudaStreamSynchronize(stream);
#endif
    } else {
      // use the existing CUDA memory allocation, but change the stored size
      sz = newsize;
    }
#else
    // don't reallocate unless it is strictly necessary since 
    // CUDA memory management operations are very costly
    if (newsize != sz) {
      if (d_ptr) 
        this->free(stream);
#if defined(TACHYON_CUMEMBUF_FORCE_SYNCHRONOUS)
      cudaStreamSynchronize(stream);
#endif

      sz = newsize;
      dmemsz = newsize;
#if CUDA_VERSION >= 11200
      cudaMallocAsync((void**) &d_ptr, dmemsz, stream);
#else
      cudaMalloc((void**) &d_ptr, dmemsz);
#endif
#if defined(TACHYON_CUMEMBUF_FORCE_SYNCHRONOUS)
      cudaStreamSynchronize(stream);
#endif
    }
#endif

  }


  /// clear "used" size to zero, but keep existing device allocation
  void clear_persist_allocation(void) {
#if defined(TACHYON_CUMEMBUF_FORCE_FREE)
    free();
#endif
    sz = 0;
  }

 
  /// free allocated memory
  void free() {
    cudaFree(d_ptr);
    d_ptr = nullptr;
    sz = 0;
    dmemsz = 0;
  }


  /// free allocated memory asynchronously
  void free(cudaStream_t stream) {
#if CUDA_VERSION >= 11200
    cudaFreeAsync(d_ptr, stream);
#else
    cudaFree(d_ptr);
#endif
#if defined(TACHYON_CUMEMBUF_FORCE_SYNCHRONOUS)
    cudaStreamSynchronize(stream);
#endif
    d_ptr = nullptr;
    sz = 0;
    dmemsz = 0;
  }


  //
  // synchronous copies that also allocate/resize the buffer 
  //
  // XXX Should seriously look at the viability of using 'GDRCopy' rather
  //     than the standard CUDA APIs for small payloads where latency 
  //     dominates and/or when they are used for repetitive minor updates
  //     to GPU-side state.  GDRCopy depends on some kernel driver components,
  //     but they are likely already installed on HPC systems w/ InfiniBand:
  //       https://github.com/NVIDIA/gdrcopy
  //    

  /// Combination of a buffer resize with synchronous upload 
  /// to GPU device memory.
  template<typename T>
  void resize_upload(const std::vector<T> &vecT) {
    set_size(vecT.size()*sizeof(T));
    upload((const T*) vecT.data(), vecT.size());
  }

  /// Combination of a buffer resize with synchronous upload 
  /// to GPU device memory.
  template<typename T>
  void resize_upload(const T *t, size_t cnt) {
    set_size(cnt*sizeof(T));
    cudaMemcpy(d_ptr, (void *)t, cnt*sizeof(T), cudaMemcpyHostToDevice);
  }
  

  //
  // async versions of resize_upload() w/ stream parameter 
  // 

  /// Combination of a buffer resize with asynchronous upload to GPU
  /// device memory using the caller-provided CUDA stream to enforce
  /// ordering requirements.
  template<typename T>
  void resize_upload(const std::vector<T> &vecT, cudaStream_t stream) {
    set_size(vecT.size()*sizeof(T), stream);
    upload((const T*) vecT.data(), vecT.size(), stream);
  }

  /// Combination of a buffer resize with asynchronous upload to GPU
  /// device memory using the caller-provided CUDA stream to enforce
  /// ordering requirements.
  template<typename T>
  void resize_upload(const T *t, size_t cnt, cudaStream_t stream) {
    set_size(cnt*sizeof(T), stream);
    cudaMemcpyAsync(d_ptr, (void *)t, cnt*sizeof(T), 
                    cudaMemcpyHostToDevice, stream);
#if defined(TACHYON_CUMEMBUF_FORCE_SYNCHRONOUS)
    cudaStreamSynchronize(stream);
#endif
  }


#if defined(TACHYON_USEPINNEDMEMORY)
  //
  // Pinned memory allocator variants of both sync and async resize_upload().
  // These must exist to support std::vector instantiations based on the 
  // Thrust pinned allocator.
  // 

  /// host pinned memory allocator variation of resize_upload()
  /// Combination of a buffer resize with asynchronous upload to GPU
  /// device memory using the caller-provided CUDA stream to enforce
  /// ordering requirements.
  template<typename T>
  void resize_upload(const std::vector<T, THRUSTPINALLOC<T>> &vecT) {
    set_size(vecT.size()*sizeof(T));
    upload((const T*) vecT.data(), vecT.size());
  }
  
  /// Host pinned memory allocator variation of resize_upload().
  /// Combination of a buffer resize with asynchronous upload to GPU
  /// device memory using the caller-provided CUDA stream to enforce
  /// ordering requirements.
  template<typename T>
  void resize_upload(const std::vector<T, THRUSTPINALLOC<T>> &vecT, cudaStream_t stream) {
    set_size(vecT.size()*sizeof(T), stream);
    upload((const T*) vecT.data(), vecT.size(), stream);
  }
#endif  


  // 
  // synchronous DMA copies
  //
  // XXX Should seriously look at the viability of using 'GDRCopy' rather
  //     than the standard CUDA APIs for small payloads where latency 
  //     dominates and/or when they are used for repetitive minor updates
  //     to GPU-side state.  GDRCopy depends on some kernel driver components,
  //     but they are likely already installed on HPC systems w/ InfiniBand:
  //       https://github.com/NVIDIA/gdrcopy
  //    

  /// Synchronous upload to GPU device memory.
  template<typename T>
  void upload(const T *t, size_t cnt) {
    cudaMemcpy(d_ptr, (void *)t, cnt*sizeof(T), cudaMemcpyHostToDevice);
  }
  
  /// Synchronous download from GPU device memory.
  template<typename T>
  void download(T *t, size_t cnt) {
    cudaMemcpy((void *)t, d_ptr, cnt*sizeof(T), cudaMemcpyDeviceToHost);
  }


  //
  // asynchronous DMA copies
  //

  /// Asynchronous upload to GPU device memory, using the caller-provided
  /// CUDA stream to enforce ordering requirements.
  template<typename T>
  void upload(const T *t, size_t cnt, cudaStream_t stream) {
    cudaMemcpyAsync(d_ptr, (void *)t, 
                    cnt*sizeof(T), cudaMemcpyHostToDevice, stream);
#if defined(TACHYON_CUMEMBUF_FORCE_SYNCHRONOUS)
    cudaStreamSynchronize(stream);
#endif
  }

  /// Asynchronous download from GPU device memory, using the caller-provided
  /// CUDA stream to enforce ordering requirements.
  template<typename T>
  void download_async(T *t, size_t cnt, cudaStream_t stream) {
    cudaMemcpyAsync((void *)t, d_ptr, 
                    cnt*sizeof(T), cudaMemcpyDeviceToHost, stream);
#if defined(TACHYON_CUMEMBUF_FORCE_SYNCHRONOUS)
    cudaStreamSynchronize(stream);
#endif
  }
};


struct TachyonInstanceGroup {
  // 
  // primitive array buffers
  // 
  std::vector<ConeArray>   conearrays;
  std::vector<CurveArray> curvearrays;
  std::vector<CylinderArray> cyarrays;
  std::vector<RingArray>     riarrays;
  std::vector<SphereArray>   sparrays;
  std::vector<TriangleMesh> trimeshes;

  // GASes get rebuilt upon any group geometry changes
  std::vector<HGRecordGroup> sbtHGRecGroups; ///< all GeomSBTHG recs
  CUMemBuf custprimsGASBuffer;               ///< final, compacted GAS
#if OPTIX_VERSION >= 70100
  CUMemBuf curvesGASBuffer;                  ///< final, compacted GAS
#endif
  CUMemBuf trimeshesGASBuffer;               ///< final, compacted GAS

  std::vector<float *> transforms;
};
 

class TachyonOptiX {
public: 
  enum ViewClipMode {
    RT_VIEWCLIP_NONE=0,                    ///< no frustum clipping
    RT_VIEWCLIP_PLANE=1,                   ///< cam/frustum front clipping plane
    RT_VIEWCLIP_SPHERE=2                   ///< cam/frustum clipping sphere
  };

  enum HeadlightMode { 
    RT_HEADLIGHT_OFF=0,                    ///< no VR headlight
    RT_HEADLIGHT_ON=1                      ///< VR headlight at cam center
  };

  enum FogMode  { 
    RT_FOG_NONE=0,                         ///< No fog
    RT_FOG_LINEAR=1,                       ///< Linear fog w/ Z-depth
    RT_FOG_EXP=2,                          ///< Exp fog w/ Z-depth
    RT_FOG_EXP2=3                          ///< Exp^2 fog w/ Z-depth
    // XXX should explicitly define radial/omnidirection fog types also
  };

  enum CameraType { 
    RT_PERSPECTIVE=0,                      ///< conventional perspective
    RT_ORTHOGRAPHIC=1,                     ///< conventional orthographic
    RT_CUBEMAP=2,                          ///< omnidirectional cubemap
    RT_DOME_MASTER=3,                      ///< planetarium dome master
    RT_EQUIRECTANGULAR=4,                  ///< omnidirectional lat/long
    RT_OCTAHEDRAL=5,                       ///< omnidirectional octahedral
    RT_OCULUS_RIFT                         ///< VR HMD
  };

  enum Verbosity { 
    RT_VERB_MIN=0,                         ///< No console output
    RT_VERB_TIMING=1,                      ///< Output timing/perf data only
    RT_VERB_DEBUG=2                        ///< Output fully verbose debug info
  };

  enum BGMode { 
    RT_BACKGROUND_TEXTURE_SOLID=0,
    RT_BACKGROUND_TEXTURE_SKY_SPHERE=1,
    RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE=2 
  };

private:
  int context_created;                     ///< flag when context is valid
  CUcontext cuda_ctx;                      ///< CUDA driver context for OptiX
  CUstream stream;                         ///< CUDA driver stream for OptiX
  OptixDeviceContext optix_ctx;            ///< OptiX context 
  OptixResult lasterr;                     ///< Last OptiX error code if any
  Verbosity verbose;                       ///< console perf/debugging output

  char shaderpath[8192];                   ///< path to OptiX shader PTX file

  wkf_timerhandle rt_timer;                ///< general purpose timer
  double time_ctx_create;                  ///< time taken to create ctx
  double time_ctx_setup;                   ///< time taken to setup/init ctx
  double time_ctx_validate;                ///< time for ctx compile+validate
  double time_ctx_AS_build;                ///< time for AS build
  double time_ctx_destroy_scene;           ///< time to destroy existing scene
  double time_ray_tracing;                 ///< time to trace the rays...
  double time_image_io;                    ///< time to write image(s) to disk

  //
  // OptiX launch parameters
  //
  tachyonLaunchParams rtLaunch;            ///< host-side launch params
  CUMemBuf launchParamsBuffer;             ///< device-side launch params buffer

  //
  // OptiX framebuffers and associated state
  // 
  int width;                               ///< image width in pixels
  int height;                              ///< image height in pixels
  int colorspace;                          ///< output image colorspace

  CUMemBuf framebuffer;                    ///< device-side framebuffer
  CUMemBuf accumulation_buffer;            ///< device-side accumulation buffer

#if defined(TACHYON_OPTIXDENOISER)
  OptixDenoiser        denoiser_ctx;       ///< denoiser handle
  OptixDenoiserOptions denoiser_options;   ///< denoiser options
  OptixDenoiserSizes   denoiser_sizes;     ///< denoiser required buf sizes

  // extra buffers required for denoising
  CUMemBuf denoiser_colorbuffer;           ///< 32-bit FP sRGB, Gamma 2.2
  CUMemBuf denoiser_denoisedbuffer;        ///< 32-bit FP sRGB, Gamma 2.2
  CUMemBuf denoiser_scratch;               ///< denoising scratch buffer
  CUMemBuf denoiser_state;                 ///< denoising state buffer
#endif

#if defined(TACHYON_RAYSTATS)
  CUMemBuf raystats1_buffer;               ///< device-side raystats buffer
  CUMemBuf raystats2_buffer;               ///< device-side raystats buffer
#endif


  //
  // OptiX pipeline and shader compilations
  //

  // the PTX module that contains all device programs
  char * rt_ptx_code_string = {};
  OptixPipelineCompileOptions pipeCompOpts = {};
  OptixModule general_module;
  OptixModule curve_module;

  std::vector<OptixProgramGroup> exceptionPGs;
  std::vector<OptixProgramGroup> raygenPGs;
  std::vector<OptixProgramGroup> missPGs;
  std::vector<OptixProgramGroup> custprimPGs;
  std::vector<OptixProgramGroup> curvePGs;
  std::vector<OptixProgramGroup> trimeshPGs;

  // SBT-associated GPU data structures
  // SBT must be entirely rebuilt upon any change to the rendering pipeline
  CUMemBuf exceptionRecordsBuffer;
  CUMemBuf raygenRecordsBuffer;
  CUMemBuf missRecordsBuffer;

  std::vector<TachyonInstanceGroup> sceneinstancegroups;

  int sync_hitgroupRecordGroups; ///< sync host+GPU copies of HG recs
  std::vector<HGRecordGroup> hitgroupRecordGroups; // all GeomSBTHG recs

  CUMemBuf hitgroupRecordsBuffer;
  OptixShaderBindingTable sbt = {};

  // OptiX RT pipeline produced from all of raygen/miss/hitgroup PGs
  OptixPipeline pipe;

  // generic temporary buffer used by AS builders
  // This temp buffer only grows and will not shrink unless the entire context
  // is destroyed, or the minimize_memory_use() API is called. 
  CUMemBuf ASTempBuffer;                   ///< temporary AS build buffer
  CUMemBuf compactedSizeBuffer;            ///< temp AS compacted size info 

  // GASes get rebuilt upon any scene geometry changes
  CUMemBuf custprimsGASBuffer;             ///< final, compacted GAS
  CUMemBuf curvesGASBuffer;                ///< final, compacted GAS
  CUMemBuf trimeshesGASBuffer;             ///< final, compacted GAS

  // IAS to combine all triangle, curve, and custom primitive GASes together
  // The scene-wide IAS buffer only grows and will not shrink unless 
  // the entire context is destroyed, or the minimize_memory_use() API 
  // is called. 
  CUMemBuf IASBuffer;                      ///< final IAS


  //  
  // primitive array buffers
  //  
  std::vector<CUMemBuf>      coneAabbBuffers;
  std::vector<CUMemBuf>      coneBaseBuffers;
  std::vector<CUMemBuf>      coneApexBuffers;
  std::vector<CUMemBuf>      coneBaseRadBuffers;
  std::vector<CUMemBuf>      coneApexRadBuffers;
  std::vector<CUMemBuf>      conePrimColorBuffers;
  std::vector<ConeArray>     conearrays;

  std::vector<CUMemBuf>      curveVertBuffers;
  std::vector<CUMemBuf>      curveVertRadBuffers;
  std::vector<CUMemBuf>      curveSegIdxBuffers;
  std::vector<CUMemBuf>      curvePrimColorBuffers;
  std::vector<CurveArray>    curvearrays;

  std::vector<CUMemBuf>      cyAabbBuffers;
  std::vector<CUMemBuf>      cyStartBuffers;
  std::vector<CUMemBuf>      cyEndBuffers;
  std::vector<CUMemBuf>      cyRadiusBuffers;
  std::vector<CUMemBuf>      cyPrimColorBuffers;
  std::vector<CylinderArray> cyarrays;

  std::vector<CUMemBuf>      quadMeshAabbBuffers;
  std::vector<CUMemBuf>      quadMeshVertBuffers;
  std::vector<CUMemBuf>      quadMeshIdxBuffers;
  std::vector<CUMemBuf>      quadMeshVertNormalBuffers;
  std::vector<CUMemBuf>      quadMeshVertPackedNormalBuffers;
  std::vector<CUMemBuf>      quadMeshVertColor3fBuffers;
  std::vector<CUMemBuf>      quadMeshVertColor4uBuffers;
  std::vector<CUMemBuf>      quadMeshPrimColorBuffers;
  std::vector<QuadMesh>      quadmeshes;
  
  std::vector<CUMemBuf>      riAabbBuffers;
  std::vector<CUMemBuf>      riCenterBuffers;
  std::vector<CUMemBuf>      riNormalBuffers;
  std::vector<CUMemBuf>      riInRadiusBuffers;
  std::vector<CUMemBuf>      riOutRadiusBuffers;
  std::vector<CUMemBuf>      riPrimColorBuffers;
  std::vector<RingArray>     riarrays;

  std::vector<CUMemBuf>      spAabbBuffers;
  std::vector<CUMemBuf>      spPosRadiusBuffers;
  std::vector<CUMemBuf>      spPrimColorBuffers;
  std::vector<SphereArray>   sparrays;
  
  std::vector<CUMemBuf>      triMeshVertBuffers;
  std::vector<CUMemBuf>      triMeshIdxBuffers;
  std::vector<CUMemBuf>      triMeshVertNormalBuffers;
  std::vector<CUMemBuf>      triMeshVertPackedNormalBuffers;
  std::vector<CUMemBuf>      triMeshVertColor3fBuffers;
  std::vector<CUMemBuf>      triMeshVertColor4uBuffers;
  std::vector<CUMemBuf>      triMeshPrimColorBuffers;
  std::vector<CUMemBuf>      triMeshTex2dBuffers;
  std::vector<CUMemBuf>      triMeshTex3dBuffers;
  std::vector<TriangleMesh>  trimeshes;
  

  //
  // OptiX shader state variables and the like
  //
  unsigned int scene_max_depth;            ///< max ray recursion depth
  int scene_max_trans;                     ///< max transmission ray depth

  float scene_epsilon;                     ///< scene-wide epsilon value

  int ext_aa_loops;                        ///< Multi-pass AA iterations
  int aa_samples;                          ///< AA samples per pixel

  int denoiser_enabled;                    ///< AI denoising enable flag
  int shadows_enabled;                     ///< shadow enable flag

  int ao_samples;                          ///< AO samples per pixel
  float ao_ambient;                        ///< AO ambient lighting scalefactor
  float ao_direct;                         ///< AO direct lighting scalefactor
  float ao_maxdist;                        ///< AO maximum occlusion distance

  int headlight_mode;                      ///< VR HMD headlight

  // clipping plane/sphere parameters
  int clipview_mode;                       ///< VR fade+clipping sphere/plane
  float clipview_start;                    ///< VR fade+clipping sphere/plane
  float clipview_end;                      ///< VR fade+clipping sphere/plane

  float cam_pos[3];                        ///< camera position
  float cam_U[3];                          ///< camera ONB "right" direction
  float cam_V[3];                          ///< camera ONB "up" direction
  float cam_W[3];                          ///< camera ONB "view" direction
  float cam_zoom;                          ///< camera zoom factor

  int cam_dof_enabled;                     ///< DoF enable/disable flag
  float cam_dof_focal_dist;                ///< DoF focal distance
  float cam_dof_fnumber;                   ///< DoF f/stop number

  int cam_stereo_enabled;                  ///< stereo enable/disable flag
  float cam_stereo_eyesep;                 ///< stereo eye separation
  float cam_stereo_convergence_dist;       ///< stereo convergence distance

  CameraType camera_type;                  ///< camera type

  // background color and/or gradient parameters
  BGMode scene_background_mode;            ///< which miss program to use...
  float scene_bg_color[3];                 ///< background color
  float scene_bg_grad_top[3];              ///< background gradient top color
  float scene_bg_grad_bot[3];              ///< background gradient bottom color
  float scene_bg_grad_updir[3];            ///< background gradient up vector
  float scene_bg_grad_topval;              ///< background gradient top value
  float scene_bg_grad_botval;              ///< background gradient bot value
  float scene_bg_grad_invrange;            ///< background gradient rcp range

  // clipping plane/sphere parameters
  int clip_mode;                           ///< clip mode
  float clip_start;                        ///< clip start (Z or radial dist)
  float clip_end;                          ///< clip end (Z or radial dist)

  // fog / depth cueing parameters
  int fog_mode;                            ///< fog mode
  float fog_start;                         ///< fog start
  float fog_end;                           ///< fog end
  float fog_density;                       ///< fog density

  std::vector<rt_texture>  texturecache;   ///< cache of textures
  std::vector<rt_material> materialcache;  ///< cache of materials

  int regen_optix_materials;               ///< flag to force re-upload to GPU
  CUMemBuf materialsBuffer;                ///< device-side materials buffer

  std::vector<rt_directional_light> directional_lights; ///< list of directional lights
  std::vector<rt_positional_light> positional_lights;   ///< list of positional lights
  int regen_optix_lights;                  ///< flag to force re-upload to GPU
  CUMemBuf directionalLightsBuffer;        ///< device-side dir light buffer
  CUMemBuf positionalLightsBuffer;         ///< device-side pos light buffer


  //
  // Scene and geometric primitive counters
  //

  // state variables to hold scene geometry
  int scene_created;

  // cylinder array primitive
  long cylinder_array_cnt;                 ///< number of cylinder in scene

  // color-per-cylinder array primitive
  long cylinder_array_color_cnt;           ///< number of cylinders in scene


  // color-per-ring array primitive
  long ring_array_color_cnt;               ///< number of rings in scene


  // sphere array primitive
  long sphere_array_cnt;                   ///< number of spheres in scene

  // color-per-sphere array primitive
  long sphere_array_color_cnt;             ///< number of spheres in scene


  // triangle mesh primitives of various types
  long tricolor_cnt;                       ///< number of triangles in scene
  long trimesh_c4u_n3b_v3f_cnt;            ///< number of triangles in scene
  long trimesh_n3b_v3f_cnt;                ///< number of triangles in scene
  long trimesh_n3f_v3f_cnt;                ///< number of triangles in scene
  long trimesh_v3f_cnt;                    ///< number of triangles in scene


  //
  // Internal methods
  //

  void destroy_context(void);

  void check_verbose_env();                 ///< check env vars for verbose out
  void render_compile_and_validate(void);

  /// sub-init routines to compile, link, the complete RT pipeline
  char *internal_compiled_ptx_src(void); 
  int read_ptx_src(const char *ptxfilename, char **ptxstring);
  
  void context_create_denoiser(void);
  void context_destroy_denoiser(void);
  void denoiser_resize_update(void);
  void denoiser_launch(void);
 
  void context_create_exception_pgms(void);
  void context_destroy_exception_pgms(void);

  void context_create_raygen_pgms(void);
  void context_destroy_raygen_pgms(void);

  void context_create_miss_pgms(void);
  void context_destroy_miss_pgms(void);

  void context_create_curve_hitgroup_pgms(void);
  void context_destroy_curve_hitgroup_pgms(void);

  void context_create_hwtri_hitgroup_pgms(void);
  void context_destroy_hwtri_hitgroup_pgms(void);

  void context_create_intersection_pgms(void);
  void context_destroy_intersection_pgms(void);

  void context_create_module(void);
  void context_destroy_module(void);

  int regen_optix_pipeline;
  void context_create_pipeline(void);
  void context_destroy_pipeline(void);

  /// Shader binding table management routines
  int regen_optix_sbt;
  void SBT_create_programs(void);
  void SBT_create_hitgroups(void);
  void SBT_clear(void);   ///< clear SBT, but retain persistent storage
  void SBT_destroy(void); ///< completely deallocate SBT storage

  /// update per-geom hitgroup flags for material or other cached values
  void SBT_update_hitgroup_geomflags(void);


  // 
  // AABB calc helpers
  //
  void AABB_cone_array(CUMemBuf &aabbBuffer,
                       const float3 *base, const float3 *apex,
                       const float *brad, const float *arad, int primcnt);

  void AABB_cylinder_array(CUMemBuf &aabbBuffer,
                           const float3 *base, const float3 *apex,
                           const float *rads, int primcnt);

  void AABB_quadmesh(CUMemBuf &aabbBuffer, const float3 *verts, 
                     const int4 *indices, int primcnt);

  void AABB_ring_array(CUMemBuf &aabbBuffer,
                       const float3 *pos, const float *rads, int primcnt);

  void AABB_sphere_array(CUMemBuf &aabbBuffer,
                         const float3 *pos, const float *rads, int primcnt);

  //
  // AABB AS build input helper method
  //
  void AS_buildinp_AABB(OptixBuildInput &asInp,
                        CUdeviceptr *aabbptr, uint32_t *flagsptr, int primcnt);                          

  //
  // Core AS builder methods
  //
  int build_GAS(std::vector<OptixBuildInput>,
                CUMemBuf &ASTmpBuf,
                CUMemBuf &GASbuffer,
                uint64_t *d_ASCompactedSize,
                OptixTraversableHandle &tvh,
                cudaStream_t GASstream);

  int build_IAS(std::vector<OptixBuildInput>,
                CUMemBuf &ASTmpBuf,
                CUMemBuf &IASbuffer,
                OptixTraversableHandle &tvh,
                cudaStream_t IASstream);

  ///
  /// Geometry AS builder methods
  ///
  OptixTraversableHandle build_custprims_GAS(void);
  OptixTraversableHandle build_curves_GAS(void);
  OptixTraversableHandle build_trimeshes_GAS(void);

  //
  // Instance AS builder methods
  //
  void build_scene_IAS(void);
  void free_scene_IAS(void);

public:
  TachyonOptiX();
  ~TachyonOptiX(void);

  /// static methods for querying OptiX-supported GPU hardware independent
  /// of whether we actually have an active context.
  static int device_list(int **, char ***); ///< static GPU device list query
  static int device_count(void);            ///< static GPU device query
  static unsigned int optix_version(void);  ///< static OptiX version query

  /// console output logging callback
  void log_callback(unsigned int level, const char *tag, const char *msg);

  void print_raystats_info(void);           ///< report performance statistics
  void print_internal_struct_info(void);    ///< diagnostic info routines

  /// programmatically set verbosity
  void set_verbose_mode(TachyonOptiX::Verbosity mode) { verbose = mode; }

  /// Override the path used to load TachyonOptiXShaders.ptx at runtime.
  /// Call before the first render() / context_create() invocation.
  void set_shader_path(const char *path) {
    if (path) {
      strncpy(shaderpath, path, sizeof(shaderpath) - 1);
      shaderpath[sizeof(shaderpath) - 1] = '\0';
    }
  }

  /// reduce active memory footprint without destroying the scene
  /// by freeing internal temporary buffers used during AS builds etc.
  void minimize_memory_use(void);

  /// Initialize (or re-initialize) the underlying CUDA/OptiX hardware context.
  /// The constructor calls this automatically with the default PTX path.
  /// Call set_shader_path() first, then call this method again if the default
  /// PTX path was wrong (e.g., the working directory differs from install dir).
  void create_context(void);


  //
  // Camera parameters
  //

  /// set the camera projection mode
  void set_camera_type(CameraType m) { 
    if (camera_type != m) {
      camera_type = m; 
      regen_optix_pipeline=1; // this requires changing the raygen program
    }
  }

  /// set the camera position
  void set_camera_pos(const float *pos) { 
    memcpy(cam_pos, pos, sizeof(cam_pos)); 
  }

  /// set the camera ONB vector orientation frame
  void set_camera_ONB(const float *U, const float *V, const float *W) { 
    memcpy(cam_U, U, sizeof(cam_U)); 
    memcpy(cam_V, V, sizeof(cam_V)); 
    memcpy(cam_W, W, sizeof(cam_W)); 
  }

  /// set camera orientation to look "at" a point in space, with a given 
  /// "up" direction (camera ONB "V" vector), the remaining ONB vectors are 
  /// computed from the camera position and "at" point in space.
  void set_camera_lookat(const float *at, const float *V);

  /// set camera zoom factor
  void set_camera_zoom(float zoomfactor) { cam_zoom = zoomfactor; }

  /// depth of field on/off
  void camera_dof_enable(int onoff) { 
    if (cam_dof_enabled != (onoff != 0)) {
      cam_dof_enabled = (onoff != 0); 
      regen_optix_pipeline=1; // this requires changing the raygen program
    }
  }

  /// set depth of field focal plane distance
  void set_camera_dof_focal_dist(float d) { cam_dof_focal_dist = d; }

  /// set depth of field f/stop number
  void set_camera_dof_fnumber(float n) { cam_dof_fnumber = n; }

  /// depth of field on/off
  void camera_stereo_enable(int onoff) { 
    if (cam_stereo_enabled != (onoff != 0)) {
      cam_stereo_enabled = (onoff != 0); 
      regen_optix_pipeline=1; // this requires changing the raygen program
    }
  }

  /// set stereo eye separation
  void set_camera_stereo_eyesep(float eyesep) { cam_stereo_eyesep = eyesep; }
  
  /// set stereo convergence distance
  void set_camera_stereo_convergence_dist(float dist) {
    cam_stereo_convergence_dist = dist;
  }


  /// set depth cueing mode and parameters
  void set_cue_mode(FogMode mode, float start, float end, float density) {
    fog_mode = mode;
    fog_start = start;
    fog_end = end;
    fog_density = density;
  }


  //
  // View clipping
  //

  /// set camera clipping plane/sphere mode and parameters
  void set_clip_sphere(ViewClipMode mode, float start, float end) {
    clip_mode = mode;
    clip_start = start;
    clip_end = end;
  }

  /// VR view clipping 
  void set_clipview_mode(int mode) { 
    clipview_mode = mode;
  };


  //
  // Scene background color, gradient, etc.
  //

  /// set background rendering mode
  void set_bg_mode(BGMode m) {
    if (scene_background_mode != m) {
      scene_background_mode = m;
      regen_optix_pipeline=1; // this requires changing the miss program
    }
  }

  /// set solid background color
  void set_bg_color(float *rgb) { memcpy(scene_bg_color, rgb, sizeof(scene_bg_color)); }

  /// set color for "top" of background gradient
  void set_bg_color_grad_top(float *rgb) { memcpy(scene_bg_grad_top, rgb, sizeof(scene_bg_grad_top)); }

  /// set color for "bottom" of background gradient
  void set_bg_color_grad_bot(float *rgb) { memcpy(scene_bg_grad_bot, rgb, sizeof(scene_bg_grad_bot)); }

  /// set world "up" direction for background gradient
  void set_bg_gradient(float *vec) { memcpy(scene_bg_grad_updir, vec, sizeof(scene_bg_grad_updir)); }

  /// set background gradient "top" value (view direction dot product)
  void set_bg_gradient_topval(float v) { scene_bg_grad_topval = v; }

  /// set background gradient "bottom" value (view direction dot product)
  void set_bg_gradient_botval(float v) { scene_bg_grad_botval = v; }


  //
  // Images, Materials, and Textures
  //

  /// locate texture via user index
  int image_index_from_user_index(int userindex); 

  /// define image to be used in a texture map
  int add_tex2d_rgba4u(const unsigned char *img, int xres, int yres, 
                       int texflags, int userindex);

  /// define image to be used in a texture map
  int add_tex3d_rgba4u(const unsigned char *img, int xres, int yres, int zres,
                       int texflags, int userindex);

  /// locate material via user index
  int material_index_from_user_index(int userindex); 

  /// add a material with an associated user-provided index 
  int add_material(float ambient, float diffuse,
                   float specular, float shininess, float reflectivity,
                   float opacity, float outline, float outlinewidth, 
                   int transmode, int userindex);

  int add_material_textured(float ambient, float diffuse,
                            float specular, float shininess, float reflectivity,
                            float opacity, float outline, float outlinewidth, 
                            int transmode, int textureindex, int userindex);

  void destroy_materials();


  //
  // Lighting modes, light definitions
  //

  /// enable/disable shadows
  void shadows_enable(int onoff) { shadows_enabled = (onoff != 0); }

  /// enable/disable denoiser
  void denoiser_enable(int onoff) { denoiser_enabled = (onoff != 0); }

  /// ambient occlusion (samples > 1 == on)
  void set_ao_samples(int cnt) { ao_samples = cnt; }

  /// set AO ambient lighting factor
  void set_ao_ambient(float aoa) { ao_ambient = aoa; }

  /// set AO direct lighting rescale factor
  void set_ao_direct(float aod) { ao_direct = aod; }

  /// set AO maximum occlusion distance
  void set_ao_maxdist(float dist) { ao_maxdist = dist; }


  void set_headlight_enable(int onoff) { 
    headlight_mode = (onoff==1) ? RT_HEADLIGHT_ON : RT_HEADLIGHT_OFF; 
  };

  void add_directional_light(const float *dir, const float *color);
  void add_positional_light(const float *pos, const float *color);
  void destroy_lights() { 
    directional_lights.clear(); 
    positional_lights.clear(); 
    regen_optix_lights=1;  
  }


  //
  // Framebuffer configuration
  //

  void framebuffer_config(int fbwidth, int fbheight, int interactive);

  void framebuffer_colorspace(int colspace);
  void framebuffer_resize(int fbwidth, int fbheight);
  void framebuffer_get_size(int &fbwidth, int &fbheight) { 
    fbwidth=width; 
    fbheight=height;
  }
  void framebuffer_clear(void);
  void framebuffer_download_rgb4u(unsigned char *imgrgb4u);
  void framebuffer_destroy(void);


  //
  // Scene management
  // 

  void destroy_scene(void);
  

  //
  // Renderer controls
  //

  /// antialiasing (samples > 1 == on)
  void set_aa_samples(int cnt) { aa_samples = (cnt < 0) ? 1 : cnt; }

  void update_rendering_state(int interactive);

  void render(); 


  //
  // Geometric primitive APIs
  // 

  /// Create geometry instance group
  int create_geom_instance_group();
  int finalize_geom_instance_group(int idx);
  int destroy_geom_instance_group(int idx);
//  int set_geom_instance_group_xforms(int idx, int n, float [][16]);


  //
  // XXX short-term host API hacks to facilitate early bring-up and testing
  //
  void add_conearray(ConeArray & model, int matidx);
  void add_curvearray(CurveArray & model, int matidx);
  void add_cylarray(CylinderArray & model, int matidx);
  void add_ringarray(RingArray & model, int matidx);
  void add_quadmesh(QuadMesh & model, int matidx);
  void add_spherearray(SphereArray & model, int matidx);
  void add_trimesh(TriangleMesh & model, int matidx);

}; 



#endif