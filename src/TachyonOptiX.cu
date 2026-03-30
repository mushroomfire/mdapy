/*
 * TachyonOptiX.cu - OptiX host-side RT engine implementation
 *
 * (C) Copyright 2013-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: TachyonOptiX.cu,v 1.89 2022/04/19 03:04:52 johns Exp $
 *
 */

/**
 *  \file TachyonOptiX.cu
 *  \brief Tachyon ray tracing host side routines and internal APIs that 
 *         provide the core ray OptiX-based RTX-accelerated tracing engine.
 *         The major responsibilities of the core engine are to manage
 *         runtime RT pipeline construction and JIT-linked shaders
 *         to build complete ray tracing pipelines, management of 
 *         RT engine state, and managing associated GPU hardware.
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
//    J. E. Stone. In, Eric Haines and Tomas Akenine-Möller, editors,
//    Ray Tracing Gems, Apress, Chapter 27, pp. 493-515, 2019.
//    https://link.springer.com/book/10.1007/978-1-4842-4427-2
//
//   "A Planetarium Dome Master Camera"
//    J. E. Stone.  In, Eric Haines and Tomas Akenine-Möller, editors,
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

#define TACHYON_INTERNAL 1
#include "TachyonOptiX.h"
// TachyonOptiXShaders.h (included transitively via TachyonOptiX.h) defines
// TACHYON_OPTIXDENOISER for OptiX >= 7.3.  In OptiX 7.6 the denoiser API
// changed: OptixDenoiserParams::denoiseAlpha is now an OptixDenoiserAlphaMode
// enum, not an int.  Undefine the flag here so the denoiser code is omitted.
#undef TACHYON_OPTIXDENOISER
#include <optix_stubs.h>
#include <optix_function_table_definition.h>


#include "ProfileHooks.h"
//#include "/home/johns/graphics/tachyon/src/ProfileHooks.h"

#if 0
#define DBG()
#else
#define DBG() if (verbose == RT_VERB_DEBUG) { printf("TachyonOptiX) %s\n", __func__); }
#endif

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  }}



//
// CUDA kernels for post-processing denoiser results
//
#if defined(TACHYON_OPTIXDENOISER)

__global__ static void post_denoise_rgba4u(uchar4 *rgba4u,
                                           float4 *rgba4f,
                                           int tonemap_mode,
                                           float tonemap_exposure,
                                           int colorspace,
                                           int xres, int yres) {
  unsigned int imgsz = xres * yres;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  //
  // Here we copy/convert the output RGBA color buffer to uchar4
  // format and the final output colorspace.  
  //
  // If denoising is enabled, then we have to invert the operations
  // performed prior to denoising so that we present the followup
  // tone mapping operators with expected inputs.
  //
  if (idx < imgsz) {
    // read in denoised sRGB approximation
    float4 sRGB_approx20 = rgba4f[idx];

    // convert from the sRGB approximation back to linear
    float4 lin = sRGB_to_linear_approx_20(sRGB_approx20);

    // invert range pre-scaling operation
    lin *= 1.25f; // this also inverts the modification to alpha

    // HDR tone mapping operators need to be applied after denoising
    // has been completed.  If we use tone mapping on an LDR input,
    // we may have to revert from sRGB to linear before applying the
    // TMO, and then convert back to sRGB.
    // Also performs color space conversion if required
    float4 tonedcol;
    tonedcol = tonemap_color(lin, tonemap_mode, tonemap_exposure, colorspace);

    float4 outcol;
    if (colorspace == RT_COLORSPACE_sRGB)
      outcol = linear_to_sRGB(tonedcol);
    else    
      outcol = tonedcol;

    // clamping is applied during conversion to uchar4
    rgba4u[idx] = make_color_rgb4u(outcol);
  }
}

#endif


//
// Main RT engine class
//

TachyonOptiX::TachyonOptiX(void) {
  verbose = RT_VERB_DEBUG;     // ensure debug macro produces output first time
  DBG();

  PROFILE_PUSH_RANGE("TachyonOptiX::TachyonOptiX()", RTPROF_GENERAL);
  rt_timer = wkf_timer_create(); // create and initialize timer
  wkf_timer_start(rt_timer);

  lasterr = OPTIX_SUCCESS;      // begin with no error state set

  context_created = 0;         // no context yet
  cuda_ctx = 0;                // take over current CUDA context, if not set
  stream = 0;                  // stream 0
  optix_ctx = nullptr;         // no valid context yet
  pipe = nullptr;              // no valid pipeline
  general_module = nullptr;    // no module has been loaded/created
  curve_module = nullptr;      // no module has been loaded/created
  scene_created = 0;           // scene has not been created

  // set default shader path for runtime demand-loading
  strcpy(shaderpath, "TachyonOptiXShaders.ptx");

  // clear timers
  time_ctx_setup = 0.0;
  time_ctx_validate = 0.0;
  time_ctx_AS_build = 0.0;
  time_ray_tracing = 0.0;
  time_image_io = 0.0;

  memset((void *) &sbt, 0, sizeof(sbt)); // clear SBT record

  // clear host-side rtLaunch OptiX launch parameter buffer
  memset(&rtLaunch, 0, sizeof(rtLaunch));

  // set default scene background state
  scene_background_mode = RT_BACKGROUND_TEXTURE_SOLID;
  memset(scene_bg_color,    0, sizeof(scene_bg_color));
  memset(scene_bg_grad_top, 0, sizeof(scene_bg_grad_top));
  memset(scene_bg_grad_bot, 0, sizeof(scene_bg_grad_bot));
  memset(scene_bg_grad_updir, 0, sizeof(scene_bg_grad_updir));
  scene_bg_grad_topval = 1.0f;
  scene_bg_grad_botval = -scene_bg_grad_topval;
  // this has to be recomputed prior to rendering when topval/botval change
  scene_bg_grad_invrange = 1.0f / (scene_bg_grad_topval - scene_bg_grad_botval);

  camera_type = RT_PERSPECTIVE;
  float tmp_pos[3] = { 0.0f, 0.0f, -1.0f };
  float tmp_U[3] = { 1.0f, 0.0f, 0.0f };
  float tmp_V[3] = { 0.0f, 1.0f, 0.0f };
  float tmp_W[3] = { 0.0f, 0.0f, 1.0f };
  memcpy(cam_pos, tmp_pos, sizeof(cam_pos));
  memcpy(cam_U, tmp_U, sizeof(cam_U));
  memcpy(cam_V, tmp_V, sizeof(cam_V));
  memcpy(cam_W, tmp_W, sizeof(cam_W));
  cam_zoom = 1.0f;                    // default field of view

  cam_dof_enabled = 0;                // disable DoF by default
  cam_dof_focal_dist = 2.0f;          // default focal plane dist
  cam_dof_fnumber = 64.0f;            // default focal ratio

  cam_stereo_enabled = 0;             // disable stereo by default
  cam_stereo_eyesep = 0.06f;          // default eye separation
  cam_stereo_convergence_dist = 2.0f; // default convergence

  clipview_mode = RT_VIEWCLIP_NONE;   // VR HMD fade+clipping plane/sphere
  clipview_start = 1.0f;              // VR HMD fade+clipping radial start dist
  clipview_end = 0.2f;                // VR HMD fade+clipping radial end dist

  headlight_mode = RT_HEADLIGHT_OFF;  // VR HMD headlight disabled by default

  denoiser_enabled = RT_DENOISER_OFF; // disable denoiser by default
  shadows_enabled = RT_SHADOWS_OFF;   // disable shadows by default

  aa_samples = 0;                     // no AA samples by default

  ao_samples = 0;                     // no AO samples by default
  ao_direct = 0.3f;                   // AO direct contribution is 30%
  ao_ambient = 0.7f;                  // AO ambient contribution is 70%
  ao_maxdist = RT_DEFAULT_MAX;        // default is no max occlusion distance

  fog_mode = RT_FOG_NONE;             // fog/cueing disabled by default
  fog_start = 0.0f;                   // default fog start at camera
  fog_end = 10.0f;                    // default fog end
  fog_density = 0.32f;                // default exp^2 fog density

  scene_max_depth = 21;               // set reasonable default ray depth 
  scene_max_trans = scene_max_depth;  // set max trans crossings to match depth
  scene_epsilon = 5.e-5f * 50;        // set default scene epsilon

  lasterr = OPTIX_SUCCESS;            // clear any error state
  width = 1024;                       // default 
  height = 1024;                      // default 

  verbose = RT_VERB_MIN;              // quiet console by default
  check_verbose_env();                // check for user-overridden verbose flag

  regen_optix_pipeline=1;             // force regen of pipeline 
  regen_optix_sbt=1;                  // force regen of SBT
  regen_optix_lights=1;               // force regen of lights

  // NOTE: create_context() is intentionally NOT called here.
  // The caller (mdapy Impl) must call set_shader_path() first, then
  // create_context() + destroy_scene() manually after construction.
  // This avoids loading TachyonOptiXShaders.ptx from the wrong cwd.

  PROFILE_POP_RANGE();
}


// destructor...
TachyonOptiX::~TachyonOptiX(void) {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptiX::~TachyonOptiX()", RTPROF_GENERAL);

  cudaDeviceSynchronize(); CUERR;

  if (context_created) {
    destroy_context();
  }

#if 0
  // XXX this is only for use with memory debugging tools!
  cudaDeviceReset();
#endif

  wkf_timer_destroy(rt_timer);

  PROFILE_POP_RANGE();
}


// Global OptiX logging callback
static void TachyonOptixLogCallback(unsigned int level,
                                    const char* tag,
                                    const char* message,
                                    void* cbdata) {
  if (cbdata != NULL) {
    TachyonOptiX *tcy = (TachyonOptiX *) cbdata;
    tcy->log_callback(level, tag, message);
  }
}


void TachyonOptiX::log_callback(unsigned int level, 
                                const char *tag, const char *msg) {
  // Log callback levels:
  //  1: fatal non-recoverable error, context needs to be destroyed
  //  2: recoverable error, invalid call params, etc.
  //  3: warning hints about slow perf, etc.
  //  4: print status or progress messages
  if ((verbose == RT_VERB_DEBUG) || (level < 4))
    printf("TachyonOptiX) [%s]: %s\n", tag, msg);
}


// check environment for verbose timing/debugging output flags
static TachyonOptiX::Verbosity get_verbose_flag(int inform=0) {
  TachyonOptiX::Verbosity myverbosity = TachyonOptiX::RT_VERB_MIN;
  char *verbstr = getenv("TACHYONOPTIXVERBOSE");
  if (verbstr != NULL) {
//    printf("TachyonOptiX) verbosity config request: '%s'\n", verbstr);
    if (!strcasecmp(verbstr, "MIN")) {
      myverbosity = TachyonOptiX::RT_VERB_MIN;
      if (inform)
        printf("TachyonOptiX) verbose setting: minimum\n");
    } else if (!strcasecmp(verbstr, "TIMING")) {
      myverbosity = TachyonOptiX::RT_VERB_TIMING;
      if (inform)
        printf("TachyonOptiX) verbose setting: timing data\n");
    } else if (!strcasecmp(verbstr, "DEBUG")) {
      myverbosity = TachyonOptiX::RT_VERB_DEBUG;
      if (inform)
        printf("TachyonOptiX) verbose setting: full debugging data\n");
    }
  }
  return myverbosity; 
}


int TachyonOptiX::device_list(int **devlist, char ***devnames) {
  TachyonOptiX::Verbosity dl_verbose = get_verbose_flag();
  if (dl_verbose == RT_VERB_DEBUG)
     printf("TachyonOptiX::device_list()\n");

  int devcount = 0;
  cudaGetDeviceCount(&devcount);
  return devcount;
}


int TachyonOptiX::device_count(void) {
  TachyonOptiX::Verbosity dl_verbose = get_verbose_flag();
  if (dl_verbose == RT_VERB_DEBUG)
     printf("TachyonOptiX::device_count()\n");
  
  return device_list(NULL, NULL);
}


unsigned int TachyonOptiX::optix_version(void) {
  TachyonOptiX::Verbosity dl_verbose = get_verbose_flag();
  if (dl_verbose == RT_VERB_DEBUG)
     printf("TachyonOptiX::optix_version()\n");

  /// The OptiX version.
  /// - major =  OPTIX_VERSION/10000
  /// - minor = (OPTIX_VERSION%10000)/100
  /// - micro =  OPTIX_VERSION%100

  unsigned int version=OPTIX_VERSION;

  return version;
}


void TachyonOptiX::check_verbose_env() {
  verbose = get_verbose_flag(1);
}


void TachyonOptiX::create_context() {
  DBG();
  time_ctx_create = 0;
  if (context_created)
    return;

  PROFILE_PUSH_RANGE("TachyonOptiX::create_context()", RTPROF_GENERAL);

  double starttime = wkf_timer_timenow(rt_timer);

  if (verbose == RT_VERB_DEBUG)
    printf("TachyonOptiX) creating context...\n");

  if (lasterr == OPTIX_SUCCESS) {
    rt_ptx_code_string = NULL;

    if (verbose == RT_VERB_DEBUG) {
      printf("TachyonOptiX) Loading PTX src from compilation...\n"); 
    }
    rt_ptx_code_string = internal_compiled_ptx_src();
    if (!rt_ptx_code_string) {
      if (verbose == RT_VERB_DEBUG) {
        printf("TachyonOptiX) Loading PTX src from disk\n"); 
      }
      if (read_ptx_src(shaderpath, &rt_ptx_code_string) != 0) {
        printf("TachyonOptiX) Failed to load PTX shaders: '%s'\n", shaderpath);
        return;
      }
    }
  }

  //
  // initialize CUDA for this thread if not already
  // 
#if 0
  cudaSetDevice(0); // XXX hack for dev/testing on 'eclipse'
#endif
  cudaFree(0); // initialize CUDA
 
  lasterr = optixInit();
  if (lasterr == OPTIX_ERROR_UNSUPPORTED_ABI_VERSION) {
    //
    // Correspondence of OptiX versions with driver ABI versions:
    // OptiX: 7.0.0  7.1.0  7.2.0  7.3.0  7.4.0
    //   ABI:    22     36     41     47     55
    //
    printf("TachyonOptiX) OptiX initialization failed driver is too old.\n");
    printf("TachyonOptiX) Driver does not support ABI version %d\n", 
           OPTIX_ABI_VERSION);
    return;
  }

  cudaStreamCreate(&stream);
  
  OptixDeviceContextOptions options = {};
  optixDeviceContextCreate(cuda_ctx, &options, &optix_ctx);

  lasterr = optixDeviceContextSetLogCallback(optix_ctx,
                                             TachyonOptixLogCallback,
                                             this,
                                             4); // enable all levels 
  

  if (lasterr == OPTIX_SUCCESS)
    context_create_module();

  double time_ptxsrc = wkf_timer_timenow(rt_timer);
  if (verbose >= RT_VERB_TIMING) {
    printf("TachyonOptiX) load PTX shader src %.1f secs\n", time_ptxsrc - starttime);
  }

  if (lasterr == OPTIX_SUCCESS)
    context_create_pipeline();

  //
  // Preallocate various performance-critical buffers and device-side
  // storage required during the performance-critical phases of rendering.
  // These buffers are either fixed-size for the entire run
  // or they are buffers that will be repeatedly reused, so we grow
  // their size but do not free or shrink them, except after a call to 
  // minimize_memory_use().
  //

  // pre-allocate compacted size buffer so we don't have to allocate
  // during in-flight AS creation
  compactedSizeBuffer.set_size(sizeof(uint64_t));

  // pre-allocate a reasonable size scene IAS buffer to avoid
  // later runtime overheads.
  IASBuffer.set_size(8L * 1024L * 1024L);

  // pre-allocate rtLaunch-sized launchParamsBuffer
  launchParamsBuffer.set_size(sizeof(rtLaunch));

#if defined(TACHYON_OPTIXDENOISER)
  // initialize denoiser and allocate buffers
  context_create_denoiser();
#endif

  double time_pipeline = wkf_timer_timenow(rt_timer);
  if (verbose >= RT_VERB_TIMING) {
    printf("TachyonOptiX) create RT pipeline %.1f secs\n", time_pipeline - time_ptxsrc);
  }

  time_ctx_create = wkf_timer_timenow(rt_timer) - starttime;

  if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) context creation time: %.2f\n", time_ctx_create);
  }

  context_created = 1;

  PROFILE_POP_RANGE();
}


void TachyonOptiX::minimize_memory_use(void) {
  if (!context_created)
    return;

  // free internal temporary buffers use for AS builds
  // or other purposes, but don't free the existing scene
  ASTempBuffer.free();
}



int TachyonOptiX::read_ptx_src(const char *ptxfilename, char **ptxstring) {
  DBG();
  FILE *ptxfp = fopen(ptxfilename, "r");
  if (ptxfp == NULL) {
    return -1;
  } 

  // find size and load RT PTX source
  fseek(ptxfp, 0, SEEK_END);
  long ptxsize = ftell(ptxfp);
  fseek(ptxfp, 0, SEEK_SET);
  *ptxstring = (char *) calloc(1, ptxsize + 1);
  if (fread(*ptxstring, ptxsize, 1, ptxfp) != 1) {
    return -1;
  }
  
  return 0; 
}



void TachyonOptiX::context_create_denoiser() {
#if defined(TACHYON_OPTIXDENOISER)
  denoiser_ctx = nullptr;
  memset(&denoiser_options, 0, sizeof(denoiser_options));
  optixDenoiserCreate(optix_ctx, OPTIX_DENOISER_MODEL_KIND_LDR,
                      &denoiser_options, &denoiser_ctx);

  denoiser_resize_update();
#endif
}


void TachyonOptiX::context_destroy_denoiser() {
#if defined(TACHYON_OPTIXDENOISER)
  if (denoiser_ctx) {
    optixDenoiserDestroy(denoiser_ctx);
    denoiser_ctx = nullptr;

    denoiser_scratch.free();
    denoiser_state.free();
    denoiser_colorbuffer.free();
    denoiser_denoisedbuffer.free();
  }
#endif
}


void TachyonOptiX::denoiser_resize_update() {
#if defined(TACHYON_OPTIXDENOISER)
  if (denoiser_ctx) {
    optixDenoiserComputeMemoryResources(denoiser_ctx, width, height,
                                        &denoiser_sizes);

    long newsz = max(denoiser_sizes.withOverlapScratchSizeInBytes,
                     denoiser_sizes.withoutOverlapScratchSizeInBytes);
    denoiser_scratch.set_size(newsz);

    denoiser_state.set_size(denoiser_sizes.stateSizeInBytes);

    optixDenoiserSetup(denoiser_ctx, stream, width, height,
                       denoiser_state.cu_dptr(), denoiser_state.get_size(),
                       denoiser_scratch.cu_dptr(), denoiser_scratch.get_size());

    int fbsz = width * height * sizeof(float4);
    denoiser_colorbuffer.set_size(fbsz, stream);
    denoiser_denoisedbuffer.set_size(fbsz, stream);
  }
#endif
}


void TachyonOptiX::denoiser_launch() {
#if defined(TACHYON_OPTIXDENOISER)
  // run denoiser on color buffer, then post-convert to uchar4 output buffer
  if (denoiser_ctx && denoiser_enabled) {
    OptixDenoiserParams denoiser_params = {};
    denoiser_params.denoiseAlpha = 1;
    denoiser_params.hdrIntensity = (CUdeviceptr)0;

    // blend between input and denoised output.
    // subframe_index will be set to a non-zero value before we get here.
    // Since SNR increases w/ sqrt(N) samples, we use N^0.5, N^0.33, or N^0.25
    // curves to gradually blend in less of the denoiser output as sample 
    // counts rise.
    denoiser_params.blendFactor = 
      1.0f - (1.0f / powf(rtLaunch.frame.subframe_index, 0.5f));

    if (verbose == RT_VERB_DEBUG) {
      printf("TachyonOptiX) Accum. Buf AI Denoising Blend Factor: %.2f\n", 
             denoiser_params.blendFactor);
    }

    OptixImage2D input_layer = {};
    input_layer.data = denoiser_colorbuffer.cu_dptr();
    input_layer.width = width;
    input_layer.height = height;
    input_layer.rowStrideInBytes = width * sizeof(float4);
    input_layer.pixelStrideInBytes = sizeof(float4);
    input_layer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixImage2D output_layer = {};
    output_layer.data = denoiser_denoisedbuffer.cu_dptr();
    output_layer.width = width;
    output_layer.height = height;
    output_layer.rowStrideInBytes = width * sizeof(float4);
    output_layer.pixelStrideInBytes = sizeof(float4);
    output_layer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixDenoiserGuideLayer denoiser_guidelayer = {};
    OptixDenoiserLayer denoiser_layer = {};
    denoiser_layer.input = input_layer;
    denoiser_layer.output = output_layer;

    optixDenoiserInvoke(denoiser_ctx, stream, &denoiser_params,
                        denoiser_state.cu_dptr(), denoiser_state.get_size(),
                        &denoiser_guidelayer, &denoiser_layer, 
                        1,    // only one layer used
                        0, 0, // not a tiled denoising run
                        denoiser_scratch.cu_dptr(), denoiser_scratch.get_size());

    // copy+convert denoised image to the uchar4 output framebuffer
    dim3 Bsz(128, 1, 1);
    dim3 Gsz((width*height + Bsz.x - 1)/Bsz.x, 1, 1);
    post_denoise_rgba4u<<<Gsz, Bsz, 0, stream>>>(
                         (uchar4 *) framebuffer.cu_dptr(),
                         (float4 *) denoiser_denoisedbuffer.cu_dptr(),
                         rtLaunch.frame.tonemap_mode,
                         rtLaunch.frame.tonemap_exposure,
                         rtLaunch.frame.colorspace,
                         width, height);
  }
#endif
}





void TachyonOptiX::context_create_exception_pgms() {
  DBG();
  exceptionPGs.resize(1);

  OptixProgramGroupOptions pgOpts = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
  pgDesc.raygen.module            = general_module;           

  pgDesc.raygen.entryFunctionName="__exception__all";

  char log[2048];
  size_t sizeof_log = sizeof(log);
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts,
                                    log, &sizeof_log, &exceptionPGs[0]);

  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) exception construction log:\n %s\n", log);
  }
}


void TachyonOptiX::context_destroy_exception_pgms() {
  DBG();
  for (auto &pg : exceptionPGs)
    optixProgramGroupDestroy(pg);
  exceptionPGs.clear();
}


void TachyonOptiX::context_create_raygen_pgms() {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptiX::context_create_raygen_pgms()", RTPROF_GENERAL);

  raygenPGs.resize(1);
      
  OptixProgramGroupOptions pgOpts = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgDesc.raygen.module            = general_module;           

  // Assign the raygen program according to the active camera
  const char *raygenfctn=nullptr;
  switch (camera_type) {
    case RT_PERSPECTIVE:
      if (cam_dof_enabled) {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_perspective_stereo_dof";
        else
          raygenfctn = "__raygen__camera_perspective_dof";
      } else {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_perspective_stereo";
        else
          raygenfctn = "__raygen__camera_perspective";
      }
      break;

    case RT_ORTHOGRAPHIC:
      if (cam_dof_enabled) {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_orthographic_stereo_dof";
        else 
          raygenfctn = "__raygen__camera_orthographic_dof";
      } else {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_orthographic_stereo";
        else 
          raygenfctn = "__raygen__camera_orthographic";
      }
      break;

    case RT_CUBEMAP:
      if (cam_dof_enabled) {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_cubemap_stereo_dof";
        else
          raygenfctn = "__raygen__camera_cubemap_dof";
      } else {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_cubemap_stereo";
        else
          raygenfctn = "__raygen__camera_cubemap";
      }
      break;

    case RT_DOME_MASTER:
      if (cam_dof_enabled) {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_dome_master_stereo_dof";
        else
          raygenfctn = "__raygen__camera_dome_master_dof";
      } else {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_dome_master_stereo";
        else
          raygenfctn = "__raygen__camera_dome_master";
      }
      break;

    case RT_EQUIRECTANGULAR:
      if (cam_dof_enabled) {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_equirectangular_stereo_dof";
        else
          raygenfctn = "__raygen__camera_equirectangular_dof";
      } else {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_equirectangular_stereo";
        else
          raygenfctn = "__raygen__camera_equirectangular";
      }
      break;

    case RT_OCTAHEDRAL:
      if (cam_dof_enabled) {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_octahedral_stereo_dof";
        else
          raygenfctn = "__raygen__camera_octahedral_dof";
      } else {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_octahedral_stereo";
        else
          raygenfctn = "__raygen__camera_octahedral";
      }
      break;

    case RT_OCULUS_RIFT:
      if (cam_dof_enabled) {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_oculus_rift_stereo_dof";
        else
          raygenfctn = "__raygen__camera_oculus_rift_dof";
      } else {
        if (cam_stereo_enabled)
          raygenfctn = "__raygen__camera_oculus_rift_stereo";
        else
          raygenfctn = "__raygen__camera_oculus_rift";
      }
      break;
  }
  pgDesc.raygen.entryFunctionName=raygenfctn;
  if (verbose == RT_VERB_DEBUG)
    printf("TachyonOptiX) raygen: '%s'\n", raygenfctn);

  char log[2048];
  size_t sizeof_log = sizeof(log);
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts,
                                    log, &sizeof_log, &raygenPGs[0]);

  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) raygen construction log:\n %s\n", log);
  }

  PROFILE_POP_RANGE();
}


void TachyonOptiX::context_destroy_raygen_pgms() {
  DBG();
  for (auto &pg : raygenPGs)
    optixProgramGroupDestroy(pg);
  raygenPGs.clear();
}


void TachyonOptiX::context_create_miss_pgms() {
  DBG();
  missPGs.resize(RT_RAY_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof(log);
      
  OptixProgramGroupOptions pgOpts = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgDesc.miss.module              = general_module;           

  //
  // radiance rays
  //
 
  // Assign the miss program according to the active background mode 
  const char *missfctn=nullptr;
  switch (scene_background_mode) {
    case RT_BACKGROUND_TEXTURE_SKY_SPHERE:
      missfctn = "__miss__radiance_gradient_bg_sky_sphere";
      break;

    case RT_BACKGROUND_TEXTURE_SKY_ORTHO_PLANE:
      missfctn = "__miss__radiance_gradient_bg_sky_plane";
      break;

    case RT_BACKGROUND_TEXTURE_SOLID:
    default:
      missfctn = "__miss__radiance_solid_bg";
      break;
  }
  pgDesc.miss.entryFunctionName=missfctn;
  if (verbose == RT_VERB_DEBUG)
    printf("TachyonOptiX) miss: '%s'\n", missfctn);

  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log, 
                                    &missPGs[RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) miss radiance construction log:\n %s\n", log);
  }

  // shadow rays
  pgDesc.miss.entryFunctionName   = "__miss__shadow_nop";
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log, 
                                    &missPGs[RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) miss shadow construction log:\n %s\n", log);
  }
}


void TachyonOptiX::context_destroy_miss_pgms() {
  DBG();
  for (auto &pg : missPGs)
    optixProgramGroupDestroy(pg);
  missPGs.clear();
}


void TachyonOptiX::context_create_curve_hitgroup_pgms() {
  DBG();
  curvePGs.resize(RT_RAY_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof( log );
      
  OptixProgramGroupOptions pgOpts     = {};
  OptixProgramGroupDesc pgDesc        = {};
  pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  auto &hg = pgDesc.hitgroup;

  hg.moduleCH            = general_module;           
  hg.moduleAH            = general_module;           

  // Assign intersection fctn from the OptiX-internal module
  hg.moduleIS = curve_module;
  hg.entryFunctionNameIS = 0; // automatically supplied for built-in module

  // radiance rays
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) curve anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) curve closesthit: %s\n", hg.entryFunctionNameCH);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log, 
                                    &curvePGs[RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) curve hitgroup radiance construction log:\n"
           " %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  // XXX if we ever want to do two-pass shadows, we might use ray masks
  // and intersect opaque geometry first and do transmissive geom last
  //   hg.entryFunctionNameAH = "__anyhit__shadow_opaque";

  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) curve anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) curve closesthit: %s\n", hg.entryFunctionNameCH);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log, 
                                    &curvePGs[RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) curve hitgroup shadow construction log:\n %s\n", log);
  }
}


void TachyonOptiX::context_destroy_curve_hitgroup_pgms() {
  DBG();
  for (auto &pg : curvePGs)
    optixProgramGroupDestroy(pg);
  curvePGs.clear();
}


void TachyonOptiX::context_create_hwtri_hitgroup_pgms() {
  DBG();
  trimeshPGs.resize(RT_RAY_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof( log );
      
  OptixProgramGroupOptions pgOpts     = {};
  OptixProgramGroupDesc pgDesc        = {};
  pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  auto &hg = pgDesc.hitgroup;

  hg.moduleCH            = general_module;           
  hg.moduleAH            = general_module;           

  // radiance rays
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) triangle anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) triangle closesthit: %s\n", hg.entryFunctionNameCH);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log, 
                                    &trimeshPGs[RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) triangle hitgroup radiance construction log:\n"
           " %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  // XXX if we ever want to do two-pass shadows, we might use ray masks
  // and intersect opaque geometry first and do transmissive geom last
  //   hg.entryFunctionNameAH = "__anyhit__shadow_opaque";

  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) triangle anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) triangle closesthit: %s\n", hg.entryFunctionNameCH);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log, 
                                    &trimeshPGs[RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) triangle hitgroup shadow construction log:\n"
           " %s\n", log);
  }
}


void TachyonOptiX::context_destroy_hwtri_hitgroup_pgms() {
  DBG();
  for (auto &pg : trimeshPGs)
    optixProgramGroupDestroy(pg);
  trimeshPGs.clear();
}


void TachyonOptiX::context_create_intersection_pgms() {
  DBG();
  custprimPGs.resize(RT_CUST_PRIM_COUNT * RT_RAY_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof(log);

  OptixProgramGroupOptions pgOpts = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  auto &hg = pgDesc.hitgroup;
  hg.moduleIS = general_module;
  hg.moduleCH = general_module;
  hg.moduleAH = general_module;

  //
  // Cones 
  //
  const int conePG = RT_CUST_PRIM_CONE * RT_RAY_TYPE_COUNT;

  // radiance rays
  hg.entryFunctionNameIS = "__intersection__cone_array_color";
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log,
                                    &custprimPGs[conePG + RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) cone radiance intersection construction log:\n"
           " %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameIS = "__intersection__cone_array_color";
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log,
                                    &custprimPGs[conePG + RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) cone shadow intersection construction log:\n"
           " %s\n", log);
  }


  //
  // Cylinders
  //
  const int cylPG = RT_CUST_PRIM_CYLINDER * RT_RAY_TYPE_COUNT;

  // radiance rays
  hg.entryFunctionNameIS = "__intersection__cylinder_array_color";
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log,
                                    &custprimPGs[cylPG + RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) cylinder radiance intersection construction log:\n"
           " %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameIS = "__intersection__cylinder_array_color";
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log,
                                    &custprimPGs[cylPG + RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) cylinder shadow intersection construction log:\n"
           " %s\n", log);
  }


  //
  // Quad mesh 
  // 
  const int quadPG = RT_CUST_PRIM_QUAD * RT_RAY_TYPE_COUNT;

  // radiance rays
  hg.entryFunctionNameIS = "__intersection__quadmesh";
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log,
                                    &custprimPGs[quadPG + RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) quad radiance intersection construction log:\n"
           " %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameIS = "__intersection__quadmesh";
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log,
                                    &custprimPGs[quadPG + RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) quad shadow intersection construction log:\n"
           " %s\n", log);
  }


  //
  // Rings
  // 
  const int ringPG = RT_CUST_PRIM_RING * RT_RAY_TYPE_COUNT;

  // radiance rays
  hg.entryFunctionNameIS = "__intersection__ring_array";
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log,
                                    &custprimPGs[ringPG + RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) ring radiance intersection construction log:\n"
           " %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameIS = "__intersection__ring_array";
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log,
                                    &custprimPGs[ringPG + RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) ring shadow intersection construction log:\n"
           " %s\n", log);
  }


  //
  // Spheres
  //
  const int spherePG = RT_CUST_PRIM_SPHERE * RT_RAY_TYPE_COUNT;

  // radiance rays
  hg.entryFunctionNameIS = "__intersection__sphere_array";
  hg.entryFunctionNameCH = "__closesthit__radiance_general";
  hg.entryFunctionNameAH = "__anyhit__radiance_nop";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log,
                                    &custprimPGs[spherePG + RT_RAY_TYPE_RADIANCE]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) sphere radiance intersection construction log:\n"
           " %s\n", log);
  }

  // shadow rays
  hg.entryFunctionNameIS = "__intersection__sphere_array";
  hg.entryFunctionNameCH = "__closesthit__shadow_nop";
  hg.entryFunctionNameAH = "__anyhit__shadow_transmission";
  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) anyhit: %s\n", hg.entryFunctionNameAH);
    printf("TachyonOptiX) closesthit: %s\n", hg.entryFunctionNameCH);
    printf("TachyonOptiX) intersection: %s\n", hg.entryFunctionNameIS);
  }
  lasterr = optixProgramGroupCreate(optix_ctx, &pgDesc, 1, &pgOpts, 
                                    log, &sizeof_log,
                                    &custprimPGs[spherePG + RT_RAY_TYPE_SHADOW]);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) sphere shadow intersection construction log:\n"
           " %s\n", log);
  }
}


void TachyonOptiX::context_destroy_intersection_pgms() {
  DBG();
  for (auto &pg : custprimPGs)
    optixProgramGroupDestroy(pg);
  custprimPGs.clear();
}


void TachyonOptiX::context_create_module() {
  DBG();

  OptixModuleCompileOptions moduleCompOpts = {};
//  moduleCompOpts.maxRegisterCount    = 50;
  moduleCompOpts.maxRegisterCount    = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  moduleCompOpts.optLevel            = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
 
  // NOTE: lineinfo is required for profiling tools like nsight compute.
  // OptiX RT PTX must also be compiled using the "--generate-line-info" flag.
#if OPTIX_VERSION >= 70400
  // OptiX 7.4 has renamed the debug level enums/macros according to 
  // their runtime performance impact 
  moduleCompOpts.debugLevel          = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
  moduleCompOpts.debugLevel          = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif

  pipeCompOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  pipeCompOpts.usesMotionBlur        = false;

  // XXX OptiX 7.4 deprecates the use of a single global payload count
  //     in favor of per-program module compile options that indicate both
  //     the number of payload values and their read/write usage, to better
  //     optimized register use.  When using the OptiX >= 7.4 payloadType 
  //     data, the global pipeline options should set numPayloadValues 
  //     to zero here.
  // See:
  // https://raytracing-docs.nvidia.com/optix7/guide/index.html#payload#payload
  //
  pipeCompOpts.numPayloadValues      = 3;
  pipeCompOpts.numAttributeValues    = 2;

  // XXX enable exceptions full-time during development/testing
  if ((getenv("TACHYONOPTIXDEBUG") != NULL)) {
    pipeCompOpts.exceptionFlags        = OPTIX_EXCEPTION_FLAG_DEBUG |
                                         OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                         OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
  } else {
    pipeCompOpts.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
  }
  pipeCompOpts.pipelineLaunchParamsVariableName = "rtLaunch";

#if (OPTIX_VERSION >= 70100)
  pipeCompOpts.usesPrimitiveTypeFlags = 
    OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
    OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR;
//  pipeCompOpts.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE;
//  pipeCompOpts.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
#endif

  char log[2048];
  size_t sizeof_log = sizeof(log);
  lasterr = optixModuleCreateFromPTX(optix_ctx, &moduleCompOpts, &pipeCompOpts,
                                     rt_ptx_code_string,
                                     strlen(rt_ptx_code_string),
                                     log, &sizeof_log, &general_module);

  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) general_module construction log:\n %s\n", log);
  }

#if OPTIX_VERSION >= 70100
  //
  // Lookup OptiX built-in intersection pgm/module for curves
  //
  OptixBuiltinISOptions ISopts = {};
  ISopts.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
//  ISopts.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
//  ISopts.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
#if OPTIX_VERSION >= 70400
  ISopts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  ISopts.curveEndcapFlags = OPTIX_CURVE_ENDCAP_DEFAULT;
#endif
  ISopts.usesMotionBlur = false;
  optixBuiltinISModuleGet(optix_ctx, &moduleCompOpts, &pipeCompOpts,
                          &ISopts, &curve_module);
#endif


}


void TachyonOptiX::context_destroy_module() {
  DBG();

  if (general_module)
    optixModuleDestroy(general_module);

  if (curve_module)
    optixModuleDestroy(curve_module);
}


void TachyonOptiX::context_create_pipeline() {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptiX::context_create_pipeline()", RTPROF_GENERAL);

  if (lasterr == OPTIX_SUCCESS)
    context_create_exception_pgms();

  if (lasterr == OPTIX_SUCCESS)
    context_create_raygen_pgms();
  if (lasterr == OPTIX_SUCCESS)
    context_create_miss_pgms();
  if (lasterr == OPTIX_SUCCESS)
    context_create_curve_hitgroup_pgms();
  if (lasterr == OPTIX_SUCCESS)
    context_create_hwtri_hitgroup_pgms();
  if (lasterr == OPTIX_SUCCESS)
    context_create_intersection_pgms();

  std::vector<OptixProgramGroup> programGroups;
  for (auto &pg : exceptionPGs)
    programGroups.push_back(pg);
  for (auto &pg : raygenPGs)
    programGroups.push_back(pg);
  for (auto &pg : missPGs)
    programGroups.push_back(pg);
  for (auto &pg : curvePGs)
    programGroups.push_back(pg);
  for (auto &pg : trimeshPGs)
    programGroups.push_back(pg);
  for (auto &pg : custprimPGs)
    programGroups.push_back(pg);

  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) creating complete pipeline...\n");
  }
  
  char log[2048];
  size_t sizeof_log = sizeof(log);
  OptixPipelineLinkOptions pipeLinkOpts = {};
  pipeLinkOpts.maxTraceDepth            = 21; // OptiX recursion limit is 31
  lasterr = optixPipelineCreate(optix_ctx, &pipeCompOpts, &pipeLinkOpts,
                                programGroups.data(), (int)programGroups.size(),
                                log, &sizeof_log, &pipe);
  if ((verbose == RT_VERB_DEBUG) && (sizeof_log > 1)) {
    printf("TachyonOptiX) pipeline construction log:\n %s\n", log);
  }

  // max allowed stack sz appears to be 64kB per category
  optixPipelineSetStackSize(pipe, 
                            8*1024, ///< direct IS/AH callable stack sz
                            8*1024, ///< direct RG/MS/CH callable stack sz
                            8*1024, ///< continuation stack sz
                            1);     ///< max traversal depth

  regen_optix_pipeline=0;
  regen_optix_sbt=1;

  PROFILE_POP_RANGE();
}


void TachyonOptiX::context_destroy_pipeline() {
  DBG();
  cudaDeviceSynchronize(); CUERR;

  SBT_destroy();

  custprimsGASBuffer.free();
#if OPTIX_VERSION >= 70100
  curvesGASBuffer.free();
#endif
  trimeshesGASBuffer.free();

  if (pipe != nullptr) {
    if (verbose == RT_VERB_DEBUG)
      printf("TachyonOptiX) destroying existing pipeline...\n");

    optixPipelineDestroy(pipe);
    pipe=nullptr;
  }

  context_destroy_raygen_pgms();
  context_destroy_miss_pgms();
  context_destroy_curve_hitgroup_pgms();
  context_destroy_hwtri_hitgroup_pgms();
  context_destroy_intersection_pgms();
  context_destroy_exception_pgms();

  regen_optix_pipeline=1;
  regen_optix_sbt=1;
}



void TachyonOptiX::SBT_create_programs() {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptiX::SBT_create_programs()", RTPROF_SBT);

  PROFILE_PUSH_RANGE("Upload SBT PGM Recs", RTPROF_SBT);

  // build exception records
  std::vector<ExceptionRecord> exceptionRecords;
  for (int i=0; i<exceptionPGs.size(); i++) {
    ExceptionRecord rec = {};
    optixSbtRecordPackHeader(exceptionPGs[i], &rec);
    rec.data = nullptr;
    exceptionRecords.push_back(rec);
  }
  exceptionRecordsBuffer.resize_upload(exceptionRecords, stream);
  sbt.exceptionRecord = exceptionRecordsBuffer.cu_dptr();

  // build raygen records
  std::vector<RaygenRecord> raygenRecords;
  for (int i=0; i<raygenPGs.size(); i++) {
    RaygenRecord rec = {};
    optixSbtRecordPackHeader(raygenPGs[i], &rec);
    rec.data = nullptr;
    raygenRecords.push_back(rec);
  }
  raygenRecordsBuffer.resize_upload(raygenRecords, stream);
  sbt.raygenRecord = raygenRecordsBuffer.cu_dptr();

  // build miss records
  std::vector<MissRecord> missRecords;
  for (int i=0; i<missPGs.size(); i++) {
    MissRecord rec = {};
    optixSbtRecordPackHeader(missPGs[i], &rec);
    rec.data = nullptr;
    missRecords.push_back(rec);
  }
  missRecordsBuffer.resize_upload(missRecords, stream);
  sbt.missRecordBase          = missRecordsBuffer.cu_dptr();
  sbt.missRecordStrideInBytes = sizeof(MissRecord);
  sbt.missRecordCount         = (int) missRecords.size();

  PROFILE_POP_RANGE();
}


void TachyonOptiX::SBT_create_hitgroups() {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptiX::SBT_create_hitgroups()", RTPROF_SBT);

  PROFILE_PUSH_RANGE("Upload SBT PGM Recs", RTPROF_SBT);
  // beginning of geometry-associated processing
  PROFILE_PUSH_RANGE("Pack Geom Recs", RTPROF_GEOM);

  // build hitgroup records
  //   Note: SBT must not contain any NULLs, stubs must exist at least
  std::vector<HGRecordGroup> HGRecGroups;

  // Add cone arrays to SBT
  const int conePG = RT_CUST_PRIM_CONE * RT_RAY_TYPE_COUNT;
  int numCones = (int) conearrays.size();
  for (int objID=0; objID<numCones; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &p = rec.radiance.data;
    p.cone.base     = (float3 *) coneBaseBuffers[objID].cu_dptr();
    p.cone.apex     = (float3 *) coneApexBuffers[objID].cu_dptr();
    p.cone.baserad  = (float *) coneBaseRadBuffers[objID].cu_dptr();
    p.cone.apexrad  = (float *) coneApexRadBuffers[objID].cu_dptr();

    // common geometry params
    p.prim_color = (float3 *) conePrimColorBuffers[objID].cu_dptr();
    p.uniform_color = conearrays[objID].uniform_color;
    p.materialindex = conearrays[objID].materialindex;
    p.geomflags = 0; // initialize geomflags to empty until updated later

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(custprimPGs[conePG + RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(custprimPGs[conePG + RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }


  // Add cylinder arrays to SBT
  const int cylPG = RT_CUST_PRIM_CYLINDER * RT_RAY_TYPE_COUNT;
  int numCyls = (int) cyarrays.size();
  for (int objID=0; objID<numCyls; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &p = rec.radiance.data;
    p.cyl.start  = (float3 *) cyStartBuffers[objID].cu_dptr();
    p.cyl.end    = (float3 *) cyEndBuffers[objID].cu_dptr();
    p.cyl.radius = (float *) cyRadiusBuffers[objID].cu_dptr();

    // common geometry params
    p.prim_color = (float3 *) cyPrimColorBuffers[objID].cu_dptr();
    p.uniform_color = cyarrays[objID].uniform_color;
    p.materialindex = cyarrays[objID].materialindex;
    p.geomflags = 0; // initialize geomflags to empty until updated later

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(custprimPGs[cylPG + RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(custprimPGs[cylPG + RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }


  // Add quad meshes to SBT
  const int quadPG = RT_CUST_PRIM_QUAD * RT_RAY_TYPE_COUNT;
  int numQuads = (int) quadmeshes.size();
  for (int objID=0; objID<numQuads; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &q = rec.radiance.data.quadmesh;
    q.vertices = (float3 *) quadMeshVertBuffers[objID].cu_dptr();
    q.indices  = (int4 *) quadMeshIdxBuffers[objID].cu_dptr();
    q.normals  = (float3 *) quadMeshVertNormalBuffers[objID].cu_dptr();
    q.packednormals  = (uint4 *) quadMeshVertPackedNormalBuffers[objID].cu_dptr();
    q.vertcolors3f = (float3 *) quadMeshVertColor3fBuffers[objID].cu_dptr();
    q.vertcolors4u = (uchar4 *) quadMeshVertColor4uBuffers[objID].cu_dptr();

    // common geometry params
    auto &p = rec.radiance.data;
    p.prim_color = (float3 *) quadMeshPrimColorBuffers[objID].cu_dptr();
    p.uniform_color = quadmeshes[objID].uniform_color;
    p.materialindex = quadmeshes[objID].materialindex;
    p.geomflags = 0; // initialize geomflags to empty until updated later

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(custprimPGs[quadPG + RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(custprimPGs[quadPG + RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }


  // Add ring arrays to SBT
  const int ringPG = RT_CUST_PRIM_RING * RT_RAY_TYPE_COUNT;
  int numRings = (int) riarrays.size();
  for (int objID=0; objID<numRings; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &p = rec.radiance.data;
    p.ring.center = (float3 *) riCenterBuffers[objID].cu_dptr();
    p.ring.norm   = (float3 *) riNormalBuffers[objID].cu_dptr();
    p.ring.inrad  = (float *) riInRadiusBuffers[objID].cu_dptr();
    p.ring.outrad = (float *) riOutRadiusBuffers[objID].cu_dptr();

    // common geometry params
    p.prim_color = (float3 *) riPrimColorBuffers[objID].cu_dptr();
    p.uniform_color = riarrays[objID].uniform_color;
    p.materialindex = riarrays[objID].materialindex;
    p.geomflags = 0; // initialize geomflags to empty until updated later

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(custprimPGs[ringPG + RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(custprimPGs[ringPG + RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }


  // Add sphere arrays to SBT
  const int spherePG = RT_CUST_PRIM_SPHERE * RT_RAY_TYPE_COUNT;
  int numSpheres = (int) sparrays.size();
  for (int objID=0; objID<numSpheres; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &p = rec.radiance.data;
    p.sphere.PosRadius = (float4 *) spPosRadiusBuffers[objID].cu_dptr();

    // common geometry params
    p.prim_color = (float3 *) spPrimColorBuffers[objID].cu_dptr();
    p.uniform_color = sparrays[objID].uniform_color;
    p.materialindex = sparrays[objID].materialindex;
    p.geomflags = 0; // initialize geomflags to empty until updated later

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(custprimPGs[spherePG + RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(custprimPGs[spherePG + RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }


#if OPTIX_VERSION >= 70100
  // Add curve arrays to SBT
  int numCurves = (int) curvearrays.size();
  for (int objID=0; objID<numCurves; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &p = rec.radiance.data;
    p.curve.vertices = (float3 *) curveVertBuffers[objID].cu_dptr();
    p.curve.vertradii = (float *) curveVertRadBuffers[objID].cu_dptr();
    p.curve.segindices  = (int *) curveSegIdxBuffers[objID].cu_dptr();

    // common geometry params
    p.prim_color = (float3 *) curvePrimColorBuffers[objID].cu_dptr();
    p.uniform_color = curvearrays[objID].uniform_color;
    p.materialindex = curvearrays[objID].materialindex;
    p.geomflags = 0; // initialize geomflags to empty until updated later

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(curvePGs[RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(curvePGs[RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }
#endif


  // Add triangle meshes to SBT
  int numTrimeshes = (int) trimeshes.size();
  for (int objID=0; objID<numTrimeshes; objID++) {
    HGRecordGroup rec = {};

    // set primitive array data on the first SBT hitgroup record of the group
    auto &t = rec.radiance.data.trimesh;
    t.vertices = (float3 *) triMeshVertBuffers[objID].cu_dptr();
    t.indices = (int3 *) triMeshIdxBuffers[objID].cu_dptr();
    t.normals = (float3 *) triMeshVertNormalBuffers[objID].cu_dptr();
    t.packednormals = (uint4 *) triMeshVertPackedNormalBuffers[objID].cu_dptr();
    t.vertcolors3f = (float3 *) triMeshVertColor3fBuffers[objID].cu_dptr();
    t.vertcolors4u = (uchar4 *) triMeshVertColor4uBuffers[objID].cu_dptr();
    t.tex2d = (float2 *) triMeshTex2dBuffers[objID].cu_dptr();
    t.tex3d = (float3 *) triMeshTex3dBuffers[objID].cu_dptr();

    // common geometry params
    auto &p = rec.radiance.data;
    p.prim_color = (float3 *) triMeshPrimColorBuffers[objID].cu_dptr();
    p.uniform_color = trimeshes[objID].uniform_color;
    p.materialindex = trimeshes[objID].materialindex;
    p.geomflags = 0; // initialize geomflags to empty until updated later

    // replicate data to all records in the group
    rec.shadow = rec.radiance;

    // write record headers
    optixSbtRecordPackHeader(trimeshPGs[RT_RAY_TYPE_RADIANCE], &rec.radiance);
    optixSbtRecordPackHeader(trimeshPGs[RT_RAY_TYPE_SHADOW], &rec.shadow);
    HGRecGroups.push_back(rec);
  }

  PROFILE_STREAM_SYNC_PRETTY(stream); // sync only for clearer profile traces

  // end of geometry-associated work
  PROFILE_POP_RANGE();

  PROFILE_PUSH_RANGE("Upload SBT", RTPROF_GEOM);

  // upload and set the final SBT hitgroup array
  int hgsz = hitgroupRecordGroups.size();
  int hgrgsz = HGRecGroups.size();
  if (hgrgsz > 0) {
    // temporarily append the contents of HGRecGroups to hitgroupRecordGroups
    // so they are also included in the SBT
    // pre-grow hitgroupRecordGroups to final size prior to append loop...
    if (hitgroupRecordGroups.capacity() < (hgsz+hgrgsz))
      hitgroupRecordGroups.reserve(hgsz+hgrgsz);

    // append HGRecGroups and upload the final HG record list to the GPU
    for (auto &r: HGRecGroups) {
      hitgroupRecordGroups.push_back(r);
    }
  }

  // update SBT hitgroup geomflags from materials before upload
  SBT_update_hitgroup_geomflags();

  hitgroupRecordsBuffer.resize_upload(hitgroupRecordGroups, stream);
  sync_hitgroupRecordGroups = 0;

  sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.cu_dptr();
  sbt.hitgroupRecordStrideInBytes = sizeof(HGRecord);

  // Each HGRecordGroup contains RT_RAY_TYPE_COUNT HGRecords, so we multiply
  // the vector size by RT_RAY_TYPE_COUNT to get the total HG record count
  sbt.hitgroupRecordCount = (int) hitgroupRecordGroups.size()*RT_RAY_TYPE_COUNT;

  if (hgrgsz > 0) {
    // delete temporarily appended HGRecGroups records 
    hitgroupRecordGroups.erase(hitgroupRecordGroups.begin()+hgsz, 
                               hitgroupRecordGroups.end());
  } 

  cudaStreamSynchronize(stream);
  regen_optix_sbt=0;

  PROFILE_POP_RANGE();
  PROFILE_POP_RANGE();
}


void TachyonOptiX::SBT_clear() {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptix::SBT_clear", RTPROF_SBT);

  // set the host-side buffer sizes to zero length, but retain the 
  // GPU device-side memory allocations so they can be reused rather
  // than forcing complete reallocation in the (very likely) cases that 
  // the new contents are very close or identical in size.
  exceptionRecordsBuffer.clear_persist_allocation();
  raygenRecordsBuffer.clear_persist_allocation();
  missRecordsBuffer.clear_persist_allocation();
  hitgroupRecordsBuffer.clear_persist_allocation(); 

  regen_optix_sbt=1;

  PROFILE_STREAM_SYNC_PRETTY(stream); // sync only for clearer profile traces

  PROFILE_POP_RANGE();
}


void TachyonOptiX::SBT_destroy() {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptix::SBT_destroy", RTPROF_SBT);

  // actually free all GPU device memory allocations
  exceptionRecordsBuffer.free(stream);
  raygenRecordsBuffer.free(stream);
  missRecordsBuffer.free(stream);
  hitgroupRecordsBuffer.free(stream); 

  // clear to all zeroes to ensure no possibility of accidental reuse
  memset((void *) &sbt, 0, sizeof(sbt));
  regen_optix_sbt=1;

  PROFILE_STREAM_SYNC_PRETTY(stream); // sync only for clearer profile traces

  PROFILE_POP_RANGE();
}


void TachyonOptiX::SBT_update_hitgroup_geomflags() {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptix::SBT_update_geomflags", RTPROF_SBT);

#if defined(TACHYON_USE_GEOMFLAGS)
  // update every hitgroup record's geomflags from the latest 
  // material list, along with geometry color/texture mode encodings, etc.
  for (auto &g: hitgroupRecordGroups) {
    // copy material flags into lowest bits of geomflags
    int matidx = g.radiance.data.materialindex;
    int matflags = materialcache[matidx].matflags;
    g.radiance.data.geomflags = matflags;
    g.shadow.data.geomflags = matflags;
  }

  sync_hitgroupRecordGroups = 1;
#endif

  PROFILE_POP_RANGE();
}



void TachyonOptiX::AABB_cone_array(CUMemBuf &aabbBuffer,
                                   const float3 *base, const float3 *apex,
                                   const float *brad, const float *arad, 
                                   int primcnt) {
  // XXX AABB calcs should be done in CUDA on the GPU...
  std::vector<OptixAabb> hostAabb(primcnt); // temp array for aabb generation
  for (int i=0; i<primcnt; i++) {
    auto &b = base[i];
    auto &a = apex[i];
    float baserad = brad[i];
    float apexrad = arad[i];

    hostAabb[i].minX = fminf(b.x - baserad, a.x - apexrad);
    hostAabb[i].minY = fminf(b.y - baserad, a.y - apexrad);
    hostAabb[i].minZ = fminf(b.z - baserad, a.z - apexrad);
    hostAabb[i].maxX = fmaxf(b.x + baserad, a.x + apexrad);
    hostAabb[i].maxY = fmaxf(b.y + baserad, a.y + apexrad);
    hostAabb[i].maxZ = fmaxf(b.z + baserad, a.z + apexrad);
  }

  aabbBuffer.resize_upload(hostAabb);
}


void TachyonOptiX::AABB_cylinder_array(CUMemBuf &aabbBuffer,
                                       const float3 *base, const float3 *apex,
                                       const float *rads, int primcnt) {
  // XXX AABB calcs should be done in CUDA on the GPU...
  std::vector<OptixAabb> hostAabb(primcnt); // temp array for aabb generation
  for (int i=0; i<primcnt; i++) {
    auto &b = base[i];
    auto &a = apex[i];
    float rad = rads[i];

    hostAabb[i].minX = fminf(b.x - rad, a.x - rad);
    hostAabb[i].minY = fminf(b.y - rad, a.y - rad);
    hostAabb[i].minZ = fminf(b.z - rad, a.z - rad);
    hostAabb[i].maxX = fmaxf(b.x + rad, a.x + rad);
    hostAabb[i].maxY = fmaxf(b.y + rad, a.y + rad);
    hostAabb[i].maxZ = fmaxf(b.z + rad, a.z + rad);
  }

  aabbBuffer.resize_upload(hostAabb);
}


void TachyonOptiX::AABB_quadmesh(CUMemBuf &aabbBuffer, const float3 *verts, 
                                 const int4 *indices, int primcnt) {
  // XXX AABB calcs should be done in CUDA on the GPU...
  std::vector<OptixAabb> hostAabb(primcnt); // temp array for aabb generation
  if (indices == NULL) {
    for (int i=0; i<primcnt; i++) {
      int idx4 = i*4;
      OptixAabb bbox;
      float3 tmp = verts[idx4];
      bbox.minX = tmp.x;
      bbox.minY = tmp.y;
      bbox.minZ = tmp.z;
      bbox.maxX = tmp.x;
      bbox.maxY = tmp.y;
      bbox.maxZ = tmp.z;

      tmp = verts[idx4+1];
      bbox.minX = fminf(bbox.minX, tmp.x);
      bbox.minY = fminf(bbox.minY, tmp.y);
      bbox.minZ = fminf(bbox.minZ, tmp.z);
      bbox.maxX = fmaxf(bbox.maxX, tmp.x);
      bbox.maxY = fmaxf(bbox.maxY, tmp.y);
      bbox.maxZ = fmaxf(bbox.maxZ, tmp.z);

      tmp = verts[idx4+2];
      bbox.minX = fminf(bbox.minX, tmp.x);
      bbox.minY = fminf(bbox.minY, tmp.y);
      bbox.minZ = fminf(bbox.minZ, tmp.z);
      bbox.maxX = fmaxf(bbox.maxX, tmp.x);
      bbox.maxY = fmaxf(bbox.maxY, tmp.y);
      bbox.maxZ = fmaxf(bbox.maxZ, tmp.z);

      tmp = verts[idx4+3];
      bbox.minX = fminf(bbox.minX, tmp.x);
      bbox.minY = fminf(bbox.minY, tmp.y);
      bbox.minZ = fminf(bbox.minZ, tmp.z);
      bbox.maxX = fmaxf(bbox.maxX, tmp.x);
      bbox.maxY = fmaxf(bbox.maxY, tmp.y);
      bbox.maxZ = fmaxf(bbox.maxZ, tmp.z);

      hostAabb[i] = bbox;
    }
  } else {
    for (int i=0; i<primcnt; i++) {
      int4 index = indices[i];
      OptixAabb bbox;
      float3 tmp = verts[index.x];
      bbox.minX = tmp.x;
      bbox.minY = tmp.y;
      bbox.minZ = tmp.z;
      bbox.maxX = tmp.x;
      bbox.maxY = tmp.y;
      bbox.maxZ = tmp.z;

      tmp = verts[index.y];
      bbox.minX = fminf(bbox.minX, tmp.x);
      bbox.minY = fminf(bbox.minY, tmp.y);
      bbox.minZ = fminf(bbox.minZ, tmp.z);
      bbox.maxX = fmaxf(bbox.maxX, tmp.x);
      bbox.maxY = fmaxf(bbox.maxY, tmp.y);
      bbox.maxZ = fmaxf(bbox.maxZ, tmp.z);

      tmp = verts[index.z];
      bbox.minX = fminf(bbox.minX, tmp.x);
      bbox.minY = fminf(bbox.minY, tmp.y);
      bbox.minZ = fminf(bbox.minZ, tmp.z);
      bbox.maxX = fmaxf(bbox.maxX, tmp.x);
      bbox.maxY = fmaxf(bbox.maxY, tmp.y);
      bbox.maxZ = fmaxf(bbox.maxZ, tmp.z);

      tmp = verts[index.w];
      bbox.minX = fminf(bbox.minX, tmp.x);
      bbox.minY = fminf(bbox.minY, tmp.y);
      bbox.minZ = fminf(bbox.minZ, tmp.z);
      bbox.maxX = fmaxf(bbox.maxX, tmp.x);
      bbox.maxY = fmaxf(bbox.maxY, tmp.y);
      bbox.maxZ = fmaxf(bbox.maxZ, tmp.z);

      hostAabb[i] = bbox;
    }
  }

  aabbBuffer.resize_upload(hostAabb);
}


void TachyonOptiX::AABB_ring_array(CUMemBuf &aabbBuffer,
                                   const float3 *pos, const float *rads, 
                                   int primcnt) {
  // XXX AABB calcs should be done in CUDA on the GPU...
  std::vector<OptixAabb> hostAabb(primcnt); // temp array for aabb generation
  for (int i=0; i<primcnt; i++) {
    float rad = rads[i];
    hostAabb[i].minX = pos[i].x - rad;
    hostAabb[i].minY = pos[i].y - rad;
    hostAabb[i].minZ = pos[i].z - rad;
    hostAabb[i].maxX = pos[i].x + rad;
    hostAabb[i].maxY = pos[i].y + rad;
    hostAabb[i].maxZ = pos[i].z + rad;
  }

  aabbBuffer.resize_upload(hostAabb);
}


void TachyonOptiX::AABB_sphere_array(CUMemBuf &aabbBuffer,
                                     const float3 *pos, const float *rads, 
                                     int primcnt) {
  // XXX AABB calcs should be done in CUDA on the GPU...
  std::vector<OptixAabb> hostAabb(primcnt); // temp array for aabb generation
  for (int i=0; i<primcnt; i++) {
    float rad = rads[i];
    hostAabb[i].minX = pos[i].x - rad;
    hostAabb[i].minY = pos[i].y - rad;
    hostAabb[i].minZ = pos[i].z - rad;
    hostAabb[i].maxX = pos[i].x + rad;
    hostAabb[i].maxY = pos[i].y + rad;
    hostAabb[i].maxZ = pos[i].z + rad;
  }

  aabbBuffer.resize_upload(hostAabb);
}



void TachyonOptiX::AS_buildinp_AABB(OptixBuildInput &asInp,
                                    CUdeviceptr *aabbptr, 
                                    uint32_t *flagptr, int primcnt) {
  asInp = {};
  asInp.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  
  // device custom primitive buffers
#if (OPTIX_VERSION >= 70100)
  auto &primArray = asInp.customPrimitiveArray;
#else
  auto &primArray = asInp.aabbArray;
#endif

  primArray.aabbBuffers                 = aabbptr;
  primArray.numPrimitives               = primcnt;
  primArray.strideInBytes               = 0; // tight-packed, sizeof(OptixAabb)

  // Ensure that anyhit is called only once for transparency handling
  *flagptr = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
  primArray.flags                       = flagptr;

  primArray.numSbtRecords               = 1;
  primArray.sbtIndexOffsetBuffer        = 0; // No per-primitive record
  primArray.sbtIndexOffsetSizeInBytes   = 0;
  primArray.sbtIndexOffsetStrideInBytes = 0;
  primArray.primitiveIndexOffset        = 0;
}


int TachyonOptiX::build_GAS(std::vector<OptixBuildInput> asInp,
                            CUMemBuf &ASTmpBuf,
                            CUMemBuf &GASbuffer,
                            uint64_t *d_ASCompactedSize,
                            OptixTraversableHandle &tvh,
                            cudaStream_t GASstream) {
  PROFILE_PUSH_RANGE("TachyonOptiX::build_GAS()", RTPROF_ACCEL);
  const int arrayCount = asInp.size();

  // BLAS setup
  OptixAccelBuildOptions asOpts           = {};
  asOpts.motionOptions.numKeys            = 1;
  asOpts.buildFlags                       = OPTIX_BUILD_FLAG_NONE |
                                            OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  asOpts.operation                        = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blasBufSizes = {};
  optixAccelComputeMemoryUsage(optix_ctx, &asOpts, asInp.data(),
                               arrayCount, &blasBufSizes);

  // prepare compaction
  OptixAccelEmitDesc emitDesc = {};
  emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = (CUdeviceptr) d_ASCompactedSize; // uint64_t in GPU device memory

  // 
  // execute build (main stage)
  // 

  // If we already have an existing temp buffer of required size
  // we use it as-is to avoid paying the reallocation time cost
  if (ASTmpBuf.get_size() < blasBufSizes.tempSizeInBytes)
    ASTmpBuf.set_size(blasBufSizes.tempSizeInBytes, GASstream);

  CUMemBuf outputBuffer;
  outputBuffer.set_size(blasBufSizes.outputSizeInBytes, GASstream);

  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) GAS/BLAS buffer sizes: temp %d  output %d\n", 
           blasBufSizes.tempSizeInBytes, blasBufSizes.outputSizeInBytes);
  }

  tvh = {}; // clear traversable handle
  optixAccelBuild(optix_ctx, GASstream, &asOpts, asInp.data(), arrayCount,
                  ASTmpBuf.cu_dptr(), ASTmpBuf.get_size(),
                  outputBuffer.cu_dptr(), outputBuffer.get_size(),
                  &tvh, &emitDesc, 1);

  cudaStreamSynchronize(GASstream);

  // XXX compaction should only be performed when (compactedSize < outputSize) 
  //     to avoid the extra compacting pass when it is not beneficial,
  //     but this still requires the GPU-Host copy of compactedSize,
  //     and we'll have to be able to swap device pointers easily.

  // fetch compactedSize back to host
  uint64_t compactedSize = 0;
  cudaMemcpyAsync(&compactedSize, d_ASCompactedSize, sizeof(uint64_t), 
                  cudaMemcpyDeviceToHost, GASstream);
  if (verbose == RT_VERB_DEBUG) {
    cudaStreamSynchronize(GASstream);
    printf("TachyonOptiX) GAS/BLAS compacted size: %ld\n", compactedSize);
  }

  // perform compaction
  GASbuffer.set_size(compactedSize, GASstream);
  optixAccelCompact(optix_ctx, GASstream, tvh,
                    GASbuffer.cu_dptr(), GASbuffer.get_size(), &tvh);

  // at this point, the final compacted AS is stored in the final Buffer
  // and the returned traversable--ephemeral data can be destroyed...
  outputBuffer.free(GASstream);

  cudaStreamSynchronize(GASstream);
//  PROFILE_STREAM_SYNC_PRETTY(GASstream); // sync only for clearer profile traces

  PROFILE_POP_RANGE();
  return 0;
}


int TachyonOptiX::build_IAS(std::vector<OptixBuildInput> asInp,
                            CUMemBuf &ASTmpBuf,
                            CUMemBuf &IASbuf,
                            OptixTraversableHandle &tvh,
                            cudaStream_t IASstream) {
  const int arrayCount = asInp.size();
  PROFILE_PUSH_RANGE("TachyonOptiX::build_IAS()", RTPROF_ACCEL);

  // TLAS setup
  OptixAccelBuildOptions asOpts         = {};
  asOpts.buildFlags                     = OPTIX_BUILD_FLAG_NONE;
  asOpts.operation                      = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes tlasBufSizes = {};
  optixAccelComputeMemoryUsage(optix_ctx, &asOpts, asInp.data(),
                               arrayCount, &tlasBufSizes);

  // execute build (main stage)

  // If we already have an existing temp buffer of required size
  // we use it as-is to avoid paying the reallocation time cost
  if (ASTmpBuf.get_size() < tlasBufSizes.tempSizeInBytes)
    ASTmpBuf.set_size(tlasBufSizes.tempSizeInBytes, IASstream);

  // If we already have an existing IAS buffer of required size
  // we use it as-is to avoid paying the reallocation time cost
  if (IASbuf.get_size() < tlasBufSizes.outputSizeInBytes)
    IASbuf.set_size(tlasBufSizes.outputSizeInBytes, IASstream);

  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) IAS/TLAS buffer sizes: temp %d  output %d\n", 
           tlasBufSizes.tempSizeInBytes, tlasBufSizes.outputSizeInBytes);
  }

  tvh = {}; // clear traversable handle
  optixAccelBuild(optix_ctx, IASstream, &asOpts, asInp.data(), arrayCount,
                  ASTmpBuf.cu_dptr(), ASTmpBuf.get_size(),
                  IASbuf.cu_dptr(), IASbuf.get_size(),
                  &tvh, nullptr, 0);

  cudaStreamSynchronize(IASstream);
//  PROFILE_STREAM_SYNC_PRETTY(IASstream); // sync only for clearer profile traces

  PROFILE_POP_RANGE();
  return 0;
}


OptixTraversableHandle TachyonOptiX::build_curves_GAS() {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptiX::build_curves_GAS()", RTPROF_ACCEL);

  OptixTraversableHandle asHandle { 0 };

#if OPTIX_VERSION >= 70100
  const int arrayCount = curvearrays.size();

  // RTX triangle inputs, preset vector sizes
  // AS build will consume device pointers, so when these
  // are freed, we should be destroying the associated AS
  curveVertBuffers.resize(arrayCount);
  curveVertRadBuffers.resize(arrayCount);
  curveSegIdxBuffers.resize(arrayCount);
//  curveVertColor3fBuffers.resize(arrayCount);
//  curveVertColor4uBuffers.resize(arrayCount);
  curvePrimColorBuffers.resize(arrayCount);

  // store per-curve data in arrays so we can dereference and
  // submit as a single-element array below
  std::vector<OptixBuildInput> asCurveInp(arrayCount);
  std::vector<CUdeviceptr> d_vertices(arrayCount);
  std::vector<CUdeviceptr> d_vertrads(arrayCount);
  std::vector<uint32_t> asCurveInpFlags(arrayCount);

  // loop over geom buffers and incorp into AS build...
  //   Uploads each curve to the GPU before building AS,
  //   stores resulting device pointers in lists, and 
  //   prepares OptixBuildInput records containing the
  //   resulting device pointers, primitive counts, and flags.
  for (int i=0; i<arrayCount; i++) {
    CurveArray &model = curvearrays[i];
    curveVertBuffers[i].resize_upload(model.vertices);
    curveVertRadBuffers[i].resize_upload(model.vertradii);
    curveSegIdxBuffers[i].resize_upload(model.segindices);

    // optional buffers 
//    curveVertColor3fBuffers[i].free();
//    curveVertColor4uBuffers[i].free();
    curvePrimColorBuffers[i].resize_upload(model.primcolors3f);

    asCurveInp[i] = {};
    asCurveInp[i].type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    d_vertices[i] = curveVertBuffers[i].cu_dptr(); // host array of dev ptrs...
    d_vertrads[i] = curveVertRadBuffers[i].cu_dptr(); // host array of dev ptrs...

    // device curve buffers
    auto &curveArray = asCurveInp[i].curveArray;
    curveArray.curveType                 = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
#if OPTIX_VERSION >= 70400
//    curveArray.curveType                 = OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM;
#endif
  
    // device curve vertex buffer 
//    curveArray.numPrimitives             = (int)model.segindices.size();
    curveArray.numPrimitives             = 1; // num segments
    curveArray.vertexBuffers             = &d_vertices[i];
    curveArray.numVertices               = 2; // (int)model.vertices.size();
    curveArray.vertexStrideInBytes       = sizeof(float3);

    // device curve width/radii buffer
    curveArray.widthBuffers              = &d_vertrads[i];
    curveArray.widthStrideInBytes        = sizeof(float); 

    // device curve normal buffer
    // normal buffers are unused in OptiX versions <= 7.4
    curveArray.normalBuffers             = NULL;
    curveArray.normalStrideInBytes       = 0;

    // device curve index buffer 
    curveArray.indexBuffer               = curveSegIdxBuffers[i].cu_dptr();
    curveArray.indexStrideInBytes        = sizeof(int);
  
    // Ensure that anyhit is called only once for transparency handling 
    asCurveInpFlags[i] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    curveArray.flag                      = asCurveInpFlags[i];
    curveArray.primitiveIndexOffset      = 0;

#if OPTIX_VERSION >= 70400
    curveArray.endcapFlags               = OPTIX_CURVE_ENDCAP_DEFAULT;
//    curveArray.endcapFlags               = OPTIX_CURVE_ENDCAP_ON;
#endif
  }

  build_GAS(asCurveInp, ASTempBuffer, curvesGASBuffer, 
            (uint64_t *) compactedSizeBuffer.cu_dptr(), asHandle, stream);
#endif

  PROFILE_POP_RANGE();
  return asHandle;
}


OptixTraversableHandle TachyonOptiX::build_trimeshes_GAS() {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptiX::build_trimeshes_GAS()", RTPROF_ACCEL);

  PROFILE_PUSH_RANGE("Trimesh Upload, AS Input", RTPROF_GEOM);
  const int arrayCount = trimeshes.size();

  // RTX triangle inputs, preset vector sizes
  // AS build will consume device pointers, so when these
  // are freed, we should be destroying the associated AS
  triMeshVertBuffers.resize(arrayCount);
  triMeshIdxBuffers.resize(arrayCount);
  triMeshVertNormalBuffers.resize(arrayCount);
  triMeshVertPackedNormalBuffers.resize(arrayCount);
  triMeshVertColor3fBuffers.resize(arrayCount);
  triMeshVertColor4uBuffers.resize(arrayCount);
  triMeshPrimColorBuffers.resize(arrayCount);
  triMeshTex2dBuffers.resize(arrayCount);
  triMeshTex3dBuffers.resize(arrayCount);

  std::vector<OptixBuildInput> asTriInp(arrayCount);
  std::vector<CUdeviceptr> d_vertices(arrayCount);
  std::vector<uint32_t> asTriInpFlags(arrayCount);

  // loop over geom buffers and incorp into AS build...
  //   Uploads each mesh to the GPU before building AS,
  //   stores resulting device pointers in lists, and 
  //   prepares OptixBuildInput records containing the
  //   resulting device pointers, primitive counts, and flags.
  for (int i=0; i<arrayCount; i++) {
    TriangleMesh &model = trimeshes[i];
    triMeshVertBuffers[i].resize_upload(model.vertices, stream);
    triMeshIdxBuffers[i].resize_upload(model.indices, stream); // optional

    // optional buffers 
    triMeshVertNormalBuffers[i].resize_upload(model.normals, stream);
    triMeshVertPackedNormalBuffers[i].resize_upload(model.packednormals, stream);
    triMeshVertColor3fBuffers[i].resize_upload(model.vertcolors3f, stream);
    triMeshVertColor4uBuffers[i].resize_upload(model.vertcolors4u, stream);
    triMeshPrimColorBuffers[i].resize_upload(model.primcolors3f, stream);
    triMeshTex2dBuffers[i].resize_upload(model.tex2d, stream);
    triMeshTex3dBuffers[i].resize_upload(model.tex3d, stream);
    cudaStreamSynchronize(stream);

    asTriInp[i] = {};
    asTriInp[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    d_vertices[i] = triMeshVertBuffers[i].cu_dptr(); // host array of dev ptrs...

    // device triangle mesh buffers
    auto &triArray = asTriInp[i].triangleArray;
   
    // device trimesh vertex buffer 
    triArray.vertexBuffers               = &d_vertices[i];
    triArray.numVertices                 = (int)model.vertices.size();
    triArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triArray.vertexStrideInBytes         = sizeof(float3);
    
    // optional device trimesh index buffer 
    if (model.indices.size() > 0) {
      triArray.indexBuffer               = triMeshIdxBuffers[i].cu_dptr();
      triArray.numIndexTriplets          = (int)model.indices.size();
      triArray.indexFormat               = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      triArray.indexStrideInBytes        = sizeof(int3);
    } else {
      triArray.indexBuffer               = 0;
      triArray.numIndexTriplets          = 0;
#if OPTIX_VERSION >= 70100
      triArray.indexFormat               = OPTIX_INDICES_FORMAT_NONE;
#endif
      triArray.indexStrideInBytes        = 0;
    }
    triArray.preTransform                = 0; // no xform matrix
   
    // Ensure that anyhit is called only once for transparency handling 
    asTriInpFlags[i] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    triArray.flags                       = &asTriInpFlags[i];
 
    triArray.numSbtRecords               = 1;
    triArray.sbtIndexOffsetBuffer        = 0; 
    triArray.sbtIndexOffsetSizeInBytes   = 0; 
    triArray.sbtIndexOffsetStrideInBytes = 0; 
    triArray.primitiveIndexOffset        = 0;
  }
    
  PROFILE_POP_RANGE();

  OptixTraversableHandle asHandle;
  build_GAS(asTriInp, ASTempBuffer, trimeshesGASBuffer, 
            (uint64_t *) compactedSizeBuffer.cu_dptr(), asHandle, stream);

  PROFILE_POP_RANGE();
  return asHandle;
}


OptixTraversableHandle TachyonOptiX::build_custprims_GAS() {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptiX::build_custprims_GAS()", RTPROF_ACCEL);

  // RTX custom primitive inputs
  // AS build will consume device pointers, so when these
  // are freed, we should be destroying the associated AS

  const int coneCount = conearrays.size();
  coneBaseBuffers.resize(coneCount);
  coneApexBuffers.resize(coneCount);
  coneBaseRadBuffers.resize(coneCount);
  coneApexRadBuffers.resize(coneCount);
  conePrimColorBuffers.resize(coneCount);
  coneAabbBuffers.resize(coneCount);

  const int cyCount = cyarrays.size();
  cyStartBuffers.resize(cyCount);
  cyEndBuffers.resize(cyCount);
  cyRadiusBuffers.resize(cyCount);
  cyPrimColorBuffers.resize(cyCount);
  cyAabbBuffers.resize(cyCount);

  const int quadCount = quadmeshes.size();
  quadMeshVertBuffers.resize(quadCount);
  quadMeshIdxBuffers.resize(quadCount);
  quadMeshVertNormalBuffers.resize(quadCount);
  quadMeshVertPackedNormalBuffers.resize(quadCount);
  quadMeshVertColor3fBuffers.resize(quadCount);
  quadMeshVertColor4uBuffers.resize(quadCount);
  quadMeshPrimColorBuffers.resize(quadCount);
  quadMeshAabbBuffers.resize(quadCount);

  const int riCount = riarrays.size();
  riCenterBuffers.resize(riCount);
  riNormalBuffers.resize(riCount);
  riInRadiusBuffers.resize(riCount);
  riOutRadiusBuffers.resize(riCount);
  riPrimColorBuffers.resize(riCount);
  riAabbBuffers.resize(riCount);

  const int spCount = sparrays.size();
  spPosRadiusBuffers.resize(spCount);
  spPrimColorBuffers.resize(spCount);
  spAabbBuffers.resize(spCount);

  const int arrayCount = coneCount + cyCount + quadCount + riCount + spCount;

  std::vector<OptixBuildInput> asInp(arrayCount);
  std::vector<CUdeviceptr> d_aabb(arrayCount);
  std::vector<uint32_t> asInpFlags(arrayCount);

  // loop over geom buffers and incorp into AS build...
  //   Uploads each mesh to the GPU before building AS,
  //   stores resulting device pointers in lists, and 
  //   prepares OptixBuildInput records containing the
  //   resulting device pointers, primitive counts, and flags.
  int bufIdx = 0;

  // Cones...
  for (int i=0; i<coneCount; i++) {
    ConeArray &m = conearrays[i];
    coneBaseBuffers[i].resize_upload(m.base);
    coneApexBuffers[i].resize_upload(m.apex);
    coneBaseRadBuffers[i].resize_upload(m.baserad);
    coneApexRadBuffers[i].resize_upload(m.apexrad);
    conePrimColorBuffers[i].resize_upload(m.primcolors3f);

    int primcnt = m.base.size();
    AABB_cone_array(coneAabbBuffers[i], m.base.data(), m.apex.data(),
                    m.baserad.data(), m.apexrad.data(), primcnt);
    int bidx = bufIdx + i;
    d_aabb[bidx] = coneAabbBuffers[i].cu_dptr();
    AS_buildinp_AABB(asInp[bidx], &d_aabb[bidx], &asInpFlags[bidx], primcnt);
  }
  bufIdx += coneCount;

  // Cylinders...
  for (int i=0; i<cyCount; i++) {
    CylinderArray &m = cyarrays[i];
    cyStartBuffers[i].resize_upload(m.start);
    cyEndBuffers[i].resize_upload(m.end);
    cyRadiusBuffers[i].resize_upload(m.radius);
    cyPrimColorBuffers[i].resize_upload(m.primcolors3f);

    int primcnt = m.radius.size();
    AABB_cylinder_array(cyAabbBuffers[i], m.start.data(), m.end.data(),
                        m.radius.data(), primcnt);
    int bidx = bufIdx + i;
    d_aabb[bidx] = cyAabbBuffers[i].cu_dptr();
    AS_buildinp_AABB(asInp[bidx], &d_aabb[bidx], &asInpFlags[bidx], primcnt);
  }
  bufIdx += cyCount;

  // Quads...
  for (int i=0; i<quadCount; i++) {
    QuadMesh &m = quadmeshes[i];
    quadMeshVertBuffers[i].resize_upload(m.vertices);
    quadMeshIdxBuffers[i].resize_upload(m.indices);
    quadMeshVertNormalBuffers[i].resize_upload(m.normals);
    quadMeshVertPackedNormalBuffers[i].resize_upload(m.packednormals);
    quadMeshVertColor3fBuffers[i].resize_upload(m.vertcolors3f);
    quadMeshVertColor4uBuffers[i].resize_upload(m.vertcolors4u);
    quadMeshPrimColorBuffers[i].resize_upload(m.primcolors3f);

    int primcnt = (m.indices.size() > 0) ? m.indices.size() : (m.vertices.size() / 4);
    AABB_quadmesh(quadMeshAabbBuffers[i], m.vertices.data(), m.indices.data(), primcnt);
    int bidx = bufIdx + i;
    d_aabb[bidx] = quadMeshAabbBuffers[i].cu_dptr();
    AS_buildinp_AABB(asInp[bidx], &d_aabb[bidx], &asInpFlags[bidx], primcnt);
  }
  bufIdx += quadCount;

  // Rings...
  for (int i=0; i<riCount; i++) {
    RingArray &m = riarrays[i];
    riCenterBuffers[i].resize_upload(m.center);
    riNormalBuffers[i].resize_upload(m.normal);
    riInRadiusBuffers[i].resize_upload(m.inrad);
    riOutRadiusBuffers[i].resize_upload(m.outrad);
    riPrimColorBuffers[i].resize_upload(m.primcolors3f);

    int primcnt = m.outrad.size();
    AABB_ring_array(riAabbBuffers[i], m.center.data(), 
                    m.outrad.data(), primcnt);
    int bidx = bufIdx + i;
    d_aabb[bidx] = riAabbBuffers[i].cu_dptr();
    AS_buildinp_AABB(asInp[bidx], &d_aabb[bidx], &asInpFlags[bidx], primcnt);
  }
  bufIdx += riCount;

  // Spheres...
  for (int i=0; i<spCount; i++) {
    SphereArray &m = sparrays[i];
    int sz = m.radius.size();
    std::vector<float4 PINALLOCS(float4)> tmp(sz);
    for (int j=0; j<sz; j++) {
      tmp[j] = make_float4(m.center[j], m.radius[i]);
    }
    spPosRadiusBuffers[i].resize_upload(tmp);
    spPrimColorBuffers[i].resize_upload(m.primcolors3f);

    int primcnt = m.radius.size();
    AABB_sphere_array(spAabbBuffers[i], m.center.data(), m.radius.data(), primcnt);
    int bidx = bufIdx + i;
    d_aabb[bidx] = spAabbBuffers[i].cu_dptr();
    AS_buildinp_AABB(asInp[bidx], &d_aabb[bidx], &asInpFlags[bidx], primcnt);
  }
  bufIdx += spCount;
    
  OptixTraversableHandle asHandle;
  build_GAS(asInp, ASTempBuffer, custprimsGASBuffer, 
            (uint64_t *) compactedSizeBuffer.cu_dptr(), asHandle, stream);

  PROFILE_POP_RANGE();
  return asHandle;
}


void TachyonOptiX::build_scene_IAS() {
  DBG();
  PROFILE_PUSH_RANGE("TachyonOptiX::build_scene_IAS()", RTPROF_ACCEL);

  OptixTraversableHandle trimeshesGAS = {};
  OptixTraversableHandle curvesGAS = {};
  OptixTraversableHandle custprimsGAS = {};

  // zero out host-side array sizes, but retain GPU-side allocation to
  // avoid costly reallocations unless absolutely necessary
  custprimsGASBuffer.clear_persist_allocation();
#if OPTIX_VERSION >= 70100
  curvesGASBuffer.clear_persist_allocation();
#endif
  trimeshesGASBuffer.clear_persist_allocation();

  //
  // (re)build GASes for each geometry class
  //
  int custprimcount = (conearrays.size() + cyarrays.size() + 
                       quadmeshes.size() + riarrays.size() + 
                       sparrays.size());
  if (custprimcount > 0) {
    custprimsGAS = build_custprims_GAS();
  }

  if (curvearrays.size() > 0) {
    curvesGAS = build_curves_GAS();
  }

  if (trimeshes.size() > 0) {
    trimeshesGAS = build_trimeshes_GAS();
  }

  int sbtOffset = 0;
  std::vector<OptixInstance> instances;

  OptixInstance tmpInst = {};
  auto &i = tmpInst;
  float identity_xform3x4[12] = {
    1.0f,  0.0f,  0.0f,  0.0f,
    0.0f,  1.0f,  0.0f,  0.0f,
    0.0f,  0.0f,  1.0f,  0.0f
  };

  // populate instance 
  memcpy(i.transform, identity_xform3x4, sizeof(identity_xform3x4));
  i.instanceId = 0;
  i.sbtOffset = 0;
  i.visibilityMask = 0xFF;
  i.flags = OPTIX_INSTANCE_FLAG_NONE;

  if (custprimsGAS) {
    i.traversableHandle = custprimsGAS;
    i.sbtOffset = sbtOffset;
    instances.push_back(i);

    sbtOffset += RT_RAY_TYPE_COUNT * custprimcount;
  }

  if (curvesGAS) {
    i.traversableHandle = curvesGAS;
    i.sbtOffset = sbtOffset;
    instances.push_back(i);

    sbtOffset += RT_RAY_TYPE_COUNT * curvearrays.size();
  }

  if (trimeshesGAS) {
    i.traversableHandle = trimeshesGAS;
    i.sbtOffset = sbtOffset;
    instances.push_back(i);

    sbtOffset += RT_RAY_TYPE_COUNT * trimeshes.size();
  }

#if 0
  printf("TachyonOptiX) custprimsGAS: %p\n", custprimsGAS);
  printf("TachyonOptiX) curvesGAS: %p\n", curvesGAS);
  printf("TachyonOptiX) trimeshesGAS: %p\n", trimeshesGAS);
  printf("TachyonOptiX) i.traversable: %p\n", i.traversableHandle);
  printf("TachyonOptiX) instance[0].traversable: %p\n", instances[0].traversableHandle);
#endif

  CUMemBuf devinstances;
  devinstances.resize_upload(instances, stream);

  std::vector<OptixBuildInput> asInstInp(1);
  asInstInp[0] = {};
  asInstInp[0].type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  asInstInp[0].instanceArray.instances    = devinstances.cu_dptr();
  asInstInp[0].instanceArray.numInstances = (int) instances.size();

  OptixTraversableHandle asHandle { 0 };
  build_IAS(asInstInp, ASTempBuffer, IASBuffer, asHandle, stream);
  rtLaunch.traversable = asHandle;

  devinstances.free(stream);

  PROFILE_STREAM_SYNC_PRETTY(stream); // sync only for clearer profile traces

  PROFILE_POP_RANGE();
}


void TachyonOptiX::destroy_context() {
  DBG();
  if (!context_created)
    return;

  destroy_scene();

  // free the normally-persistent, and globally used 
  // AS temp buffer, IAS, and compacted size buffers
  ASTempBuffer.free();
  compactedSizeBuffer.free();
  IASBuffer.free();  

  context_destroy_pipeline();
  context_destroy_module();
  framebuffer_destroy();
#if defined(TACHYON_OPTIXDENOISER)
  context_destroy_denoiser();
#endif

  // launch params buffer refers to materials/lights buffers 
  // so we destroy it first...
  launchParamsBuffer.free();
  materialsBuffer.free();
  directionalLightsBuffer.free(); 
  positionalLightsBuffer.free(); 
 
  optixDeviceContextDestroy(optix_ctx);

  regen_optix_pipeline=1;
  regen_optix_sbt=1;
  regen_optix_lights=1;
}



//
// Images, Textures, Materials
//

int TachyonOptiX::image_index_from_user_index(int userindex) {
  return userindex; // XXX short-term hack
}


int TachyonOptiX::add_tex2d_rgba4u(const unsigned char *img, 
                                   int xres, int yres,
                                   int texflags, int userindex) {
//  DBG();

  int oldtexcount = texturecache.size();
  if (oldtexcount <= userindex) {
    rt_texture t;

    // XXX do something noticable so we see that we got a bad entry...
    memset(&t, 0, sizeof(t));
    t.userindex = -1; // negative user index indicates an unused or bad entry

    texturecache.resize(userindex+1);
    for (int i=oldtexcount; i<=userindex; i++) {
      texturecache[i]=t;
    }
  }

  if (texturecache[userindex].userindex > 0) {
    return userindex;
  } else {
    if (verbose == RT_VERB_DEBUG) printf("TachyonOptiX) Adding texture[%d]\n", userindex);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t texArray;
    cudaMallocArray(&texArray, &channelDesc, xres, yres);

    // Set pitch of the source (the width in memory in bytes of the 2D array 
    // pointed to by src, including padding), we dont have any padding
    const size_t spitch = xres * sizeof(float);
    cudaMemcpy2DToArray(texArray, 0, 0, img, spitch, xres * sizeof(float),
                        yres, cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = texArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    if (texflags & RT_TEX_COLORSPACE_sRGB)
      texDesc.sRGB = 1;
    else
      texDesc.sRGB = 0;
    

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    texturecache[userindex].texflags = texflags;
    texturecache[userindex].d_img = texArray;
    texturecache[userindex].tex = texObj;
    texturecache[userindex].userindex=userindex;
  }

  return userindex;
}


int TachyonOptiX::add_tex3d_rgba4u(const unsigned char *img, 
                                   int xres, int yres, int zres,
                                   int texflags, int userindex) {
//  DBG();

  int oldtexcount = texturecache.size();
  if (oldtexcount <= userindex) {
    rt_texture t;

    // XXX do something noticable so we see that we got a bad entry...
    memset(&t, 0, sizeof(t));
    t.userindex = -1; // negative user index indicates an unused or bad entry

    texturecache.resize(userindex+1);
    for (int i=oldtexcount; i<=userindex; i++) {
      texturecache[i]=t;
    }
  }

  if (texturecache[userindex].userindex > 0) {
    return userindex;
  } else {
    if (verbose == RT_VERB_DEBUG) printf("TachyonOptiX) Adding texture[%d]\n", userindex);

    // Compute grid extents and channel description for the 3-D array
    cudaExtent gridExtent = make_cudaExtent(xres, yres, zres);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    cudaArray_t texArray;
    cudaMalloc3DArray(&texArray, &channelDesc, gridExtent);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)img,
                                              gridExtent.width*sizeof(uchar4),
                                              gridExtent.width,
                                              gridExtent.height);
    copyParams.dstArray = texArray;
    copyParams.extent   = gridExtent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = texArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    if (texflags & RT_TEX_COLORSPACE_sRGB)
      texDesc.sRGB = 1;
    else
      texDesc.sRGB = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    texturecache[userindex].texflags = texflags;
    texturecache[userindex].d_img = texArray;
    texturecache[userindex].tex = texObj;
    texturecache[userindex].userindex=userindex;
  }

  return userindex;
}


int TachyonOptiX::material_index_from_user_index(int userindex) {
  return userindex; // XXX short-term hack
}

int TachyonOptiX::add_material(float ambient, float diffuse, float specular,
                               float shininess, float reflectivity,
                               float opacity, float outline, float outlinewidth,
                               int transmode, int userindex) {
  return add_material_textured(ambient, diffuse, specular, shininess, 
                               reflectivity, opacity, outline, outlinewidth,
                               transmode, -1, userindex);
}

int TachyonOptiX::add_material_textured(float ambient, float diffuse, 
                                        float specular, float shininess, 
                                        float reflectivity, float opacity, 
                                        float outline, float outlinewidth,
                                        int transmode, 
                                        int texindex, int userindex) {
//  DBG();

  int oldmatcount = materialcache.size();
  if (oldmatcount <= userindex) {
    rt_material m;

    // XXX do something noticable so we see that we got a bad entry...
    m.ambient = 0.5f;
    m.diffuse = 0.7f;
    m.specular = 0.0f;
    m.shininess = 10.0f;
    m.reflectivity = 0.0f;
    m.opacity = 1.0f;
    m.transmode = 0;
    m.tex = 0;
    m.matflags = 0;
    m.userindex = -1; // negative user index indicates an unused or bad entry

    materialcache.resize(userindex+1);
    for (int i=oldmatcount; i<=userindex; i++) {
      materialcache[i]=m;
    }
  }

  if (materialcache[userindex].userindex > 0) {
    return userindex;
  } else {
    if (verbose == RT_VERB_DEBUG) printf("TachyonOptiX) Adding material[%d]\n", userindex);

    materialcache[userindex].ambient      = ambient;
    materialcache[userindex].diffuse      = diffuse;
    materialcache[userindex].specular     = specular;
    materialcache[userindex].shininess    = shininess;
    materialcache[userindex].reflectivity = reflectivity;
    materialcache[userindex].opacity      = opacity;
    materialcache[userindex].outline      = outline;
    materialcache[userindex].outlinewidth = outlinewidth;
    materialcache[userindex].transmode    = transmode;
    materialcache[userindex].tex          = 0;
    materialcache[userindex].matflags     = 0;
    materialcache[userindex].userindex=userindex;

    if (opacity < 1.0f) {
      materialcache[userindex].matflags |= RT_MAT_ALPHA; 
    }
  
    // set texture object 
    if (texindex >= 0) {
      materialcache[userindex].tex = texturecache[texindex].tex;
#if 0
      printf("mat[%d] texid: %llu\n", userindex, texturecache[texindex].tex);
#endif

      // set flags when texture alpha or cutout transparency is in use
      if (texturecache[texindex].texflags & RT_TEX_ALPHA) {
        materialcache[userindex].matflags |= RT_MAT_TEXALPHA; 
#if 0
        printf("mat[%d] uses cutout alpha texture, matflags: %08x\n", 
               userindex, materialcache[userindex].matflags);
#endif
      }
    }

    regen_optix_materials=1; // force a fresh material table upload to the GPU
  }

  return userindex;
}


void TachyonOptiX::destroy_materials() {
  if (verbose == RT_VERB_DEBUG) printf("TachyonOptiX) init_materials()\n");

  materialcache.clear();
  regen_optix_materials=1; // force a fresh material table upload to the GPU
}


void TachyonOptiX::add_directional_light(const float *dir, const float *color) {
  rt_directional_light l;
  l.dir = normalize(make_float3(dir[0], dir[1], dir[2]));
//  l.color = make_float3(color[0], color[1], color[2]);
  directional_lights.push_back(l);
  regen_optix_lights=1;
}


void TachyonOptiX::add_positional_light(const float *pos, const float *color) {
  rt_positional_light l;
  l.pos = make_float3(pos[0], pos[1], pos[2]);
//  l.color = make_float3(color[0], color[1], color[2]);
  positional_lights.push_back(l);
  regen_optix_lights=1;
}


void TachyonOptiX::destroy_scene() {
  DBG();
  double starttime = wkf_timer_timenow(rt_timer);
  time_ctx_destroy_scene = 0;

  // zero out all object counters
  cylinder_array_cnt = 0;
  cylinder_array_color_cnt = 0;
  ring_array_color_cnt = 0;
  sphere_array_cnt = 0;
  sphere_array_color_cnt = 0;
  tricolor_cnt = 0;
  trimesh_c4u_n3b_v3f_cnt = 0;
  trimesh_n3b_v3f_cnt = 0;
  trimesh_n3f_v3f_cnt = 0;
  trimesh_v3f_cnt = 0;

  if (!context_created)
    return;

  // XXX this renderer class isn't tracking scene state yet
  scene_created = 1;
  if (scene_created) {
    destroy_materials();
    destroy_lights();

    for (auto &&buf : coneAabbBuffers) buf.free();
    coneAabbBuffers.clear();
    for (auto &&buf : coneBaseBuffers) buf.free();
    coneBaseBuffers.clear();
    for (auto &&buf : coneApexBuffers) buf.free();
    coneApexBuffers.clear();
    for (auto &&buf : coneBaseRadBuffers) buf.free();
    coneBaseRadBuffers.clear();
    for (auto &&buf : coneApexRadBuffers) buf.free();
    coneApexRadBuffers.clear();
    for (auto &&buf : conePrimColorBuffers) buf.free();
    conePrimColorBuffers.clear();
    conearrays.clear();
  
    for (auto &&buf : curveVertBuffers) buf.free();
    curveVertBuffers.clear();
    for (auto &&buf : curveVertRadBuffers) buf.free();
    curveVertRadBuffers.clear();
    for (auto &&buf : curveSegIdxBuffers) buf.free();
    curveSegIdxBuffers.clear();
    for (auto &&buf : curvePrimColorBuffers) buf.free();
    curvePrimColorBuffers.clear();
    curvearrays.clear();

    for (auto &&buf : cyAabbBuffers) buf.free();
    cyAabbBuffers.clear();
    for (auto &&buf : cyStartBuffers) buf.free();
    cyStartBuffers.clear();
    for (auto &&buf : cyEndBuffers) buf.free();
    cyEndBuffers.clear();
    for (auto &&buf : cyRadiusBuffers) buf.free();
    cyRadiusBuffers.clear();
    for (auto &&buf : cyPrimColorBuffers) buf.free();
    cyPrimColorBuffers.clear();
    cyarrays.clear();

    for (auto &&buf : quadMeshAabbBuffers) buf.free();
    quadMeshAabbBuffers.clear();
    for (auto &&buf : quadMeshVertBuffers) buf.free();
    quadMeshVertBuffers.clear();
    for (auto &&buf : quadMeshIdxBuffers) buf.free();
    quadMeshIdxBuffers.clear();
    for (auto &&buf : quadMeshVertNormalBuffers) buf.free();
    quadMeshVertNormalBuffers.clear();
    for (auto &&buf : quadMeshVertPackedNormalBuffers) buf.free();
    quadMeshVertPackedNormalBuffers.clear();
    for (auto &&buf : quadMeshVertColor3fBuffers) buf.free();
    quadMeshVertColor3fBuffers.clear();
    for (auto &&buf : quadMeshVertColor4uBuffers) buf.free();
    quadMeshVertColor4uBuffers.clear();
    for (auto &&buf : quadMeshPrimColorBuffers) buf.free();
    quadMeshPrimColorBuffers.clear();
    quadmeshes.clear(); 

    for (auto &&buf : riAabbBuffers) buf.free();
    riAabbBuffers.clear();
    for (auto &&buf : riCenterBuffers) buf.free();
    riCenterBuffers.clear();
    for (auto &&buf : riNormalBuffers) buf.free();
    riNormalBuffers.clear();
    for (auto &&buf : riInRadiusBuffers) buf.free();
    riInRadiusBuffers.clear();
    for (auto &&buf : riOutRadiusBuffers) buf.free();
    riOutRadiusBuffers.clear();
    for (auto &&buf : riPrimColorBuffers) buf.free();
    riPrimColorBuffers.clear();
    riarrays.clear(); 

    for (auto &&buf : spAabbBuffers) buf.free();
    spAabbBuffers.clear();
    for (auto &&buf : spPosRadiusBuffers) buf.free();
    spPosRadiusBuffers.clear();
    for (auto &&buf : spPrimColorBuffers) buf.free();
    spPrimColorBuffers.clear();
    sparrays.clear();

    for (auto &&buf : triMeshVertBuffers) buf.free();
    triMeshVertBuffers.clear();
    for (auto &&buf : triMeshIdxBuffers) buf.free();
    triMeshIdxBuffers.clear();
    for (auto &&buf : triMeshVertNormalBuffers) buf.free();
    triMeshVertNormalBuffers.clear();
    for (auto &&buf : triMeshVertPackedNormalBuffers) buf.free();
    triMeshVertPackedNormalBuffers.clear();
    for (auto &&buf : triMeshVertColor3fBuffers) buf.free();
    triMeshVertColor3fBuffers.clear();
    for (auto &&buf : triMeshVertColor4uBuffers) buf.free();
    triMeshVertColor4uBuffers.clear();
    for (auto &&buf : triMeshPrimColorBuffers) buf.free();
    triMeshPrimColorBuffers.clear();
    for (auto &&buf : triMeshTex2dBuffers) buf.free();
    triMeshTex2dBuffers.clear();
    for (auto &&buf : triMeshTex3dBuffers) buf.free();
    triMeshTex3dBuffers.clear();
    trimeshes.clear();

    SBT_clear(); // only zero out the SBT, don't free GPU mem

    // zero out host-side array sizes, but retain GPU-side allocation to
    // avoid costly reallocations unless absolutely necessary
    custprimsGASBuffer.clear_persist_allocation();
#if OPTIX_VERSION >= 70100
    curvesGASBuffer.clear_persist_allocation();
#endif
    trimeshesGASBuffer.clear_persist_allocation();
  }

  double endtime = wkf_timer_timenow(rt_timer);
  time_ctx_destroy_scene = endtime - starttime;

  scene_created = 0; // scene has been destroyed
}


void TachyonOptiX::set_camera_lookat(const float *at, const float *upV) {
  // force position update to be committed to the rtLaunch struct too...
  rtLaunch.cam.pos = make_float3(cam_pos[0], cam_pos[1], cam_pos[2]);
  float3 lookat = make_float3(at[0], at[1], at[2]);
  float3 V = make_float3(upV[0], upV[1], upV[2]);
  rtLaunch.cam.W = normalize(lookat - rtLaunch.cam.pos);
  rtLaunch.cam.U = normalize(cross(rtLaunch.cam.W, V));
  rtLaunch.cam.V = normalize(cross(rtLaunch.cam.U, rtLaunch.cam.W));

  // copy new ONB vectors back to top level data structure
  cam_U[0] = rtLaunch.cam.U.x;
  cam_U[1] = rtLaunch.cam.U.y;
  cam_U[2] = rtLaunch.cam.U.z;

  cam_V[0] = rtLaunch.cam.V.x;
  cam_V[1] = rtLaunch.cam.V.y;
  cam_V[2] = rtLaunch.cam.V.z;

  cam_W[0] = rtLaunch.cam.W.x;
  cam_W[1] = rtLaunch.cam.W.y;
  cam_W[2] = rtLaunch.cam.W.z;
}


void TachyonOptiX::framebuffer_config(int fbwidth, int fbheight,
                                      int interactive) {
  DBG();
  if (!context_created)
    return;

  framebuffer_resize(fbwidth, fbheight);
  
  // do anything special for interactive...
}

void TachyonOptiX::framebuffer_colorspace(int colspace) {
  colorspace=colspace;
}

void TachyonOptiX::framebuffer_resize(int fbwidth, int fbheight) {
  DBG();
  if (!context_created)
    return;
  PROFILE_PUSH_RANGE("TachyonOptiX::framebuffer_resize()", RTPROF_GENERAL);

  width = fbwidth;
  height = fbheight;

  int fbsz = width * height * sizeof(uchar4);
  framebuffer.set_size(fbsz, stream);

  int acsz = width * height * sizeof(float4);
  accumulation_buffer.set_size(acsz, stream);

#if defined(TACHYON_OPTIXDENOISER)
  denoiser_resize_update();
#endif

#if defined(TACHYON_RAYSTATS)
  int assz = width * height * sizeof(uint4);
  raystats1_buffer.set_size(assz, stream);
  raystats2_buffer.set_size(assz, stream);
#endif

  framebuffer_clear();

  if (verbose == RT_VERB_DEBUG)
    printf("TachyonOptiX) framebuffer_resize(%d x %d)\n", width, height);

  PROFILE_POP_RANGE();
}


void TachyonOptiX::framebuffer_clear() {
  rtLaunch.frame.subframe_index = 0;     // only reset when accum buf cleared
  rtLaunch.frame.accum_normalize = 1.0f; // only reset when accum buf cleared

#if 1
  //
  // only clear the FBs as part of 1st rendering pass
  //
  rtLaunch.frame.fb_clearall = 1;        // clear all bufs during rendering
#else
  //
  // force-clear the FBs irrespective of rendering
  //
  auto fbsz = framebuffer.get_size(); 
  cudaMemsetAsync(framebuffer.dptr(), 0, fbsz, stream);

  auto acsz = accumulation_buffer.get_size(); 
  cudaMemsetAsync(accumulation_buffer.dptr(), 0, acsz, stream);

#if defined(TACHYON_RAYSTATS)
  // clear stats buffers
  auto assz = raystats1_buffer.get_size(); 
  cudaMemsetAsync(raystats1_buffer.dptr(), 0, assz, stream);
  cudaMemsetAsync(raystats2_buffer.dptr(), 0, assz, stream);
#endif

  cudaStreamSynchronize(stream);
#endif

  if (verbose == RT_VERB_DEBUG)
    printf("TachyonOptiX) framebuffer_clear(%d x %d)\n", width, height);
}


void TachyonOptiX::framebuffer_download_rgb4u(unsigned char *imgrgb4u) {
  DBG();
  framebuffer.download(imgrgb4u, width * height * sizeof(int));
}


void TachyonOptiX::framebuffer_destroy() {
  DBG();
  if (!context_created)
    return;

  framebuffer.free();
  accumulation_buffer.free();
#if defined(TACHYON_RAYSTATS)
  raystats1_buffer.free();
  raystats2_buffer.free();
#endif
}


void TachyonOptiX::render_compile_and_validate(void) {
  DBG();
  if (!context_created)
    return;

  //
  // finalize context validation, compilation, and AS generation
  //
  double startctxtime = wkf_timer_timenow(rt_timer);

  PROFILE_PUSH_RANGE("TachyonOptiX::render_compile_and_validate()", RTPROF_RENDER);

  // (re)build OptiX raygen/hitgroup/miss program pipeline
  if (regen_optix_pipeline) {
    if (pipe != nullptr)
      context_destroy_pipeline();
    context_create_pipeline();

    if ((lasterr != OPTIX_SUCCESS) /* && (verbose == RT_VERB_DEBUG) */ )
      printf("TachyonOptiX) An error occured during pipeline regen!\n"); 
  }

  double start_AS_build = wkf_timer_timenow(rt_timer);

  // start IAS + SBT (re)builds
  build_scene_IAS();
  if ((lasterr != OPTIX_SUCCESS) /* && (verbose == RT_VERB_DEBUG) */ )
    printf("TachyonOptiX) An error occured during AS regen!\n"); 

  // (re)build SBT 
  if (regen_optix_sbt) {
    SBT_clear(); // only zero out the SBT, don't free GPU mem
    SBT_create_programs();
    SBT_create_hitgroups();
#if 0
  } else if (regen_optix_materials) {
    // XXX alpha/opacity optimization:
    // Update hitgroup-flattened opacity information used to improve
    // anyhit performance by avoiding pointer/index chasing.
    // XXX update hitgroup records with new geomflags from materials, etc. 
    SBT_update_hitgroup_geomflags();
    hitgroupRecordsBuffer.resize_upload(h, stream);
#endif
  }


  if ((lasterr != OPTIX_SUCCESS) /* && (verbose == RT_VERB_DEBUG) */ )
    printf("TachyonOptiX) An error occured during SBT regen!\n"); 

  time_ctx_AS_build = wkf_timer_timenow(rt_timer) - start_AS_build;

  // upload current materials
  if (regen_optix_materials) {
    materialsBuffer.resize_upload(materialcache);
    regen_optix_materials=0; // no need to re-upload until a change occurs
  }

  // upload current lights
  if (regen_optix_lights) {
    directionalLightsBuffer.resize_upload(directional_lights);
    positionalLightsBuffer.resize_upload(positional_lights);
    regen_optix_lights=0; // no need to re-upload until a change occurs
  }

  if ((lasterr != OPTIX_SUCCESS) /* && (verbose == RT_VERB_DEBUG) */ )
    printf("TachyonOptiX) An error occured during materials/lights regen!\n"); 

  //
  // update the launch parameters that we'll pass to the optix launch:
  //
  rtLaunch.frame.size = make_int2(width, height);
  rtLaunch.frame.colorspace = colorspace;

  // XXX tone mapping params are hard-coded
  rtLaunch.frame.tonemap_mode = RT_TONEMAP_CLAMP;
  rtLaunch.frame.tonemap_exposure = 1.0f;

  rtLaunch.frame.framebuffer = (uchar4*) framebuffer.cu_dptr();
  rtLaunch.frame.accum_buffer = (float4*) accumulation_buffer.cu_dptr();

#if defined(TACHYON_OPTIXDENOISER)
  rtLaunch.frame.denoiser_colorbuffer = (float4*) denoiser_colorbuffer.cu_dptr();
  rtLaunch.frame.denoiser_enabled = denoiser_enabled;
#endif

#if defined(TACHYON_RAYSTATS)
  rtLaunch.frame.raystats1_buffer = (uint4*) raystats1_buffer.cu_dptr();
  rtLaunch.frame.raystats2_buffer = (uint4*) raystats2_buffer.cu_dptr();
#endif

  // update material table pointer
  rtLaunch.materials = (rt_material *) materialsBuffer.cu_dptr();

  // finalize camera parms
  rtLaunch.cam.pos = make_float3(cam_pos[0], cam_pos[1], cam_pos[2]);
  rtLaunch.cam.U   = make_float3(cam_U[0], cam_U[1], cam_U[2]);
  rtLaunch.cam.V   = make_float3(cam_V[0], cam_V[1], cam_V[2]);
  rtLaunch.cam.W   = make_float3(cam_W[0], cam_W[1], cam_W[2]);
  rtLaunch.cam.zoom = cam_zoom;

  rtLaunch.cam.dof_enabled = cam_dof_enabled;
  rtLaunch.cam.dof_focal_dist = cam_dof_focal_dist;
  rtLaunch.cam.dof_aperture_rad = cam_dof_focal_dist / (2.0f * cam_zoom * cam_dof_fnumber);

  rtLaunch.cam.stereo_enabled = cam_stereo_enabled;
  rtLaunch.cam.stereo_eyesep = cam_stereo_eyesep;
  rtLaunch.cam.stereo_convergence_dist = cam_stereo_convergence_dist;


  // populate rtLaunch scene data
  rtLaunch.scene.bg_color = make_float3(scene_bg_color[0],
                                        scene_bg_color[1],
                                        scene_bg_color[2]);
  rtLaunch.scene.bg_color_grad_top = make_float3(scene_bg_grad_top[0],
                                                 scene_bg_grad_top[1],
                                                 scene_bg_grad_top[2]);
  rtLaunch.scene.bg_color_grad_bot = make_float3(scene_bg_grad_bot[0],
                                                 scene_bg_grad_bot[1],
                                                 scene_bg_grad_bot[2]);
  rtLaunch.scene.bg_grad_updir = make_float3(scene_bg_grad_updir[0],
                                             scene_bg_grad_updir[1],
                                             scene_bg_grad_updir[2]);
  rtLaunch.scene.bg_grad_topval = scene_bg_grad_topval;
  rtLaunch.scene.bg_grad_botval = scene_bg_grad_botval;

  // this has to be recomputed prior to rendering when topval/botval change
  scene_bg_grad_invrange = 1.0f / (scene_bg_grad_topval - scene_bg_grad_botval);
  rtLaunch.scene.bg_grad_invrange = scene_bg_grad_invrange;

  // Add noise to gradient backgrounds to prevent Mach banding effects,
  // particularly noticable in video streams or movie renderings.
  // Compute the delta between the top and bottom gradient colors and
  // calculate the noise magnitude required, such that by adding it to the
  // scalar interpolation parameter we get more than +/-1ulp in the
  // resulting interpolated color, as represented in an 8bpp framebuffer.
  float maxcoldelta = fmaxf(fabsf(rtLaunch.scene.bg_color_grad_top - rtLaunch.scene.bg_color_grad_bot));

  // Ideally the noise mag calc would take into account both max color delta
  // and launch_dim.y to avoid banding even with very subtle gradients.
  rtLaunch.scene.bg_grad_noisemag = (3.0f/256.0f) / (maxcoldelta + 0.0005);

  rtLaunch.scene.fog_mode = fog_mode;
  rtLaunch.scene.fog_start = fog_start;
  rtLaunch.scene.fog_end = fog_end;
  rtLaunch.scene.fog_density = fog_density;

  rtLaunch.scene.epsilon = scene_epsilon;
  rtLaunch.max_depth = scene_max_depth;
  rtLaunch.max_trans = scene_max_trans;

  rtLaunch.aa_samples = 1; // aa_samples;

  rtLaunch.lights.shadows_enabled = shadows_enabled;
  rtLaunch.lights.ao_samples = ao_samples;
  if (ao_samples)
    rtLaunch.lights.ao_lightscale = 2.0f / ao_samples;
  else
    rtLaunch.lights.ao_lightscale = 0.0f;

  rtLaunch.lights.ao_ambient = ao_ambient;
  rtLaunch.lights.ao_direct  = ao_direct;
  rtLaunch.lights.ao_maxdist = ao_maxdist;
  rtLaunch.lights.headlight_mode = headlight_mode;

  rtLaunch.lights.num_dir_lights = directional_lights.size();
  rtLaunch.lights.dir_lights = (float3 *) directionalLightsBuffer.cu_dptr();
  rtLaunch.lights.num_pos_lights = positional_lights.size();
  rtLaunch.lights.pos_lights = (float3 *) positionalLightsBuffer.cu_dptr();

  time_ctx_validate = wkf_timer_timenow(rt_timer) - startctxtime;

  if (verbose == RT_VERB_DEBUG) {
    printf("TachyonOptiX) launching render: %d x %d\n", width, height);
  }

  PROFILE_POP_RANGE();
}


void TachyonOptiX::update_rendering_state(int interactive) {
  DBG();
  if (!context_created)
    return;
}


void TachyonOptiX::render() {
  DBG();
  if (!context_created)
    return;

  wkf_timer_start(rt_timer);
  double rendstarttime = wkf_timer_timenow(rt_timer);

  PROFILE_PUSH_RANGE("TachyonOptiX::render()", RTPROF_RENDER);

  update_rendering_state(0);
  render_compile_and_validate();
  double starttime = wkf_timer_timenow(rt_timer);

  //
  // run the renderer
  //
  if (lasterr == OPTIX_SUCCESS) {
    // Render only to the accumulation buffer for one less than the
    // total required number of passes
    rtLaunch.frame.update_colorbuffer = 0;

    int samples_per_pass = 1;
    rtLaunch.aa_samples = samples_per_pass; 

    PROFILE_PUSH_RANGE("TachyonOptiX--Render Loop", RTPROF_RENDERRT);
    for (int p=0; p<aa_samples; p+=samples_per_pass) {
      PROFILE_PUSH_RANGE("TachyonOptiX--launchParamsBuffer.upload()", RTPROF_TRANSFER);

      // advance to next subrame index, needed by both RNGs and
      // for accumulation_buffer handling
      // calc normalization factor for the final subframe index we'll have
      // just as we return and copy out the color buffer
      rtLaunch.frame.subframe_index += samples_per_pass;
      rtLaunch.frame.accum_normalize = 1.0f / float(rtLaunch.frame.subframe_index);

      // copy the accumulation buffer image data to the framebuffer and perform
      // type conversion and normaliztion on the image data when we reach
      // the last subframe in this internal rendering loop.
      if (p >= (aa_samples - samples_per_pass)) {
        rtLaunch.frame.update_colorbuffer = 1;
      }

      // update launch params with current buffer and subframe info      
      launchParamsBuffer.upload(&rtLaunch, 1, stream);
      PROFILE_STREAM_SYNC_PRETTY(stream); // sync only for clearer profile traces
      PROFILE_POP_RANGE();

      PROFILE_PUSH_RANGE("TachyonOptiX--optixLaunch()", RTPROF_RENDERRT);
      lasterr = optixLaunch(pipe, stream,
                            launchParamsBuffer.cu_dptr(),
                            launchParamsBuffer.get_size(),
                            &sbt,
                            rtLaunch.frame.size.x,
                            rtLaunch.frame.size.y,
                            1);

      cudaStreamSynchronize(stream);
      PROFILE_POP_RANGE();

      // ensure framebuffer clear is disabled after the first rendering pass
      rtLaunch.frame.fb_clearall = 0;
    }

    if (lasterr != OPTIX_SUCCESS) {
      printf("TachyonOptiX) Error during rendering.  Rendering aborted.\n");
    }

    double rtendtime = wkf_timer_timenow(rt_timer);
    time_ray_tracing = rtendtime - starttime;
    double totalrendertime = rtendtime - rendstarttime;

    //
    // Perform denoising if enabled and available
    //
    PROFILE_PUSH_RANGE("TachyonOptiX--denoiser_launch()", RTPROF_RENDERRT);
    denoiser_launch();
    cudaStreamSynchronize(stream);
    double denoiseendtime = wkf_timer_timenow(rt_timer);
    double denoise_time = denoiseendtime - rtendtime;
    PROFILE_POP_RANGE();

    if (lasterr != OPTIX_SUCCESS) {
      printf("TachyonOptiX) Error during denoising.  Rendering aborted.\n");
    }

    if (verbose == RT_VERB_TIMING || verbose == RT_VERB_DEBUG) {
      printf("TachyonOptiX) Render Time: %.2fms, %.1fFPS\n", 
             totalrendertime * 1.0e3, 1.0 / totalrendertime);
      printf("TachyonOptiX)   (AS %.2fms, RT %.2fms, DN %.2fms, io %.2fms)\n",
             time_ctx_AS_build * 1.0e3, time_ray_tracing * 1.0e3, 
             denoise_time * 1.0e3, time_image_io * 1.0e3);
    }
  } else {
    printf("TachyonOptiX) An error occured prior to rendering. Rendering aborted.\n");
  }
  
  PROFILE_POP_RANGE();
  PROFILE_POP_RANGE();
}



//
// Report ray tracing performance statistics
//
void TachyonOptiX::print_raystats_info(void) {
#if defined(TACHYON_RAYSTATS)
  // no stats data
  if (rtLaunch.frame.size.x < 1 || rtLaunch.frame.size.y < 1) {
    printf("TachyonOptiX) No data in ray stats buffers!\n");
    return;
  }
  CUERR

  int framesz = rtLaunch.frame.size.x * rtLaunch.frame.size.y;
  size_t bufsz = framesz * sizeof(uint4);
  uint4 *raystats1 = (uint4 *) calloc(1, bufsz);
  uint4 *raystats2 = (uint4 *) calloc(1, bufsz);
  raystats1_buffer.download(raystats1, framesz);
  raystats2_buffer.download(raystats2, framesz);
  CUERR

  // no stats data
  if (rtLaunch.frame.size.x < 1 || rtLaunch.frame.size.y < 1) {
    printf("TachyonOptiX) No data in ray stats buffers!\n");
    return;
  }

  // collect and sum all per-pixel ray stats
  unsigned long misses=0, transkips=0, primaryrays=0, shadowlights=0, 
           shadowao=0, transrays=0, reflrays=0;

  // accumulate per-pixel ray stats into totals
  for (int i=0; i<framesz; i++) {
    primaryrays  += raystats1[i].x;
    shadowlights += raystats1[i].y;
    shadowao     += raystats1[i].z;
    misses       += raystats1[i].w;
    transrays    += raystats2[i].x;
    transkips    += raystats2[i].y;
    // XXX raystats2[i].z unused at present...
    reflrays     += raystats2[i].w;
  }
  unsigned long totalrays = primaryrays + shadowlights + shadowao
                          + transrays + reflrays;

  printf("TachyonOptiX)\n");
  printf("TachyonOptiX) TachyonOptiX Scene Ray Tracing Statistics:\n");
  printf("TachyonOptiX) ----------------------------------------\n");
  printf("TachyonOptiX) Image resolution: %d x %d \n",
         rtLaunch.frame.size.x, rtLaunch.frame.size.y);
  printf("TachyonOptiX) Pixel count: %d\n", framesz);
  printf("TachyonOptiX) ----------------------------------------\n");
  printf("TachyonOptiX)                     Misses: %lu\n", misses);
  printf("TachyonOptiX) Transmission Any-Hit Skips: %lu\n", transkips);
  printf("TachyonOptiX) ----------------------------------------\n");
  printf("TachyonOptiX)               Primary Rays: %lu\n", primaryrays);
  printf("TachyonOptiX)      Dir-Light Shadow Rays: %lu\n", shadowlights);
  printf("TachyonOptiX)             AO Shadow Rays: %lu\n", shadowao);
  printf("TachyonOptiX)          Transmission Rays: %lu\n", transrays);
  printf("TachyonOptiX)            Reflection Rays: %lu\n", reflrays);
  printf("TachyonOptiX) ----------------------------------------\n");
  printf("TachyonOptiX)                 Total Rays: %lu\n", totalrays);
  printf("TachyonOptiX)                 Total Rays: %g\n", totalrays * 1.0);
  if (time_ray_tracing > 0.0) {
    printf("TachyonOptiX)  Pure ray tracing rays/sec: %g\n", totalrays / time_ray_tracing);
  }
  double totalruntime = time_ray_tracing + time_ctx_AS_build;
  if (totalruntime > 0.0) {
    printf("TachyonOptiX) Overall effective rays/sec: %g\n", totalrays / totalruntime);
  } 
  printf("TachyonOptiX)\n");

  free(raystats1);
  free(raystats2);
#else
  printf("TachyonOptiX) Compiled without ray stats buffers!\n");
#endif
}




//
// A few structure padding/alignment/size diagnostic helper routines
//
void TachyonOptiX::print_internal_struct_info() { 
  printf("TachyonOptiX) internal data structure information\n"); 

  printf("TachyonOptiX) Hitgroup SBT record info:\n");
  printf("   SBT rec align size: %d b\n", OPTIX_SBT_RECORD_ALIGNMENT);
  printf("           total size: %d b\n", sizeof(HGRecord));   
  printf("          header size: %d b\n", sizeof(((HGRecord*)0)->header));
  printf("          data offset: %d b\n", offsetof(HGRecord, data));
  printf("            data size: %d b\n", sizeof(((HGRecord*)0)->data));
  printf("        material size: %d b\n", offsetof(HGRecord, data.cone) - offsetof(HGRecord, data.prim_color));
  printf("        geometry size: %d b\n", sizeof(HGRecord) - offsetof(HGRecord, data.trimesh));
  printf("\n");
  printf("    prim_color offset: %d b\n", offsetof(HGRecord, data.prim_color));
  printf(" uniform_color offset: %d b\n", offsetof(HGRecord, data.uniform_color));
  printf(" materialindex offset: %d b\n", offsetof(HGRecord, data.materialindex));
  printf("      geometry offset: %d b\n", offsetof(HGRecord, data.cone));

  printf("\n");
  printf("  geometry union size: %d b\n", sizeof(HGRecord) - offsetof(HGRecord, data.trimesh));
  printf("              cone sz: %d b\n", sizeof(((HGRecord*)0)->data.cone   ));
  printf("               cyl sz: %d b\n", sizeof(((HGRecord*)0)->data.cyl    ));
  printf("              ring sz: %d b\n", sizeof(((HGRecord*)0)->data.ring   ));
  printf("            sphere sz: %d b\n", sizeof(((HGRecord*)0)->data.sphere ));
  printf("           trimesh sz: %d b\n", sizeof(((HGRecord*)0)->data.trimesh));
  printf("   WASTED hitgroup sz: %d b\n", sizeof(HGRecord) - (sizeof(((HGRecord*)0)->header) + sizeof(((HGRecord*)0)->data)));
  printf("\n");
}



//
// geometry instance group management
//
int TachyonOptiX::create_geom_instance_group() {
  TachyonInstanceGroup g = {};
  sceneinstancegroups.push_back(g);
  return int(sceneinstancegroups.size()) - 1;
}

int TachyonOptiX::finalize_geom_instance_group(int idx) {
  TachyonInstanceGroup &g = sceneinstancegroups[idx];
  return 0;
}


int TachyonOptiX::destroy_geom_instance_group(int idx) {
  return 0;
}


#if 0
int TachyonOptiX::set_geom_instance_group_xforms(int idx, int n, float [][16]) {
  return 0;
}
#endif


//
// XXX short-term host API hacks to facilitate early bring-up and testing
//
void TachyonOptiX::add_conearray(ConeArray & newmodel, int materialidx) {
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  conearrays.push_back(newmodel);
  regen_optix_sbt=1;
}

void TachyonOptiX::add_curvearray(CurveArray & newmodel, int materialidx) {
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  curvearrays.push_back(newmodel);
  regen_optix_sbt=1;
}

void TachyonOptiX::add_cylarray(CylinderArray & newmodel, int materialidx) {
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  cyarrays.push_back(newmodel);
  regen_optix_sbt=1;
}


void TachyonOptiX::add_quadmesh(QuadMesh & newmodel, int materialidx) {
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  quadmeshes.push_back(newmodel);
  regen_optix_sbt=1;
}


void TachyonOptiX::add_ringarray(RingArray & newmodel, int materialidx) {
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  riarrays.push_back(newmodel);
  regen_optix_sbt=1;
}


void TachyonOptiX::add_spherearray(SphereArray & newmodel, int materialidx) {
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  sparrays.push_back(newmodel);
  regen_optix_sbt=1;
}


void TachyonOptiX::add_trimesh(TriangleMesh & newmodel, int materialidx) {
  if (!context_created)
    return;

  newmodel.materialindex = materialidx; // XXX overwrite hack...

  trimeshes.push_back(newmodel);
  regen_optix_sbt=1;
}


//
// Compiled-in PTX src, if available
// 
char *TachyonOptiX::internal_compiled_ptx_src(void) {
#if 1 && defined(TACHYON_INTERNAL_COMPILED_SRC)
  const char *ptxsrc = 
  #include "TachyonOptiXShaders.ptxinc"
  ;

  int len = strlen(ptxsrc);
  char *ptxstring = (char *) calloc(1, len + 1);
  strcpy(ptxstring, ptxsrc);
  return ptxstring;
#else
  return NULL;
#endif
}





