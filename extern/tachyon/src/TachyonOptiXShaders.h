/*
 * TachyonOptiXShaders.h - prototypes for OptiX PTX shader routines 
 *
 * (C) Copyright 2013-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: TachyonOptiXShaders.h,v 1.67 2022/04/19 02:54:24 johns Exp $
 *
 */

/**
 *  \file TachyonOptiXShaders.h
 *  \brief Tachyon ray tracing engine core routines and data structures
 *         compiled to PTX for runtime JIT to build complete ray tracing 
 *         pipelines.  Key data structures defined here are shared both by
 *         the compiled PTX core ray tracing routines, and by the host code
 *         that assembles the complete ray tracing pipeline and launches
 *         the pipeline kernels.
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

#ifndef TACHYONOPTIXSHADERS_H
#define TACHYONOPTIXSHADERS_H

#if 0
/// Compile-time flag for collection and reporting of ray statistics
#define TACHYON_RAYSTATS 1
#endif

#if OPTIX_VERSION >= 70300
#define TACHYON_OPTIXDENOISER 1
#endif

// enable use of geometry flags to accelerate various work
#define TACHYON_USE_GEOMFLAGS 1

//
// Constants shared by both host and device code
//
#define RT_DEFAULT_MAX 1e27f

//
// Beginning of OptiX data structures
//

// Enable reversed traversal of any-hit rays for shadows/AO.
// This optimization yields a 20% performance gain in many cases.
// #define USE_REVERSE_SHADOW_RAYS 1

// Use reverse rays by default rather than only when enabled interactively
// #define USE_REVERSE_SHADOW_RAYS_DEFAULT 1
enum RtShadowMode {
  RT_SHADOWS_OFF=0,          ///< shadows disabled
  RT_SHADOWS_ON=1,           ///< shadows on, std. impl.
  RT_SHADOWS_ON_REVERSE=2    ///< any-hit traversal reversal
};

enum RtDenoiserMode {
  RT_DENOISER_OFF=0,          ///< denoiser disabled
  RT_DENOISER_ON=1,           ///< denosier on, std. impl.
};

enum RtTonemapMode {
  RT_TONEMAP_CLAMP=0,         ///< only clamp the color values [0,1]
  RT_TONEMAP_ACES,            ///< ACES style approximation
  RT_TONEMAP_REINHARD,        ///< Reinhard style, color 
  RT_TONEMAP_REINHARD_EXT,    ///< "Extended" Reinhard style, color 
  RT_TONEMAP_REINHARD_EXT_L,  ///< "Extended" Reinhard style, luminance
  RT_TONEMAP_COUNT            ///< total count of ray types
};

enum RayType {
  RT_RAY_TYPE_RADIANCE=0,     ///< normal radiance rays
  RT_RAY_TYPE_SHADOW=1,       ///< shadow probe/AO rays
  RT_RAY_TYPE_COUNT           ///< total count of ray types
};

//
// OptiX 7.x geometry type-associated "hit kind" enums
//
enum RtHitKind {
  RT_HIT_HWTRIANGLE=1,       ///< RTX triangle

  // XXX custom prims offset to start at 2 (see below)
  RT_HIT_CONE,               ///< custom prim cone
  RT_HIT_CYLINDER,           ///< custom prim cyliner
  RT_HIT_QUAD,               ///< custom prim quadrilateral
  RT_HIT_RING,               ///< custom prim ring
  RT_HIT_SPHERE,             ///< custom prim sphere
  RT_HIT_CURVE,              ///< OptiX 7.x built-in curve prims
};      


// simplify runtime code for OptiX 7.0.0
#if defined(OPTIX_PRIMITIVE_TYPE_CUSTOM)
#define RT_CUSTPRIM    (OPTIX_PRIMITIVE_TYPE_CUSTOM << 16)
#define RT_TRI_BUILTIN (OPTIX_PRIMITIVE_TYPE_TRIANGLE << 16)
#else 
#define RT_CUSTPRIM    0 // OptiX 7.0.0
#define RT_TRI_BUILTIN OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE
#endif

enum RtMergedPrimKind {
  //
  // Tachyon custom primitives:
  //   XXX to prevent the triangle front/back hit kindl ow-bit masking scheme
  //       (see below) from interfering with the custom prim types, 
  //       the lowest byte of their enums must start at values above 0x02 
  RT_PRM_CONE       = RT_CUSTPRIM | RT_HIT_CONE,     ///< custom prim cone
  RT_PRM_CYLINDER   = RT_CUSTPRIM | RT_HIT_CYLINDER, ///< custom prim cylinder
  RT_PRM_QUAD       = RT_CUSTPRIM | RT_HIT_QUAD,     ///< custom prim quadrilateral
  RT_PRM_RING       = RT_CUSTPRIM | RT_HIT_RING,     ///< custom prim ring
  RT_PRM_SPHERE     = RT_CUSTPRIM | RT_HIT_SPHERE,   ///< custom prim sphere

  //
  // OptiX 7.x built-in primitives
  //
  // XXX we handle both front+back face triangles with a single case by 
  //     masking off the low bit from the hit kind value and the enums:
  RT_PRM_TRIANGLE   = RT_TRI_BUILTIN | 
                       (0xFE & OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE),

#if OPTIX_VERSION >= 70400
  RT_PRM_CATMULLROM = (OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM << 16), 
#endif
#if OPTIX_VERSION >= 70200
  RT_PRM_LINEAR     = (OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR << 16),
#endif
};      


// Enums used for custom primitive PGM indexing in SBT + GAS
enum RtCustPrim { 
  RT_CUST_PRIM_CONE=0,            ///< cone SBT index multiplier
  RT_CUST_PRIM_CYLINDER,          ///< cylinder SBT index multiplier
  RT_CUST_PRIM_QUAD,              ///< quad SBT index multiplier
  RT_CUST_PRIM_RING,              ///< ring SBT index multiplier
  RT_CUST_PRIM_SPHERE,            ///< sphere SBT index multiplier
  RT_CUST_PRIM_COUNT              ///< total count of SBT geometric multipliers
};

enum RtColorSpace {
  RT_COLORSPACE_LINEAR=0,         ///< linear rgba, gamma 1.0
  RT_COLORSPACE_sRGB=1,           ///< Adobe sRGB (gamma 2.2)
  RT_COLORSPACE_COUNT             ///< total count of available colorspaces
};

enum RtTexFlags {
  RT_TEX_NONE=0,                  ///< default behavior
  RT_TEX_COLORSPACE_LINEAR = 0,   ///< linear rgba, gamma 1.0
  RT_TEX_COLORSPACE_sRGB   = 0x1, ///< Adobe sRGB (gamma 2.2)
  RT_TEX_ALPHA             = 0x2  ///< enable cutout/transparency
};

enum RtMatFlags {
  RT_MAT_NONE     = 0,            ///< default behavior
  RT_MAT_ALPHA    = 0x1,          ///< enable alpha transparency
  RT_MAT_TEXALPHA = 0x2,          ///< enable tex cutout transparency 
};


//
// Images, Materials, Textures...
//

/// structure containing Tachyon texture (only used on host side)
typedef struct {
  int texflags;                   ///< linear/sRGB colorspace | texturing flags
  float3 texgen_origin;           ///< world coordinate texgen origin
  float3 texgen_uaxis;            ///< world coordinate texgen U axis
  float3 texgen_vaxis;            ///< world coordinate texgen V axis
  float3 texgen_waxis;            ///< world coordinate texgen W axis
  cudaArray_t d_img;              ///< GPU allocated image buffer
  cudaTextureObject_t tex;        ///< texture, non-zero if valid
  int userindex;                  ///< material user index, positive if valid
} rt_texture;


/// structure containing Tachyon material properties
typedef struct {
  float opacity;                  ///< surface opacity 
  float ambient;                  ///< constant ambient light factor
  float diffuse;                  ///< diffuse reflectance coefficient
  float specular;                 ///< specular reflectance coefficient
  float shininess;                ///< specular highlight size (exponential)
  float reflectivity;             ///< mirror reflectance coefficient
  float outline;                  ///< outline shading coefficient 
  float outlinewidth;             ///< width of outline shading effect
  int transmode;                  ///< transparency behavior
  cudaTextureObject_t tex;        ///< texture, non-zero if valid
  int matflags;                   ///< alpha/cutout transparency flags
  int userindex;                  ///< material user index, positive if valid
} rt_material;


//
// Lighting data structures
//
typedef struct {
  float3 dir;                     ///< directional light direction
//  float3 color; // not yet used
} rt_directional_light;

typedef struct {
  float3 pos;                     ///< point light position
//  float3 color; // not yet used
} rt_positional_light;



//
// Shader Binding Table (SBT) Data Structures
//
struct ConeArraySBT {
  float3 *base;
  float3 *apex;
  float  *baserad;
  float  *apexrad;
};

struct CurveArraySBT {
  float3 *vertices;
  float  *vertradii;
  int    *segindices;
};

struct CylinderArraySBT {
  float3 *start;
  float3 *end;
  float  *radius;
};

struct QuadMeshSBT {
  float3 *vertices;
  int4   *indices;
  float3 *normals;
  uint4  *packednormals;          ///< packed normals: ng [n0 n1 n2]
  float3 *vertcolors3f; 
  uchar4 *vertcolors4u;           ///< unsigned char color representation
};

struct RingArraySBT {
  float3 *center;
  float3 *norm;
  float  *inrad;
  float  *outrad; 
};

struct SphereArraySBT {
  float4 *PosRadius;              ///< X,Y,Z,Radius packed for coalescing
};

struct TriMeshSBT {
  float3 *vertices;
  int3   *indices;
  float3 *normals;
  uint4  *packednormals;          ///< packed normals: ng [n0 n1 n2]
  float3 *vertcolors3f; 
  uchar4 *vertcolors4u;           ///< unsigned char color representation
  float2 *tex2d;                  ///< 2-D texture coordinate buffer
  float3 *tex3d;                  ///< 3-D texture coordinate buffer
};

struct GeomSBTHG {
#if defined(TACHYON_USE_GEOMFLAGS)
  // XXX alpha/opacity AH optimization flags to skip material fetching
  int geomflags; 
#endif
  float3 *prim_color;             ///< optional per-primitive color array
  float3 uniform_color;           ///< uniform color for entire sphere array
  int materialindex;              ///< material index for this array

  union {
    ConeArraySBT cone;
    CurveArraySBT curve;
    CylinderArraySBT cyl;
    QuadMeshSBT quadmesh;
    RingArraySBT ring;
    SphereArraySBT sphere;
    TriMeshSBT trimesh;
  };
};



/// SBT record for a hitgroup program
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HGRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  GeomSBTHG data;
};

/// Store all hitgroup records for a given geometry together for 
/// simpler dynamic updates.  At present, we have pairs of records,
/// for radiance and shadow rayss.  Records differ only in their header.
/// Each HGRecordGroup contains RT_RAY_TYPE_COUNT HGRecords, so when querying
/// the size of any vector containers or other data structures to count total
/// hitgroup records, we need to remember to multiply by RT_RAY_TYPE_COUNT.
struct HGRecordGroup {
  HGRecord radiance;
  HGRecord shadow;
};


/// SBT record for an exception program
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) ExceptionRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};

/// SBT record for a raygen program
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};

/// SBT record for a miss program
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};


/// Tachyon OptiX global launch parameter structure containing the active
/// camera, framebuffer, materials, and any global scene parameters required
/// for shading.
struct tachyonLaunchParams {
  struct {
    int2 size;                    ///< framebuffer size
    int subframe_index;           ///< accumulation subframe index
    int update_colorbuffer;       ///< accumulation copyout flag 
    int fb_clearall;              ///< clear/overwrite all FB components
    int colorspace;               ///< output colorspace
    int tonemap_mode;             ///< output tone mapping mode
    float tonemap_exposure;       ///< tone mapping exposure gain parameter
    uchar4 *framebuffer;          ///< 8-bit unorm RGBA framebuffer

#if defined(TACHYON_OPTIXDENOISER)
    // buffers required for denoising 
    float4 *denoiser_colorbuffer; ///< linear, normalized 32-bit FP RGBA buffer
    int denoiser_enabled;         ///< flag to write to denoiser color buffer
#endif

    float accum_normalize;        ///< precalc 1.0f / subframe_index
    float4 *accum_buffer;         ///< 32-bit FP RGBA accumulation buffer

#if defined(TACHYON_RAYSTATS)
    uint4 *raystats1_buffer;      ///< x=prim, y=shad-dir, z=shad-ao, w=miss
    uint4 *raystats2_buffer;      ///< x=trans, y=trans-skip, z=?, w=refl
#endif
  } frame;

  struct {
    float3 bg_color;              ///< miss background color
    float3 bg_color_grad_top;     ///< miss background gradient (top)
    float3 bg_color_grad_bot;     ///< miss background gradient (bottom)
    float3 bg_grad_updir;         ///< miss background gradient up direction
    float  bg_grad_topval;        ///< miss background gradient top value
    float  bg_grad_botval;        ///< miss background gradient bottom value
    float  bg_grad_invrange;      ///< miss background gradient inverse range
    float  bg_grad_noisemag;      ///< miss background gradient noise magnitude
    int    fog_mode;              ///< fog type (or off)
    float  fog_start;             ///< radial/linear fog start distance
    float  fog_end;               ///< radial/linear fog end/max distance
    float  fog_density;           ///< exponential fog density
    float  epsilon;               ///< global epsilon value
  } scene;

  struct {
    int shadows_enabled;          ///< global shadow flag
    int ao_samples;               ///< number of AO samples per AA ray
    float ao_lightscale;          ///< 2.0f/float(ao_samples)
    float ao_ambient;             ///< AO ambient factor
    float ao_direct;              ///< AO direct lighting scaling factor
    float ao_maxdist;             ///< AO maximum occlusion distance
    int headlight_mode;           ///< Extra VR camera-located headlight
    int num_dir_lights;           ///< directional light count
    float3 *dir_lights;           ///< list of directional light directions
    int num_pos_lights;           ///< positional light count
    float3 *pos_lights;           ///< list of positional light positions
  } lights;

  struct {
    float3 pos;                   ///< camera position
    float3 U;                     ///< camera orthonormal U (right) axis
    float3 V;                     ///< camera orthonormal V (up) axis
    float3 W;                     ///< camera orthonormal W (view) axis
    float zoom;                   ///< camera zoom factor 
    int   dof_enabled;            ///< DoF (defocus blur) on/off
    float dof_aperture_rad;       ///< DoF (defocus blur) aperture radius
    float dof_focal_dist;         ///< DoF focal plane distance
    int   stereo_enabled;         ///< stereo rendering on/off
    float stereo_eyesep;          ///< stereo eye separation, in world coords
    float stereo_convergence_dist; ///< stereo convergence distance (world)
  } cam;

  // VR HMD fade+clipping plane/sphere
  int clipview_mode;              ///< VR clipping view on/off
  float clipview_start;           ///< clipping sphere/plane start coord
  float clipview_end;             ///< clipping sphere/plane end coord

  rt_material *materials;         ///< device memory material array

  int max_depth;                  ///< global max ray tracing recursion depth
  int max_trans;                  ///< max transparent surface crossing count
  int aa_samples;                 ///< AA samples per launch

  OptixTraversableHandle traversable; ///< global OptiX scene traversable handle
};



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

//
// Eliminate compiler warnings about any unused functions
//
//#pragma push
// suppress "function was declared but never referenced warning"
//#pragma nv_diag_suppress 177


//
// Vector math helper routines
//

//
// float2 vector operators
//
inline __host__ __device__ float2 operator+(const float2& a, const float2& b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ float2 operator+(const float2& a, const float s) {
  return make_float2(a.x + s, a.y + s);
}

inline __host__ __device__ float2 operator-(const float2& a, const float2& b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ float2 operator-(const float2& a, const float s) {
  return make_float2(a.x - s, a.y - s);
}

inline __host__ __device__ float2 operator-(const float s, const float2& a) {
  return make_float2(s - a.x, s - a.y);
}

inline __host__ __device__ float2 operator*(const float2& a, const float2& b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ float2 operator*(const float s, const float2& a) {
  return make_float2(a.x * s, a.y * s);
}

inline __host__ __device__ float2 operator*(const float2& a, const float s) {
  return make_float2(a.x * s, a.y * s);
}

inline __host__ __device__ void operator*=(float2& a, const float s) {
  a.x *= s; a.y *= s;
}

inline __host__ __device__ float2 operator/(const float s, const float2& a) {
  return make_float2(s/a.x, s/a.y);
}



//
// float3 vector operators
//
inline __host__ __device__ float3 make_float3(const float s) {
  return make_float3(s, s, s);
}

inline __host__ __device__ float3 make_float3(const float4& a) {
  return make_float3(a.x, a.y, a.z);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(const float3& a, const float3 &b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __host__ __device__ float3 operator-(const float3& a) {
  return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ void operator+=(float3& a, const float3& b) {
  a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __host__ __device__ float3 operator+(const float3& a, const float &b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float3 operator*(const float3& a, const float3 &b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator*(float s, const float3 &a) {
  return make_float3(s * a.x, s * a.y, s * a.z);
}

inline __host__ __device__ float3 operator*(const float3 &a, const float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ void operator*=(float3& a, const float s) {
  a.x *= s; a.y *= s; a.z *= s;
}

inline __host__ __device__ void operator*=(float3& a, const float3 &b) {
  a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __host__ __device__ float3 operator/(const float3 &a, const float3 &b) {
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}


//
// float4 vector operators
//
inline __host__ __device__ float4 make_float4(const float3 &a, const float &b) {
  return make_float4(a.x, a.y, a.z, b);
}

inline __host__ __device__ float4 make_float4(const float a) {
  return make_float4(a, a, a, a);
}

inline __host__ __device__ void operator+=(float4& a, const float4& b) {
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

inline __host__ __device__ float4 operator*(const float4& a, const float s) {
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline __host__ __device__ void operator*=(float4& a, const float &b) {
  a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}


//
// operators with subsequent type conversions
//
inline __host__ __device__ float3 operator*(char4 a, const float s) {
  return make_float3(s * a.x, s * a.y, s * a.z);
}

inline __host__ __device__ float3 operator*(uchar4 a, const float s) {
  return make_float3(s * a.x, s * a.y, s * a.z);
}


//
// math fctns...
//
inline __host__ __device__ float3 fabsf(const float3& a) {
  return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

inline __host__ __device__ float3 fmaxf(const float3& a, const float3& b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline __host__ __device__ float fmaxf(const float3& a) {
  return fmaxf(fmaxf(a.x, a.y), a.z);
}

inline __host__ __device__ float dot(const float3 & a, const float3 & b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ float dot(const float4 & a, const float4 & b) {
  return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

inline __host__ __device__ float length(const float3 & v) {
  return sqrtf(dot(v, v));
}


/// Normalize input vector to unit length.
inline __host__ __device__ float3 normalize(const float3 & v) {
#if defined(__CUDACC__) || defined(__NVCC__)
  float invlen = rsqrtf(dot(v, v));
#else
  float invlen = 1.0f / sqrtf(dot(v, v));
#endif
  float3 out;
  out.x = v.x * invlen;
  out.y = v.y * invlen;
  out.z = v.z * invlen;
  return out;
}


/// Normalize input vector to unit length, and return its original length.
inline __host__ __device__ float3 normalize_len(const float3 v, float &l) {
  l = length(v);
  float invlen = 1.0f / l;
  float3 out;
  out.x = v.x * invlen;
  out.y = v.y * invlen;
  out.z = v.z * invlen;
  return out;
}


/// Normalize input vector to unit length, and return the 
/// reciprocal of its original length.
inline __host__ __device__ float3 normalize_invlen(const float3 v, float &invlen) {
#if defined(__CUDACC__) || defined(__NVCC__)
  invlen = rsqrtf(dot(v, v));
#else
  invlen = 1.0f / sqrtf(dot(v, v));
#endif
  float3 out;
  out.x = v.x * invlen;
  out.y = v.y * invlen;
  out.z = v.z * invlen;
  return out;
}


/// calculate the cross product between vectors a and b.
inline __host__ __device__ float3 cross(const float3 & a, const float3 & b) {
  float3 out;
  out.x =  a.y * b.z - b.y * a.z;
  out.y = -a.x * b.z + b.x * a.z;
  out.z =  a.x * b.y - b.x * a.y;
  return out;
}


/// calculate reflection direction from incident direction i,
/// and surface normal n.
inline __host__ __device__ float3 reflect(const float3& i, const float3& n) {
  return i - 2.0f * n * dot(n, i);
}


/// Ensure that an interpolated surface normal n faces in the same direction
/// as dictated by a geometric normal nref, as seen from incident vector i.
inline __host__ __device__ float3 faceforward(const float3& n, const float3& i,
                                              const float3& nref) {
  return n * copysignf(1.0f, dot(i, nref));
}


//
// PRNGs
//

//
// Various random number routines
//   https://en.wikipedia.org/wiki/List_of_random_number_generators
//

#define UINT32_RAND_MAX     4294967296.0f      // max uint32 random value
#define UINT32_RAND_MAX_INV 2.3283064365e-10f  // normalize uint32 RNs

//
// Survey of parallel RNGS suited to GPUs, by L'Ecuyer et al.:
//   Random numbers for parallel computers: Requirements and methods,
//   with emphasis on GPUs.
//   Pierre L'Ecuyer, David Munger, Boris Oreshkina, and Richard Simard.
//   Mathematics and Computers in Simulation 135:3-17, 2017.
//   https://doi.org/10.1016/j.matcom.2016.05.005
//
// Counter-based RNGs introduced by Salmon @ D.E. Shaw Research:
//   "Parallel random numbers: as easy as 1, 2, 3", by Salmon et al.,
//    D. E. Shaw Research:
//   http://doi.org/10.1145/2063384.2063405
//   https://www.thesalmons.org/john/random123/releases/latest/docs/index.html
//   https://en.wikipedia.org/wiki/Counter-based_random_number_generator_(CBRNG)
//


//
// Quick and dirty 32-bit LCG random number generator [Fishman 1990]:
//   A=1099087573 B=0 M=2^32
//   Period: 10^9
// Fastest gun in the west, but fails many tests after 10^6 samples,
// and fails all statistics tests after 10^7 samples.
// It fares better than the Numerical Recipes LCG.  This is the fastest
// power of two rand, and has the best multiplier for 2^32, found by
// brute force[Fishman 1990].  Test results:
//   http://www.iro.umontreal.ca/~lecuyer/myftp/papers/testu01.pdf
//   http://www.shadlen.org/ichbin/random/
//
static __host__ __device__ __inline__
uint32_t qnd_rng(uint32_t &idum) {
  idum *= 1099087573;
  return idum; // already 32-bits, no need to mask result
}


//
// Middle Square Weyl Sequence ("msws")
//   This is an improved variant of von Neumann's middle square RNG
//   that uses Weyl sequences to provide a long period.  Claimed as
//   fastest traditional seeded RNG that passes statistical tests.
//   V5: Bernard Widynski, May 2020.
//   https://arxiv.org/abs/1704.00358
//
//   Additional notes and commentary:
//     https://en.wikipedia.org/wiki/Middle-square_method
//     https://pthree.org/2018/07/30/middle-square-weyl-sequence-prng/
//
//   Reported to passes both BigCrush and PractRand tests:
//     "An Empirical Study of Non-Cryptographically Secure
//      Pseudorandom Number Generators," M. Singh, P. Singh and P. Kumar,
//      2020 International Conference on Computer Science, Engineering
//      and Applications (ICCSEA), 2020,
//      http://doi.org/10.1109/ICCSEA49143.2020.9132873
//
static __host__ __device__ __inline__
uint32_t msws_rng(uint64_t &x, uint64_t &w) {
  const uint64_t s = 0xb5ad4eceda1ce2a9;
  x *= x;                // square the value per von Neumann's RNG
  w += s;                // add in Weyl sequence for longer period
  x += w;                // apply to x
  x = (x>>32) | (x<<32); // select "middle square" as per von Neumann's RNG
  return x;              // implied truncation to lower 32-bit result
}



//
// Squares: A Fast Counter-Based RNG
//   This is a counter-based RNG based on John von Neumann's
//   Middle Square RNG, with the Weyl sequence added to provide a long period.
//   V3: Bernard Widynski, Nov 2020.
//   https://arxiv.org/abs/2004.06278
//
// This RNG claims to outperform all of the original the counter-based RNGs
// in "Parallel random numbers: as easy as 1, 2, 3",
//   by Salmon et al., http://doi.org/10.1145/2063384.2063405
//   https://en.wikipedia.org/wiki/Counter-based_random_number_generator_(CBRNG)
// That being said, key generation technique is important in this case.
//
#define SQUARES_RNG_KEY1 0x1235d7fcb4dfec21  // a few good keys...
#define SQUARES_RNG_KEY2 0x418627e323f457a1  // a few good keys...
#define SQUARES_RNG_KEY3 0x83fc79d43614975f  // a few good keys...
#define SQUARES_RNG_KEY4 0xc62f73498cb654e3  // a few good keys...

// Template to allow compile-time selection of number of rounds (2, 3, 4).
// Roughly 5 integer ALU operations per round, 4 rounds is standard.
template<unsigned int ROUNDS> static __host__ __device__ __inline__
uint32_t squares_rng(uint64_t counter, uint64_t key) {
  uint64_t x, y, z;
  y = x = counter * key;
  z = x + key;

  x = x*x + y;                // round 1, middle square, add Weyl seq
  x = (x>>32) | (x<<32);      // round 1, bit rotation

  x = x*x + z;                // round 2, middle square, add Weyl seq
  if (ROUNDS == 2) {
    return x >> 32;           // round 2, upper 32-bits are bit-rotated result
  } else {
    x = (x>>32) | (x<<32);    // round 2, bit rotation

    x = x*x + y;              // round 3, middle square, add Weyl seq
    if (ROUNDS == 3) {
      return x >> 32;         // round 3, upper 32-bits are bit-rotated result
    } else {
      x = (x>>32) | (x<<32);  // round 3, bit rotation

      x = x*x + z;            // round 4, middle square, add Weyl seq
      return x >> 32;         // round 4, upper 32-bits are bit-rotated result
    }
  }
}


//
// Hashing based PRNGs
//


//
// TEA, a tiny encryption algorithm.
// D. Wheeler and R. Needham, 2nd Intl. Workshop Fast Software Encryption,
// LNCS, pp. 363-366, 1994.
//
// GPU Random Numbers via the Tiny Encryption Algorithm
// F. Zafar, M. Olano, and A. Curtis.
// HPG '10 Proceedings of the Conference on High Performance Graphics,
// pp. 133-141, 2010.
// https://dl.acm.org/doi/10.5555/1921479.1921500
//
// Tea has avalanche effect in output from one bit input delta after 6 rounds
//
template<unsigned int ROUNDS> static __host__ __device__ __inline__
unsigned int tea(uint32_t val0, uint32_t val1) {
  uint32_t v0 = val0;
  uint32_t v1 = val1;
  uint32_t s0 = 0;

  for (unsigned int n = 0; n < ROUNDS; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}



//
// QRNGs
//


//
// Low discrepancy sequences based on the Golden Ratio, described in
// Golden Ratio Sequences for Low-Discrepancy Sampling,
// Colas Schretter and Leif Kobbelt, pp. 95-104, JGT 16(2), 2012.
//
// Other useful online references:
//   http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
//

// compute Nth value in 1-D sequence
static __device__ __inline__
float goldenratioseq1d(int n) {
  const double g = 1.61803398874989484820458683436563;
  const double a1 = 1.0 / g;
  const double seed = 0.5;
  double ngold;
  ngold = (seed + (a1 * n));
  return ngold - trunc(ngold);
}


// incremental formulation to obtain the next value in the sequence
static __device__ __inline__
void goldenratioseq1d_incr(float &x) {
  const double g = 1.61803398874989484820458683436563;
  const double a1 = 1.0 / g;
  float ngold = x + a1;
  x = ngold - truncf(ngold);
}


// compute Nth point in 2-D sequence
static __device__ __inline__
void goldenratioseq2d(int n, float2 &xy) {
  const double g = 1.32471795724474602596;
  const double a1 = 1.0 / g;
  const double a2 = 1.0 / (g*g);
  const double seed = 0.5;
  double ngold;

  ngold = (seed + (a1 * n));
  xy.x = (float) (ngold - trunc(ngold));

  ngold = (seed + (a2 * n));
  xy.y = (float) (ngold - trunc(ngold));
}


// incremental formulation to obtain the next value in the sequence
static __device__ __inline__
void goldenratioseq2d_incr(float2 &xy) {
  const float g = 1.32471795724474602596;
  const float a1 = 1.0 / g;
  const float a2 = 1.0 / (g*g);
  float ngold;

  ngold = xy.x + a1;
  xy.x = (ngold - trunc(ngold));

  ngold = xy.y + a2;
  xy.y = (ngold - trunc(ngold));
}


// compute Nth point in 3-D sequence
static __device__ __inline__
void goldenratioseq3d(int n, float3 &xyz) {
  const double g = 1.22074408460575947536;
  const double a1 = 1.0 / g;
  const double a2 = 1.0 / (g*g);
  const double a3 = 1.0 / (g*g*g);
  const double seed = 0.5;
  double ngold;

  ngold = (seed + (a1 * n));
  xyz.x = (float) (ngold - trunc(ngold));

  ngold = (seed + (a2 * n));
  xyz.y = (float) (ngold - trunc(ngold));

  ngold = (seed + (a3 * n));
  xyz.z = (float) (ngold - trunc(ngold));
}


// incremental formulation to obtain the next value in the sequence
static __device__ __inline__
void goldenratioseq3d_incr(float3 &xyz) {
  const float g = 1.22074408460575947536;
  const float a1 = 1.0 / g;
  const float a2 = 1.0 / (g*g);
  const float a3 = 1.0 / (g*g*g);
  float ngold;

  ngold = xyz.x + a1;
  xyz.x = (ngold - trunc(ngold));

  ngold = xyz.y + a2;
  xyz.y = (ngold - trunc(ngold));

  ngold = xyz.z + a3;
  xyz.z = (ngold - trunc(ngold));
}


// compute Nth point in 4-D sequence
static __device__ __inline__
void goldenratioseq4d(int n, float2 &xy1, float2 &xy2) {
  const double g = 1.167303978261418740;
  const double a1 = 1.0 / g;
  const double a2 = 1.0 / (g*g);
  const double a3 = 1.0 / (g*g*g);
  const double a4 = 1.0 / (g*g*g*g);
  const double seed = 0.5;
  double ngold;

  ngold = (seed + (a1 * n));
  xy1.x = (float) (ngold - trunc(ngold));

  ngold = (seed + (a2 * n));
  xy1.y = (float) (ngold - trunc(ngold));

  ngold = (seed + (a3 * n));
  xy2.x = (float) (ngold - trunc(ngold));

  ngold = (seed + (a4 * n));
  xy2.y = (float) (ngold - trunc(ngold));
}


// incremental formulation to obtain the next value in the sequence
static __device__ __inline__
void goldenratioseq4d_incr(float2 &xy1, float2 &xy2) {
  const double g = 1.167303978261418740;
  const float a1 = 1.0 / g;
  const float a2 = 1.0 / (g*g);
  const float a3 = 1.0 / (g*g*g);
  const float a4 = 1.0 / (g*g*g*g);
  float ngold;

  ngold = xy1.x + a1;
  xy1.x = (ngold - trunc(ngold));

  ngold = xy1.y + a2;
  xy1.y = (ngold - trunc(ngold));

  ngold = xy2.x + a3;
  xy2.x = (ngold - trunc(ngold));

  ngold = xy2.y + a4;
  xy2.y = (ngold - trunc(ngold));
}



//
// stochastic sampling helper routines
//

// Generate an offset to jitter AA samples in the image plane
static __device__ __inline__
void jitter_offset2f(unsigned int &pval, float2 &xy) {
  xy.x = (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 0.5f;
  xy.y = (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 0.5f;
}


// Generate an offset to jitter DoF samples in the Circle of Confusion
static __device__ __inline__
void jitter_disc2f(unsigned int &pval, float2 &xy, float radius) {
#if 1
  // Since the GPU RT currently uses super cheap/sleazy LCG RNGs,
  // it is best to avoid using sample picking, which can fail if
  // we use a multiply-only RNG and we hit a zero in the PRN sequence.
  // The special functions are slow, but have bounded runtime and
  // minimal branch divergence.
  float   r=(qnd_rng(pval) * UINT32_RAND_MAX_INV);
  float phi=(qnd_rng(pval) * UINT32_RAND_MAX_INV) * 2.0f * M_PIf;
  __sincosf(phi, &xy.x, &xy.y); // fast approximation
  xy *= sqrtf(r) * radius;
#else
  // Pick uniform samples that fall within the disc --
  // this scheme can hang in an endless loop if a poor quality
  // RNG is used and it gets stuck in a short PRN sub-sequence
  do {
    xy.x = 2.0f * (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 1.0f;
    xy.y = 2.0f * (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 1.0f;
  } while ((xy.x*xy.x + xy.y*xy.y) > 1.0f);
  xy *= radius;
#endif
}


// Generate an offset to jitter AA samples in the image plane using
// a low-discrepancy sequence
static __device__ __inline__
void jitter_offset2f_qrn(float2 qrnxy, float2 &xy) {
  xy = qrnxy - make_float2(0.5f, 0.5f);
}


// Generate an offset to jitter DoF samples in the Circle of Confusion,
// using low-discrepancy sequences based on the Golden Ratio
static __device__ __inline__
void jitter_disc2f_qrn(float2 &qrnxy, float2 &xy, float radius) {
  goldenratioseq2d_incr(qrnxy);
  float   r=qrnxy.x;
  float phi=qrnxy.y * 2.0f * M_PIf;
  __sincosf(phi, &xy.x, &xy.y); // fast approximation
  xy *= sqrtf(r) * radius;
}


//
// Protect functions that are only GPU-callable, e.g., those that
// use GPU-specific intrinsics such as __saturatef() or others.
//
#if defined(TACHYON_INTERNAL)

// Generate a randomly oriented ray
static __device__ __inline__
void jitter_sphere3f(unsigned int &pval, float3 &dir) {
#if 1
  //
  // Use GPU fast/approximate math routines
  //
  /* Archimedes' cylindrical projection scheme       */
  /* generate a point on a unit cylinder and project */
  /* back onto the sphere.  This approach is likely  */
  /* faster for SIMD hardware, despite the use of    */
  /* transcendental functions.                       */
  float u1 = qnd_rng(pval) * UINT32_RAND_MAX_INV;
  dir.z = 2.0f * u1 - 1.0f;
  float R = __fsqrt_rn(1.0f - dir.z*dir.z);  // fast approximation
  float u2 = qnd_rng(pval) * UINT32_RAND_MAX_INV;
  float phi = 2.0f * M_PIf * u2;
  float sinphi, cosphi;
  __sincosf(phi, &sinphi, &cosphi); // fast approximation
  dir.x = R * cosphi;
  dir.y = R * sinphi;
#elif 1
  /* Archimedes' cylindrical projection scheme       */
  /* generate a point on a unit cylinder and project */
  /* back onto the sphere.  This approach is likely  */
  /* faster for SIMD hardware, despite the use of    */
  /* transcendental functions.                       */
  float u1 = qnd_rng(pval) * UINT32_RAND_MAX_INV;
  dir.z = 2.0f * u1 - 1.0f;
  float R = sqrtf(1.0f - dir.z*dir.z);

  float u2 = qnd_rng(pval) * UINT32_RAND_MAX_INV;
  float phi = 2.0f * M_PIf * u2;
  float sinphi, cosphi;
  sincosf(phi, &sinphi, &cosphi);
  dir.x = R * cosphi;
  dir.y = R * sinphi;
#else
  /* Marsaglia's uniform sphere sampling scheme           */
  /* In order to correctly sample a sphere, using rays    */
  /* generated randomly within a cube we must throw out   */
  /* direction vectors longer than 1.0, otherwise we'll   */
  /* oversample the corners of the cube relative to       */
  /* a true sphere.                                       */
  float len;
  float3 d;
  do {
    d.x = (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 0.5f;
    d.y = (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 0.5f;
    d.z = (qnd_rng(pval) * UINT32_RAND_MAX_INV) - 0.5f;
    len = dot(d, d);
  } while (len > 0.250f);
  float invlen = rsqrtf(len);

  /* finish normalizing the direction vector */
  dir = d * invlen;
#endif
}


//
// Spherical Fibonacci pattern to create a uniformly
// distributed sample pattern on a sphere.
//   Spherical Fibonacci mapping.
//   B. Keinert, M. Innmann, M. Sänger, and M. Stamminger.
//   ACM Transactions on Graphics, 34:193:1-193:7, 2015.
//   http://doi.org/10.1145/2816795.2818131
//
static __device__ __inline__
float3 sphericalFibonacci(float i, float totaln) {
  const float PHI = sqrtf(5.0f) * 0.5f + 0.5f;
  float fraction = (i * (PHI - 1.0f)) - floorf(i * (PHI - 1.0f));

  float phi = 2.0f * M_PI * fraction;
  float cosTheta = 1.0f - (2.0f * i + 1.0f) * (1.0f / totaln);
  float sinTheta = sqrt(__saturatef(1.0f - cosTheta * cosTheta));

  float cosPhi, sinPhi;
  sincosf(phi, &cosPhi, &sinPhi);
  return make_float3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
}

#endif // TACHYON_INTERNAL



//
// Convert between 2-D planar coordinates and an octahedral mapping.
// This is useful for both omnidirectional cameras and image formats,
// and for surface normal compression/quantization.
//
// This implementation follows the method described here:
//   "A Survey of Efficient Representations for Independent Unit Vectors",
//   Cigolle et al., J. Computer Graphics Techniques 3(2), 2014.
//   http://jcgt.org/published/0003/02/01/
//
// UNORM: convert internal SNORM output range [-1,1] to UNORM [0,1] range
//        UNORM mode costs extra instructions
//
template <int UNORM>
static __host__ __device__ __inline__ float2 OctEncode(float3 n) {
  const float invL1Norm = 1.0f / (fabsf(n.x) + fabsf(n.y) + fabsf(n.z));
  float2 projected;
  if (n.z < 0.0f) {
    projected = 1.0f - make_float2(fabsf(n.y), fabsf(n.x)) * invL1Norm;
    projected.x = copysignf(projected.x, n.x);
    projected.y = copysignf(projected.y, n.y);
  } else {
    projected = make_float2(n.x, n.y) * invL1Norm;
  }

  // convert from SNORM to UNORM
  if (UNORM)
    projected = projected * 0.5f + 0.5f; // convert to UNORM range [0,1]

  return projected;
}


//
// XXX TODO: implement a high-precision OctPEncode() variant, based on
//           floored snorms and an error minimization scheme using a
//           comparison of internally decoded values for least error
//

//
// Direct adaptation from Cigolle et al, with optional UNORM mode.
//
// UNORM: convert from UNORM input domain [0,1] to internal SNORM [-1,1] domain
//        UNORM mode costs extra instructions
//
template <int UNORM>
static __host__ __device__ __inline__ float3 OctDecode(float2 projected) {
  // convert from UNORM input domain to native SNORM internal domain
  if (UNORM)
    projected *= 2.0f - 1.0f; // convert to SNORM range [-1,1]

  float3 n = make_float3(projected.x,
                         projected.y,
                         1.0f - (fabsf(projected.x) + fabsf(projected.y)));
  if (n.z < 0.0f) {
    float oldX = n.x;
    n.x = copysignf(1.0f - fabsf(n.y), oldX);
    n.y = copysignf(1.0f - fabsf(oldX), n.y);
  }

  return n;
}


//
// Protect functions that are only GPU-callable, e.g., those that
// use GPU-specific intrinsics such as __saturatef() or others.
//
#if defined(TACHYON_INTERNAL)

//
// Faster version by Rune Stubbe (2017) that avoids branching in decode:
//   https://twitter.com/Stubbesaurus/status/937994790553227264
//   https://twitter.com/Stubbesaurus/status/937994790553227264/photo/1
// https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding
// Another variant:
// http://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
// UNORM: convert from UNORM input domain [0,1] to internal SNORM [-1,1] domain
//        UNORM mode costs extra instructions
//
template <int UNORM>
static __device__ __inline__ float3 OctDecode_fast(float2 projected) {
  // convert from UNORM input domain to native SNORM internal domain
  if (UNORM)
    projected *= 2.0f - 1.0f; // convert to SNORM range [-1,1]

  float3 n = make_float3(projected.x,
                         projected.y,
                         1.0f - fabsf(projected.x) - fabsf(projected.y));
  float t = __saturatef(-n.z); // or max(-n.z, 0.0)
  n.x += (n.x >= 0.0f) ? -t : t;
  n.y += (n.y >= 0.0f) ? -t : t;

  return n;
}

#endif // TACHYON_INTERNAL


//
// Methods for packing normals into a 4-byte quantity, such as a
// [u]int or [u]char4, and similar.  See JCGT article by Cigolle et al.,
// "A Survey of Efficient Representations for Independent Unit Vectors",
// J. Computer Graphics Techniques 3(2), 2014.
// http://jcgt.org/published/0003/02/01/
//
static __host__ __device__ __inline__ uint convfloat2uint32(float2 f2) {
  f2 = f2 * 0.5f + 0.5f;
  uint packed;
  packed = ((uint) (f2.x * 65535)) | ((uint) (f2.y * 65535) << 16);
  return packed;
}

static __host__ __device__ __inline__ float2 convuint32float2(uint packed) {
  float2 f2;
  f2.x = (float)((packed      ) & 0x0000ffff) / 65535;
  f2.y = (float)((packed >> 16) & 0x0000ffff) / 65535;
  return f2 * 2.0f - 1.0f;
}



#if 1

//
// oct32: 32-bit octahedral normal encoding using [su]norm16x2 quantization
// Meyer et al., "On Floating Point Normal Vectors", In Proc. 21st
// Eurographics Conference on Rendering.
//   http://dx.doi.org/10.1111/j.1467-8659.2010.01737.x
//
static __host__ __device__ __inline__ uint packNormal(const float3& normal) {
  float2 octf2 = OctEncode<0>(normal);
  return convfloat2uint32(octf2);
}

static __host__ __device__ __inline__ float3 unpackNormal(uint packed) {
  float2 octf2 = convuint32float2(packed);
  return OctDecode<0>(octf2);
}

#elif 0

//
// snorm10x3: signed 10-bit-per-component scalar unit real representation
// Better representation than unorm.
// Supported by most fixed-function graphics hardware.
// https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_snorm.txt
//   i=round(clamp(r,-1,1) * (2^(b-1) - 1)
//   r=clamp(i/(2^(b-1) - 1), -1, 1)
//

#elif 1

// OpenGL GLbyte signed quantization scheme
//   i = r * (2^b - 1) - 0.5;
//   r = (2i + 1)/(2^b - 1)
static __host__ __device__ __inline__ uint packNormal(const float3& normal) {
  // conversion to GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
  const float3 N = normal * 127.5f - 0.5f;
  const char4 packed = make_char4(N.x, N.y, N.z, 0);
  return *((uint *) &packed);
}

static __host__ __device__ __inline__ float3 unpackNormal(uint packed) {
  char4 c4norm = *((char4 *) &packed);

  // conversion from GLbyte format, Table 2.6, p. 44 of OpenGL spec 1.2.1
  // float = (2c+1)/(2^8-1)
  const float ci2f = 1.0f / 255.0f;
  const float cn2f = 1.0f / 127.5f;
  float3 N = c4norm * cn2f + ci2f;

  return N;
}
#endif



//
// Device functions to convert between linear and sRGB colorspaces
//
// It's important to note that accurate conversions between
// linear and sRGB color spaces require the use of
// floating point or deep bit depth integer arithmetic.
// We use the CUDA texturing hardware to perform sRGB to
// linear color conversion during texture sampling.
//
// Some useful example results from improper conversion techniques:
//   https://blog.demofox.org/2018/03/10/dont-convert-srgb-u8-to-linear-u8/
//


//
// Conversion between sRGB and linear using the official equations
//
static __forceinline__ __device__
float4 sRGB_to_linear(const float4 &rgba) {
  float4 lin;
  if (rgba.x <= 0.0404482362771082f) {
    lin.x = rgba.x * 0.0773993f; // divide by 12.92f;
  } else {
    lin.x = powf(((rgba.x + 0.055f)/1.055f), 2.4f);
  }

  if (rgba.y <= 0.0404482362771082f) {
    lin.y = rgba.y * 0.0773993f; // divide by 12.92f;
  } else {
    lin.y = powf(((rgba.y + 0.055f)/1.055f), 2.4f);
  }

  if (rgba.z <= 0.0404482362771082f) {
    lin.z = rgba.z * 0.0773993f; // divide by 12.92f;
  } else {
    lin.z = powf(((rgba.z + 0.055f)/1.055f), 2.4f);
  }

  lin.w = rgba.w; // alpha remains linear regardless of color space

  return lin;
}


//
// Conversion between linear and sRGB using the official equations
//
static __forceinline__ __device__
float4 linear_to_sRGB(const float4 &lin) {
  float4 rgba;
  if (lin.x > 0.0031308f) {
    rgba.x = 1.055f * (powf(lin.x, (1.0f / 2.4f))) - 0.055f;
  } else {
    rgba.x = 12.92f * lin.x;
  }

  if (lin.y > 0.0031308f) {
    rgba.y = 1.055f * (powf(lin.y, (1.0f / 2.4f))) - 0.055f;
  } else {
    rgba.y = 12.92f * lin.y;
  }

  if (lin.z > 0.0031308f) {
    rgba.z = 1.055f * (powf(lin.z, (1.0f / 2.4f))) - 0.055f;
  } else {
    rgba.z = 12.92f * lin.z;
  }

  rgba.w = lin.w; // alpha remains linear regardless of color space

  return rgba;
}



//
// Fast, approximate conversion between linear and sRGB:
//   https://excamera.com/sphinx/article-srgb.html 
//   http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
//
static __forceinline__ __device__
float4 sRGB_to_linear_approx(const float4 &rgba) {
  float3 sRGB = make_float3(rgba);
  float3 lin = sRGB * (sRGB * (sRGB * 0.305306011f + 0.682171111f) + 0.012522878f);
  return make_float4(lin, rgba.w); // preserve linear alpha
}


//
// Fast, approximate conversion between sRGB and linear:
//   https://excamera.com/sphinx/article-srgb.html 
//   http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
//
static __forceinline__ __device__
float4 linear_to_sRGB_approx(const float4 &linear) {
  float3 lin = make_float3(linear);
  float3 S1 = make_float3(sqrtf(lin.x), sqrtf(lin.y), sqrtf(lin.z));
  float3 S2 = make_float3(sqrtf(S1.x), sqrtf(S1.y), sqrtf(S1.z));
  float3 S3 = make_float3(sqrtf(S2.x), sqrtf(S2.y), sqrtf(S2.z));
  float3 sRGB = 0.662002687f * S1 + 0.684122060f * S2 
                - 0.323583601f * S3 - 0.0225411470f * lin; 
  return make_float4(sRGB, linear.w); // preserver linear alpha
}



//
// Fastest low-approximate conversion between linear and sRGB (gamma 2.0):
//
static __forceinline__ __device__
float4 sRGB_to_linear_approx_20(const float4 &rgba) {
  float3 sRGB = make_float3(rgba);
  return make_float4(sRGB * sRGB, rgba.w); // preserve linear alpha
}


//
// Fastest low-approximate conversion between sRGB and linear (gamma 2.0):
//
static __forceinline__ __device__
float4 linear_to_sRGB_approx_20(const float4 &linear) {
  float3 lin = make_float3(linear);
  float3 sRGB = make_float3(sqrtf(lin.x), sqrtf(lin.y), sqrtf(lin.z));
  return make_float4(sRGB, linear.w); // preserver linear alpha
}



//
// Tone mapping and color grading device functions.
// Useful references:
//   Photographic Tone Reproduction for Digital Images
//   E. Reinhard, M. Stark, P. Shirley, J. Ferwerda
//   ACM Transactions on Graphics, 21(3) pp. 267-276, 2002.
//   https://doi.org/10.1145/566654.566575
//
//   Tone Mapping of HDR Images: A Review
//   Y. Salih, W. Md-Esa, A. Malik, N. Saad.
//   http://doi.org/10.1109/ICIAS.2012.6306220
//
// Others:
// http://filmicworlds.com/blog/filmic-tonemapping-operators/
// http://filmicworlds.com/blog/filmic-tonemapping-with-piecewise-power-curves/
// http://filmicworlds.com/blog/minimal-color-grading-tools/
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
// https://bartwronski.com/2022/02/28/exposure-fusion-local-tonemapping-for-real-time-rendering/
//   https://bartwronski.github.io/local_tonemapping_js_demo/


//
// Calculate relative luminance from linear RGB w/ perceptual coefficients:
//   https://en.wikipedia.org/wiki/Relative_luminance
//
static __device__ __inline__ 
float luminance(float3 c) {
  return dot(c, make_float3(0.2126f, 0.7152f, 0.0722f));; 
}


//
// Rescale RGB colors to achieve desired luminance
//
static __device__ __inline__ 
float3 rescale_luminance(float3 c, float newluminance) {
  float l = luminance(c);
  return c * (newluminance / l);
}


//
// ACES filmic tone mapping approximations:
//   https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
//   https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
//
static __device__ __inline__ 
float3 ACES_TMO(float3 c) {
  float3 num = c * (2.51f * c + make_float3(0.03f));
  float3 den = c * (2.43f * c + make_float3(0.59f)) + make_float3(0.14f);
  float3 t = num / den;

  return t; // clamping is deferred 
}


//
// Reinhard style tone mapping
//
static __device__ __inline__ 
float3 reinhard_TMO(float3 c) {
  return c / (make_float3(1.0f) + c); 
}


//
// Extended Reinhard style tone mapping:
//   https://64.github.io/tonemapping/
//
static __device__ __inline__ 
float3 reinhard_extended_TMO(float3 c, float maxwhite) {
  float3 num = c * (make_float3(1.0f) + (c / make_float3(maxwhite * maxwhite)));
  return num / (make_float3(1.0f) + c); 
}


//
// Extended Reinhard style tone mapping applied to luminance:
//   https://64.github.io/tonemapping/
//
static __device__ __inline__ 
float3 reinhard_extended_luminance_TMO(float3 c, float maxL) {
  float oldL = luminance(c);
  float num = oldL * (1.0f + (oldL / (maxL * maxL)));
  float newL = num / (1.0f + oldL);
  return rescale_luminance(c, newL);
}


//
// Protect functions that are only GPU-callable, e.g., those that
// use GPU-specific intrinsics such as __saturatef() or others.
//
#if defined(TACHYON_INTERNAL)

// clamp vector to range [0,1] using __saturatef() intrinsic
static __device__ __inline__ float3 clamp_float3(const float3 &a) {
  return make_float3(__saturatef(a.x), __saturatef(a.y), __saturatef(a.z));
}

// clamp vector to range [0,1] using __saturatef() intrinsic
static __device__ __inline__ float4 clamp_float4(const float4 &a) {
  return make_float4(__saturatef(a.x), __saturatef(a.y), 
                     __saturatef(a.z), __saturatef(a.w));
}


//
// Color conversion operations
//

/// Convert float3 rgb data to uchar4 with alpha channel set to 255.
static __device__ __inline__ uchar4 make_color_rgb4u(const float3& c) {
  return make_uchar4(static_cast<unsigned char>(__saturatef(c.x)*255.99f),
                     static_cast<unsigned char>(__saturatef(c.y)*255.99f),
                     static_cast<unsigned char>(__saturatef(c.z)*255.99f),
                     255u);
}

/// Convert float4 rgba data to uchar4 unorm color representation.
static __device__ __inline__ uchar4 make_color_rgb4u(const float4& c) {
  return make_uchar4(static_cast<unsigned char>(__saturatef(c.x)*255.99f),
                     static_cast<unsigned char>(__saturatef(c.y)*255.99f),
                     static_cast<unsigned char>(__saturatef(c.z)*255.99f),
                     static_cast<unsigned char>(__saturatef(c.w)*255.99f));
}



//
// HDR tone mapping
//
static __inline__ __device__
float4 tonemap_color(const float4 & colrgba4f, int tonemap_mode, 
                     float tonemap_exposure, int colorspace) {
  float alpha = colrgba4f.w; // preserve linear alpha channel
  float3 color = make_float3(colrgba4f) * tonemap_exposure;

  switch (tonemap_mode) {
    case RT_TONEMAP_ACES:
      color = ACES_TMO(color);
      break;

    case RT_TONEMAP_REINHARD:
      color = reinhard_TMO(color);
      break;

    case RT_TONEMAP_REINHARD_EXT:
      color = reinhard_extended_TMO(color, 1.0f);
      break;

    case RT_TONEMAP_REINHARD_EXT_L:
      color = reinhard_extended_luminance_TMO(color, 1.0f);
      break;

    case RT_TONEMAP_CLAMP:
    default:
      break;
  }

  float4 outcolor = make_float4(color, alpha);

  // range clamping is deferred until storage format conversion
  return outcolor;
}

#endif // TACHYON_INTERNAL



//
// End of potentially unreferenced functions
//
//#pragma pop


#endif // TACHYONOPTIXSHADERS_H
