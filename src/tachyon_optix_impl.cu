// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
/**
 * tachyon_optix_impl.cu  -  TachyonOptiXRenderer PIMPL implementation
 *
 * Compiled by nvcc ONLY.  Never include from a g++ translation unit.
 *
 * Key design rules:
 *  1. Do NOT include tachyon_render.h or tachyon.h - they conflict with
 *     TachyonOptiX.h (rt_texture, rt_directional_light, RT_FOG_NONE clash).
 *  2. Redefine the minimal plain-data structs (Vec3, CameraParams, etc.)
 *     independently so this file has no dependency on tachyon.h.
 *  3. Use the same namespace as the rest of mdapy (mdapy_tachyon).
 */

// MSVC requires _USE_MATH_DEFINES before <cmath> to expose M_PI.
// Define it unconditionally here so nvcc + MSVC also see M_PI.
#if !defined(_USE_MATH_DEFINES)
#  define _USE_MATH_DEFINES
#endif

// Tell TachyonOptiX.h to use tachyon's own util.h, not VMD's WKFUtils.h
#if !defined(TACHYONINTERNAL)
#  define TACHYONINTERNAL 1
#endif
#if !defined(TACHYON_INTERNAL)
#  define TACHYON_INTERNAL 1
#endif
// Do NOT define TACHYON_OPTIXDENOISER - the denoiser code has an OptiX 7.6
// API incompatibility (denoiseAlpha is now OptixDenoiserAlphaMode enum, not int).

#include <cuda_runtime.h>
#include "TachyonOptiX.h"   // from extern/tachyon/src/ — nvcc only!

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <string>

// ---------------------------------------------------------------------------
// Cross-platform shared-library directory resolution
//
// On Linux/macOS: use dladdr() from <dlfcn.h> to locate this .so at runtime.
// On Windows:     use GetModuleHandleEx() + GetModuleFileName() from <windows.h>
//                 (dlfcn.h does not exist on Windows).
//
// Both implementations return the directory that contains the current shared
// library, with a trailing slash (e.g. "/path/to/mdapy/" or "C:/path/mdapy/").
// The PTX file is installed into that same directory by CMake's install() rule.
// ---------------------------------------------------------------------------
#if defined(_WIN32) || defined(_WIN64)

#  include <windows.h>

// Returns the directory containing the current DLL, with trailing slash.
// Uses the address of this function itself as the anchor.
static std::string get_so_directory() {
    HMODULE hm = NULL;
    // GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: find module by address
    // GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: don't increment refcount
    if (GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&get_so_directory),
            &hm)) {
        char path[MAX_PATH];
        DWORD len = GetModuleFileNameA(hm, path, MAX_PATH);
        if (len > 0 && len < MAX_PATH) {
            std::string p(path, len);
            // Normalize backslashes to forward slashes for consistency
            for (auto &c : p) if (c == '\\') c = '/';
            auto sep = p.rfind('/');
            if (sep != std::string::npos)
                return p.substr(0, sep + 1);
        }
    }
    return "./";
}

#else  // Linux / macOS

#  include <dlfcn.h>
#  include <climits>   // PATH_MAX

// Returns the directory containing the current .so, with trailing slash.
// Uses the address of this function itself as the anchor for dladdr.
static std::string get_so_directory() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&get_so_directory), &info)
            && info.dli_fname) {
        // realpath() resolves symlinks and relative components, giving a
        // canonical absolute path (important in virtual-env installs).
        char resolved[PATH_MAX];
        const char *fpath = info.dli_fname;
        if (realpath(fpath, resolved))
            fpath = resolved;
        std::string so_path(fpath);
        auto sep = so_path.rfind('/');
        if (sep != std::string::npos)
            return so_path.substr(0, sep + 1);
    }
    return "./";
}

#endif  // _WIN32

// ---------------------------------------------------------------------------
// Minimal plain-data struct copies (no tachyon.h dependency).
// Must stay in sync with tachyon_render.h.
// ---------------------------------------------------------------------------
namespace mdapy_tachyon {

struct Vec3 {
    double x{0}, y{0}, z{0};
    Vec3() = default;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s)      const { return {x*s,   y*s,   z*s  }; }
    double dot(const Vec3& o)     const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3& o)     const {
        return {y*o.z-z*o.y, z*o.x-x*o.z, x*o.y-y*o.x};
    }
    double norm()     const { return std::sqrt(dot(*this)); }
    Vec3 normalized() const { double n=norm(); return n>1e-12?(*this)*(1.0/n):Vec3{0,0,1}; }
};

struct CameraParams {
    bool   isPerspective = true;
    double fieldOfView   = 40.0 * M_PI / 180.0;
    Vec3   position  { 0,  0, 50};
    Vec3   direction { 0,  0, -1};
    Vec3   up        { 0,  1,  0};
    double znear     = 0.0;
    bool   dofEnabled  = false;
    double dofFocalLen = 40.0;
    double dofAperture = 0.01;
};

struct RenderParams {
    int  width  = 800;
    int  height = 600;
    bool antialiasingEnabled = true;
    int  antialiasingSamples = 12;
    bool   directLightEnabled   = true;
    bool   shadowsEnabled       = true;
    double directLightIntensity = 0.90;
    bool   aoEnabled    = true;
    int    aoSamples    = 12;
    double aoBrightness = 0.80;
    double aoMaxDist    = 3.402823e+38f;
    float bgR = 0.f, bgG = 0.f, bgB = 0.f, bgA = 1.f;
    int   numThreads = 0;
};

struct ParticleData {
    const double* positions = nullptr;
    const float*  colors    = nullptr;
    const float*  radii     = nullptr;
    size_t        count     = 0;
};

struct BoxEdgeData {
    const double* points = nullptr;
    size_t        count  = 0;
    float r=1.f, g=1.f, b=1.f, a=1.f;
    float radius = 0.05f;
};

// ---------------------------------------------------------------------------
// Coordinate helpers (flip Z for Tachyon's left-handed frame)
// ---------------------------------------------------------------------------
static inline float3 tvec3(double x, double y, double z) {
    return make_float3(float(x), float(y), float(-z));
}

static void build_onb(const CameraParams &cp,
                      float U[3], float V[3], float W[3])
{
    Vec3 dir = cp.direction.normalized(); dir.z = -dir.z;
    Vec3 up  = cp.up.normalized();        up.z  = -up.z;
    Vec3 w = dir;
    // Tachyon CPU convention: rightvec = upvec × viewvec
    // GPU U must match CPU rightvec; using w.cross(up) gives the opposite sign.
    Vec3 u = up.cross(w).normalized();
    Vec3 v = u.cross(w);
    U[0]=float(u.x); U[1]=float(u.y); U[2]=float(u.z);
    V[0]=float(v.x); V[1]=float(v.y); V[2]=float(v.z);
    W[0]=float(w.x); W[1]=float(w.y); W[2]=float(w.z);
}

// ---------------------------------------------------------------------------
// PIMPL implementation
// ---------------------------------------------------------------------------
// Replicate the TachyonOptiXRenderer class declaration here instead of
// including tachyon_optix_render.h, which cannot be included in an nvcc TU
// (CUDA/OptiX headers clash with tachyon.h symbols).
// Must stay in sync with tachyon_optix_render.h.
class TachyonOptiXRenderer {
public:
    TachyonOptiXRenderer();
    ~TachyonOptiXRenderer();
    TachyonOptiXRenderer(const TachyonOptiXRenderer &)            = delete;
    TachyonOptiXRenderer &operator=(const TachyonOptiXRenderer &) = delete;
    std::vector<uint8_t> render(const RenderParams &rp,
                                const CameraParams &cp,
                                const ParticleData &pd,
                                const BoxEdgeData  *box = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct TachyonOptiXRenderer::Impl {
    TachyonOptiX ctx;

    Impl() {
        ctx.set_verbose_mode(TachyonOptiX::RT_VERB_MIN);

        // ── Pre-flight check 1: CUDA GPU availability ────────────────────
        // TachyonOptiX::device_count() uses cudaGetDeviceCount() which is
        // safe to call before any context has been created.
        if (TachyonOptiX::device_count() == 0) {
            throw std::runtime_error(
                "[mdapy] No CUDA-capable GPU found. "
                "Use backend='cpu' to fall back to CPU rendering.");
        }

        // ── Resolve absolute path to TachyonOptiXShaders.ptx ────────────
        //
        // CMake installs both _tachyon*.so/_tachyon*.pyd and
        // TachyonOptiXShaders.ptx into the same mdapy package directory.
        // get_so_directory() locates that directory at runtime:
        //   Windows  → GetModuleHandleEx + GetModuleFileName
        //   Linux    → dladdr + realpath
        //
        // The TachyonOptiX default constructor already calls create_context()
        // with the hard-coded relative path "TachyonOptiXShaders.ptx", which
        // silently fails when the working directory is not the package dir.
        // Because create_context() only sets context_created=1 on success,
        // our explicit call below will correctly retry with the resolved path.
        std::string dir      = get_so_directory();
        std::string ptx_path = dir + "TachyonOptiXShaders.ptx";

        // ── Pre-flight check 2: PTX file must exist ──────────────────────
        // Fail early with a clear message instead of a later segfault.
        {
            FILE *fp = fopen(ptx_path.c_str(), "r");
            if (fp) {
                fclose(fp);
            } else {
                throw std::runtime_error(
                    "[mdapy] TachyonOptiXShaders.ptx not found at:\n"
                    "  " + ptx_path + "\n"
                    "  Ensure mdapy was installed via 'cmake --install' so that\n"
                    "  TachyonOptiXShaders.ptx is placed next to _tachyon*.so / _tachyon*.pyd.\n"
                    "  Use backend='cpu' to fall back to CPU rendering.");
            }
        }

        // Set the correct PTX path BEFORE calling create_context().
        // create_context() checks `if (context_created) return;` so if the
        // constructor's internal call failed (no PTX at relative path), this
        // second call will succeed with the resolved absolute path.
        // ── Initialize CUDA runtime (required before OptiX) ──────────────
        cudaFree(0);  // no-op that triggers CUDA runtime initialization

        // ── create_context() calls optixInit() internally.
        // TachyonOptiX.cu has been patched to return early (instead of
        // crashing) if optixInit() fails for any reason other than
        // OPTIX_ERROR_UNSUPPORTED_ABI_VERSION.
        ctx.set_shader_path(ptx_path.c_str());
        ctx.create_context();
        ctx.destroy_scene();

        // ── Post-flight check: CUDA error after context creation ─────────
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error(
                std::string("[mdapy] TachyonOptiX GPU context creation failed: ") +
                cudaGetErrorString(cuda_err) + "\n"
                "  NVIDIA driver may be too old for OptiX 7.6 (minimum driver: 520.xx)\n"
                "  Use backend='cpu' to fall back to CPU rendering.");
        }
    }

    int matCounter = 0; // per-frame material slot counter, reset in render()

    int makeMaterial(float r, float g, float b, float alpha) {
        // userindex must be >= 0.  Passing -1 causes materialcache[-1]
        // (out-of-bounds crash) in add_material_textured when the cache is
        // empty after destroy_scene().  Use an explicit counter so each call
        // gets a fresh, valid slot within the current frame.
        //
        // ambient=0.3: matches the CPU Tachyon material (ambient=0.3, diffuse=0.8).
        // The GPU shader adds p_Ka as white constant ambient (result += p_Ka),
        // which provides a base illumination floor that prevents shadow sides
        // from going completely black — consistent with the CPU renderer.
        return ctx.add_material(0.3f,0.8f,0.0f,40.0f,0.0f,alpha,0.0f,0.0f,0,matCounter++);
    }

    void setupCamera(const CameraParams &cp) {
        using CT = TachyonOptiX::CameraType;
        float U[3],V[3],W[3];
        build_onb(cp,U,V,W);
        float pos[3]={float(cp.position.x),float(cp.position.y),-float(cp.position.z)};

        if (cp.isPerspective) {
            ctx.set_camera_type(CT::RT_PERSPECTIVE);
            ctx.set_camera_pos(pos);
            ctx.set_camera_ONB(U,V,W);
            // GPU OptiX shader ray direction: normalize(d.x*U + d.y*V + W)
            // where d.y ranges from -cam_zoom to +cam_zoom across the image.
            // → half-FoV = arctan(cam_zoom), so cam_zoom = tan(half_fov).
            // NOTE: The Tachyon CPU renderer uses the DIFFERENT formula
            //       camzoom = 0.5/tan(half_fov)  (as in rt_camera_zoom).
            // Do NOT use the CPU formula here.
            double zs = 1.0;
            if (cp.dofEnabled && cp.dofFocalLen>0 && cp.dofAperture>0) {
                ctx.camera_dof_enable(1);
                ctx.set_camera_dof_focal_dist(float(cp.dofFocalLen));
                ctx.set_camera_dof_fnumber(float(cp.dofFocalLen/(2.0*cp.dofAperture)));
                zs = cp.dofFocalLen;
            } else { ctx.camera_dof_enable(0); }
            ctx.set_camera_zoom(float(std::tan(cp.fieldOfView*0.5)/zs));
        } else {
            ctx.set_camera_type(CT::RT_ORTHOGRAPHIC);
            Vec3 dir=cp.direction.normalized(); dir.z=-dir.z;
            pos[0]+=float(dir.x)*float(cp.znear);
            pos[1]+=float(dir.y)*float(cp.znear);
            pos[2]+=float(dir.z)*float(cp.znear);
            ctx.set_camera_pos(pos);
            ctx.set_camera_ONB(U,V,W);
            // GPU OptiX ortho shader: ray origin offset d.y = cam_zoom*(2*iy/H-1),
            // so cam_zoom is the half-height of the viewport in world units.
            // fieldOfView is already that half-height.
            // NOTE: The Tachyon CPU formula is camzoom = 0.5/fieldOfView.
            // Do NOT use the CPU formula here.
            ctx.set_camera_zoom(float(cp.fieldOfView));
        }
    }

    void addParticles(const ParticleData &pd) {
        if (!pd.count) return;
        SphereArray sa; sa.materialindex = makeMaterial(1,1,1,1);
        for (size_t i=0; i<pd.count; ++i) {
            float alpha=pd.colors[i*4+3];
            if (alpha<=0.f) continue;
            sa.center.push_back(tvec3(pd.positions[i*3],pd.positions[i*3+1],pd.positions[i*3+2]));
            sa.radius.push_back(pd.radii[i]);
            sa.primcolors3f.push_back(make_float3(pd.colors[i*4],pd.colors[i*4+1],pd.colors[i*4+2]));
        }
        if (!sa.center.empty()) ctx.add_spherearray(sa,sa.materialindex);
    }

    void addBoxEdges(const BoxEdgeData &be) {
        int mat = makeMaterial(be.r,be.g,be.b,be.a);
        float3 col = make_float3(be.r, be.g, be.b);
        CylinderArray ca; ca.materialindex=mat;
        RingArray ra;     ra.materialindex=mat;
        for (size_t i=0; i<be.count; ++i) {
            const double *p1=be.points+i*6, *p2=be.points+i*6+3;
            float3 a=tvec3(p1[0],p1[1],p1[2]), b=tvec3(p2[0],p2[1],p2[2]);
            ca.start.push_back(a); ca.end.push_back(b); ca.radius.push_back(be.radius);
            ca.primcolors3f.push_back(col);
            float3 ax={b.x-a.x,b.y-a.y,b.z-a.z}, nax={-ax.x,-ax.y,-ax.z};
            ra.center.push_back(a); ra.normal.push_back(nax); ra.inrad.push_back(0); ra.outrad.push_back(be.radius);
            ra.primcolors3f.push_back(col);
            ra.center.push_back(b); ra.normal.push_back(ax);  ra.inrad.push_back(0); ra.outrad.push_back(be.radius);
            ra.primcolors3f.push_back(col);
        }
        if (!ca.start.empty())  ctx.add_cylarray(ca,mat);
        if (!ra.center.empty()) ctx.add_ringarray(ra,mat);
    }

    std::vector<uint8_t> render(const RenderParams &rp,
                                const CameraParams &cp,
                                const ParticleData &pd,
                                const BoxEdgeData  *box)
    {
        // destroy_scene() must come FIRST: it calls destroy_lights() and
        // destroy_materials(), so any lights/materials added before it would
        // be silently discarded.
        ctx.destroy_scene();
        matCounter = 0; // reset per-frame material slot counter

        ctx.framebuffer_config(rp.width, rp.height, 0);
        ctx.framebuffer_clear();

        ctx.set_bg_mode(TachyonOptiX::RT_BACKGROUND_TEXTURE_SOLID);
        float bgcol[3]={rp.bgR,rp.bgG,rp.bgB};
        ctx.set_bg_color(bgcol);
        ctx.set_aa_samples(rp.antialiasingEnabled ? rp.antialiasingSamples : 1);
        if (rp.aoEnabled) {
            ctx.set_ao_samples(rp.aoSamples);
            ctx.set_ao_ambient(float(rp.aoBrightness));
            // Match CPU behavior: rt_rescale_lights(scene, 0.2f) scales the
            // direct light contribution to 0.2 when AO is enabled.
            // ao_direct is the GPU equivalent — it multiplies the summed
            // direct-light result before the AO term is added (see shader:
            //   result *= ao_direct;
            //   result += ao_ambient * col * Kd * shade_ao(...); ).
            ctx.set_ao_direct(0.2f);
            ctx.set_ao_maxdist(float(rp.aoMaxDist));
        } else { ctx.set_ao_samples(0); }

        if (rp.directLightEnabled) {
            Vec3 d=cp.direction.normalized(), up=cp.up.normalized();
            Vec3 r=d.cross(up).normalized(), u=r.cross(d).normalized();
            Vec3 wl=r*0.2+u*(-0.2)+d*(-1.0);
            float ldir[3]={float(wl.x),float(wl.y),float(-wl.z)};
            float lcol[3]={float(rp.directLightIntensity),float(rp.directLightIntensity),float(rp.directLightIntensity)};
            ctx.add_directional_light(ldir,lcol);
        }

        setupCamera(cp);

        addParticles(pd);
        if (box && box->count>0) {
            addBoxEdges(*box);
        }

        ctx.update_rendering_state(0);
        ctx.render();

        // framebuffer_download_rgb4u downloads width*height*sizeof(int) bytes,
        // i.e. 4 bytes per pixel (RGBA unsigned bytes), NOT 3 bytes.
        // Allocating only width*height*3 would overflow the buffer and cause
        // a segmentation fault.
        const int npixels = rp.width * rp.height;
        std::vector<uint8_t> rgba(npixels * 4);
        ctx.framebuffer_download_rgb4u(rgba.data());  // fills RGBA, 4 bytes/px

        // Override the alpha channel with the configured background alpha.
        // The GPU framebuffer alpha may not match the Python-configured value.
        const uint8_t alpha = static_cast<uint8_t>(
            std::max(0.f, std::min(1.f, rp.bgA)) * 255.f + 0.5f);
        for (int i = 0; i < npixels; ++i)
            rgba[i*4 + 3] = alpha;

        return rgba;
    }
};

// Public API
TachyonOptiXRenderer::TachyonOptiXRenderer() : impl_(std::make_unique<Impl>()) {}
TachyonOptiXRenderer::~TachyonOptiXRenderer() = default;
std::vector<uint8_t>
TachyonOptiXRenderer::render(const RenderParams &rp, const CameraParams &cp,
                              const ParticleData &pd, const BoxEdgeData *box)
{ return impl_->render(rp,cp,pd,box); }

} // namespace mdapy_tachyon
