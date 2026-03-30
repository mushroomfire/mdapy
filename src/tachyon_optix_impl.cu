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
 *  3. Use the same namespace as the rest of mdapy (mdapy_tachyon or mdapy -
 *     must match tachyon_optix_render.h).
 */

// ── Tell TachyonOptiX.h to use tachyon's own util.h, not VMD's WKFUtils.h ──
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
#include <dlfcn.h>   // dladdr — locate the .so at runtime to find PTX sibling

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
    Vec3 u = w.cross(up).normalized();
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

// Anchor symbol used by dladdr to locate this shared library at runtime.
// Must be a plain function — nvcc rejects member-fn-ptr and lambda→void* casts.
static void _mdapy_tachyon_anchor() {}

struct TachyonOptiXRenderer::Impl {
    TachyonOptiX ctx;

    Impl() {
        ctx.set_verbose_mode(TachyonOptiX::RT_VERB_MIN);

        // Resolve the absolute path to TachyonOptiXShaders.ptx.
        // The PTX file is installed next to _tachyon*.so (both land in the
        // mdapy package directory via CMake's install(FILES ...) rule).
        // dladdr on _mdapy_tachyon_anchor gives us the path of this .so.
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&_mdapy_tachyon_anchor), &info)
                && info.dli_fname) {
            std::string so_path(info.dli_fname);
            auto sep = so_path.rfind('/');
            std::string dir = (sep != std::string::npos) ? so_path.substr(0, sep + 1) : "./";
            std::string ptx_path = dir + "TachyonOptiXShaders.ptx";
            printf("[mdapy] dladdr so_path : %s\n", so_path.c_str());
            printf("[mdapy] dladdr ptx_path: %s\n", ptx_path.c_str());
            ctx.set_shader_path(ptx_path.c_str());
        } else {
            printf("[mdapy] dladdr failed — shaderpath left as default\n");
        }

        // create_context() was removed from TachyonOptiX constructor so we
        // can set the correct shaderpath first.  Call it here explicitly.
        ctx.create_context();
        ctx.destroy_scene();
    }

    int makeMaterial(float r, float g, float b, float alpha) {
        return ctx.add_material(0.3f,0.8f,0.0f,40.0f,0.0f,alpha,0.0f,0.0f,0,-1);
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
            double zs = 1.0;
            if (cp.dofEnabled && cp.dofFocalLen>0 && cp.dofAperture>0) {
                ctx.camera_dof_enable(1);
                ctx.set_camera_dof_focal_dist(float(cp.dofFocalLen));
                ctx.set_camera_dof_fnumber(float(cp.dofFocalLen/(2.0*cp.dofAperture)));
                zs = cp.dofFocalLen;
            } else { ctx.camera_dof_enable(0); }
            ctx.set_camera_zoom(float(0.5/std::tan(cp.fieldOfView*0.5)/zs));
        } else {
            ctx.set_camera_type(CT::RT_ORTHOGRAPHIC);
            Vec3 dir=cp.direction.normalized(); dir.z=-dir.z;
            pos[0]+=float(dir.x)*float(cp.znear);
            pos[1]+=float(dir.y)*float(cp.znear);
            pos[2]+=float(dir.z)*float(cp.znear);
            ctx.set_camera_pos(pos);
            ctx.set_camera_ONB(U,V,W);
            ctx.set_camera_zoom(float(0.5/cp.fieldOfView));
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
        CylinderArray ca; ca.materialindex=mat;
        RingArray ra;     ra.materialindex=mat;
        for (size_t i=0; i<be.count; ++i) {
            const double *p1=be.points+i*6, *p2=be.points+i*6+3;
            float3 a=tvec3(p1[0],p1[1],p1[2]), b=tvec3(p2[0],p2[1],p2[2]);
            ca.start.push_back(a); ca.end.push_back(b); ca.radius.push_back(be.radius);
            float3 ax={b.x-a.x,b.y-a.y,b.z-a.z}, nax={-ax.x,-ax.y,-ax.z};
            ra.center.push_back(a); ra.normal.push_back(nax); ra.inrad.push_back(0); ra.outrad.push_back(be.radius);
            ra.center.push_back(b); ra.normal.push_back(ax);  ra.inrad.push_back(0); ra.outrad.push_back(be.radius);
        }
        if (!ca.start.empty())  ctx.add_cylarray(ca,mat);
        if (!ra.center.empty()) ctx.add_ringarray(ra,mat);
    }

    std::vector<uint8_t> render(const RenderParams &rp,
                                const CameraParams &cp,
                                const ParticleData &pd,
                                const BoxEdgeData  *box)
    {
        ctx.framebuffer_config(rp.width, rp.height, 0);
        ctx.framebuffer_clear();
        ctx.set_bg_mode(TachyonOptiX::RT_BACKGROUND_TEXTURE_SOLID);
        float bgcol[3]={rp.bgR,rp.bgG,rp.bgB};
        ctx.set_bg_color(bgcol);
        ctx.set_aa_samples(rp.antialiasingEnabled ? rp.antialiasingSamples : 1);
        if (rp.aoEnabled) {
            ctx.set_ao_samples(rp.aoSamples);
            ctx.set_ao_ambient(float(rp.aoBrightness));
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
        ctx.destroy_scene();
        addParticles(pd);
        if (box && box->count>0) addBoxEdges(*box);
        ctx.update_rendering_state(0);
        ctx.render();

        std::vector<uint8_t> rgb24(rp.width*rp.height*3);
        ctx.framebuffer_download_rgb4u(rgb24.data());
        uint8_t alpha=uint8_t(std::max(0.f,std::min(1.f,rp.bgA))*255.f+0.5f);
        std::vector<uint8_t> rgba(rp.width*rp.height*4);
        for (int i=0; i<rp.width*rp.height; ++i) {
            rgba[i*4]=rgb24[i*3]; rgba[i*4+1]=rgb24[i*3+1];
            rgba[i*4+2]=rgb24[i*3+2]; rgba[i*4+3]=alpha;
        }
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
