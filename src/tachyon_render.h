#pragma once
/**
 * tachyon_render.h  —  mdapy Tachyon 渲染器
 *
 * 只使用 Tachyon 公开 API (tachyon.h)，无内部头文件依赖。
 * 相机模型与 OVITO TachyonRenderer 完全一致。
 *
 * ★ 关键修复：rt_newscene() 默认把输出文件设为 /tmp/outfile.tga
 *   (writeimagefile=1)，并把内部格式设为 RGB96F。
 *   若不调用 rt_outputfile(scene, "") 把 writeimagefile 清零，
 *   renderscene() 渲染完后会调用 writeimage()，把我们的 RGBA32 buffer
 *   当成 RGB96F float buffer 解引用 → segfault。
 *   修复：在 rt_rawimage_rgba32() 之前，必须先调用 rt_outputfile(scene, "")。
 */

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

extern "C" {
#include <tachyon.h>
}

namespace mdapy {

// ─── 向量辅助 ──────────────────────────────────────────────────────────────────
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
    Vec3 normalized() const {
        double n = norm();
        return n > 1e-12 ? (*this)*(1.0/n) : Vec3{0,0,1};
    }
};

// Tachyon 坐标约定（与 OVITO 相同）：z 分量取反
// OVITO 源码: rt_vector(v.x(), v.y(), -v.z())
inline apivector tvec(double x, double y, double z) { return rt_vector(x, y, -z); }
inline apivector tvec(const Vec3& v)                { return rt_vector(v.x, v.y, -v.z); }

// ─── 相机参数（对标 OVITO ViewProjectionParameters）──────────────────────────
struct CameraParams {
    bool   isPerspective = true;
    // 透视: 垂直视角(弧度)  正交: 视口半高(世界坐标)
    double fieldOfView   = 40.0 * M_PI / 180.0;

    Vec3   position  { 0,  0, 50};
    Vec3   direction { 0,  0, -1};
    Vec3   up        { 0,  1,  0};
    double znear     = 0.0;    // 正交相机近裁剪面偏移

    // 景深（仅透视）
    bool   dofEnabled  = false;
    double dofFocalLen = 40.0;
    double dofAperture = 0.01;
};

// ─── 渲染参数 ─────────────────────────────────────────────────────────────────
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

    float bgR = 0.f, bgG = 0.f, bgB = 0.f, bgA = 1.f;
    int   numThreads = 0;   // 0 = Tachyon 自动
};

// ─── 粒子输入 ─────────────────────────────────────────────────────────────────
struct ParticleData {
    const double* positions = nullptr;  // [N,3] float64
    const float*  colors    = nullptr;  // [N,4] float32 RGBA ∈[0,1]
    const float*  radii     = nullptr;  // [N]   float32
    size_t        count     = 0;
};

// ─── 晶胞棱线输入 ─────────────────────────────────────────────────────────────
struct BoxEdgeData {
    const double* points = nullptr;  // [M,2,3] float64，每棱两端点
    size_t        count  = 0;
    float r=1.f, g=1.f, b=1.f, a=1.f;
    float radius = 0.05f;
};

// ─────────────────────────────────────────────────────────────────────────────
//  TachyonRenderer
// ─────────────────────────────────────────────────────────────────────────────
class TachyonRenderer {
public:
    TachyonRenderer() {
        rt_set_ui_message([](int, char*){});
        rt_set_ui_progress([](int){});
        rt_initialize(0, nullptr);
    }
    ~TachyonRenderer() { rt_finalize(); }

    // 禁止拷贝（Tachyon 全局只能初始化一次）
    TachyonRenderer(const TachyonRenderer&)            = delete;
    TachyonRenderer& operator=(const TachyonRenderer&) = delete;

    /**
     * 渲染一帧。
     * 返回 RGBA uint8，shape = [height, width, 4]，行从上到下。
     */
    std::vector<uint8_t> render(
        const RenderParams& rp,
        const CameraParams& cp,
        const ParticleData& pd,
        const BoxEdgeData*  box = nullptr)
    {
        SceneHandle scene = rt_newscene();

        // ── 分辨率 ────────────────────────────────────────────────────────
        rt_resolution(scene, rp.width, rp.height);
        if (rp.numThreads > 0)
            rt_set_numthreads(scene, rp.numThreads);
        if (rp.antialiasingEnabled)
            rt_aa_maxsamples(scene, rp.antialiasingSamples);

        // ── ★ 关键修复：禁用文件输出，必须在 rt_rawimage_rgba32 之前调用 ──
        // rt_newscene() 默认 writeimagefile=1 + imgbufformat=RGB96F
        // 如果不清零，renderscene() 最后会把我们的 RGBA32 buffer 当
        // RGB96F float 来读取 → 内存越界 → segfault
        rt_outputfile(scene, "");   // 设置 writeimagefile=0

        // ── 原始图像 buffer（Tachyon 从下往上填充，之后我们翻转）────────
        std::vector<uint8_t> rawbuf(rp.width * rp.height * 4, 0);
        rt_rawimage_rgba32(scene, rawbuf.data());

        // ── 背景 & 基本设置 ───────────────────────────────────────────────
        colora bg{rp.bgR, rp.bgG, rp.bgB, rp.bgA};
        rt_background(scene, bg);
        rt_phong_shader(scene, RT_SHADER_NULL_PHONG);
        rt_trans_mode(scene, RT_TRANS_VMD);
        rt_camera_raydepth(scene, 1000);

        // ── 相机 ──────────────────────────────────────────────────────────
        setupCamera(scene, cp);

        // ── 灯光 ──────────────────────────────────────────────────────────
        setupLight(scene, cp, rp);

        // ── Shader 模式 ───────────────────────────────────────────────────
        if (rp.aoEnabled || (rp.directLightEnabled && rp.shadowsEnabled))
            rt_shadermode(scene, RT_SHADER_FULL);
        else
            rt_shadermode(scene, RT_SHADER_MEDIUM);

        // ── Ambient Occlusion ─────────────────────────────────────────────
        if (rp.aoEnabled) {
            apicolor skycol;
            skycol.r = skycol.g = skycol.b = static_cast<float>(rp.aoBrightness);
            rt_rescale_lights(scene, 0.2f);
            rt_ambient_occlusion(scene, rp.aoSamples, skycol);
        }

        // ── 几何体 ────────────────────────────────────────────────────────
        addParticles(scene, pd);
        if (box && box->count > 0)
            addBoxEdges(scene, *box);

        // ── 渲染 ──────────────────────────────────────────────────────────
        rt_renderscene(scene);

        // ── 垂直翻转（Tachyon 从底行到顶行填充）─────────────────────────
        std::vector<uint8_t> result(rawbuf.size());
        const int bpl = rp.width * 4;
        for (int y = 0; y < rp.height; y++)
            std::memcpy(result.data() + (rp.height-1-y)*bpl,
                        rawbuf.data() + y*bpl, bpl);

        rt_deletescene(scene);
        return result;
    }

private:
    // ── 相机设置（完全对标 OVITO TachyonRenderer::renderFrame）────────────
    static void setupCamera(SceneHandle scene, const CameraParams& cp) {
        const Vec3 pos = cp.position;
        const Vec3 dir = cp.direction.normalized();
        const Vec3 up  = cp.up.normalized();

        if (cp.isPerspective) {
            double zoomScale = 1.0;
            if (!cp.dofEnabled || cp.dofFocalLen <= 0 || cp.dofAperture <= 0) {
                rt_camera_projection(scene, RT_PROJECTION_PERSPECTIVE);
            } else {
                rt_camera_projection(scene, RT_PROJECTION_PERSPECTIVE_DOF);
                rt_camera_dof(scene, cp.dofFocalLen, cp.dofAperture);
                zoomScale = cp.dofFocalLen;
            }
            // OVITO: zoom = 0.5 / tan(fov*0.5) / zoomScale
            rt_camera_position(scene, tvec(pos), tvec(dir), tvec(up));
            rt_camera_zoom(scene, 0.5 / std::tan(cp.fieldOfView * 0.5) / zoomScale);
        } else {
            // OVITO: zoom = 0.5 / fieldOfView，相机沿 dir 偏移 znear
            rt_camera_projection(scene, RT_PROJECTION_ORTHOGRAPHIC);
            Vec3 orthoPos = pos + dir * (cp.znear - 1e-9);
            rt_camera_position(scene, tvec(orthoPos), tvec(dir), tvec(up));
            rt_camera_zoom(scene, 0.5 / cp.fieldOfView);
        }
    }

    // ── 方向光（对标 OVITO：相机空间 (0.2,-0.2,-1.0) → 世界空间）──────────
    static void setupLight(SceneHandle scene,
                           const CameraParams& cp,
                           const RenderParams& rp)
    {
        if (!rp.directLightEnabled) return;
        apitexture lt{};
        lt.col.r = lt.col.g = lt.col.b = static_cast<float>(rp.directLightIntensity);
        lt.ambient = lt.opacity = lt.diffuse = 1.f;
        void* ltex = rt_texture(scene, &lt);

        // 重建相机坐标轴
        const Vec3 d = cp.direction.normalized();
        const Vec3 r = d.cross(cp.up.normalized()).normalized();
        const Vec3 u = r.cross(d).normalized();
        // OVITO lightDir = inverseViewMatrix * (0.2,-0.2,-1.0)
        const Vec3 wl = r*0.2 + u*(-0.2) + d*(-1.0);
        rt_directional_light(scene, ltex, tvec(wl));
    }

    // ── 创建材质（对标 OVITO getTachyonTexture）──────────────────────────────
    static void* makeTex(SceneHandle scene,
                         float cr, float cg, float cb, float alpha)
    {
        apitexture tex{};
        tex.ambient     = 0.3f;
        tex.diffuse     = 0.8f;
        tex.specular    = 0.0f;
        tex.opacity     = alpha;
        tex.col.r       = cr;
        tex.col.g       = cg;
        tex.col.b       = cb;
        tex.texturefunc = RT_TEXTURE_CONSTANT;
        return rt_texture(scene, &tex);
    }

    // ── 球形粒子（对标 OVITO SphericalShape 分支）────────────────────────────
    static void addParticles(SceneHandle scene, const ParticleData& pd) {
        for (size_t i = 0; i < pd.count; i++) {
            const float alpha = pd.colors[i*4+3];
            if (alpha <= 0.f) continue;
            void* tex = makeTex(scene,
                pd.colors[i*4+0], pd.colors[i*4+1],
                pd.colors[i*4+2], alpha);
            rt_sphere(scene, tex,
                tvec(pd.positions[i*3+0],
                     pd.positions[i*3+1],
                     pd.positions[i*3+2]),
                pd.radii[i]);
        }
    }

    // ── 晶胞棱线（有限圆柱 + 两端封盖）──────────────────────────────────────
    static void addBoxEdges(SceneHandle scene, const BoxEdgeData& be) {
        void* tex = makeTex(scene, be.r, be.g, be.b, be.a);
        for (size_t i = 0; i < be.count; i++) {
            const double* p1 = be.points + i*6;
            const double* p2 = be.points + i*6 + 3;
            const Vec3 a{p1[0], p1[1], p1[2]};
            const Vec3 b{p2[0], p2[1], p2[2]};
            const Vec3 axis = b - a;
            const Vec3 neg  = axis * (-1.0);
            rt_fcylinder(scene, tex, tvec(a), tvec(axis), be.radius);
            rt_ring(scene, tex, tvec(a), tvec(neg),  0.f, be.radius);
            rt_ring(scene, tex, tvec(b), tvec(axis), 0.f, be.radius);
        }
    }
};

} // namespace mdapy