// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
#pragma once
/**
 * tachyon_render.h  —  mdapy Tachyon 渲染器（适配 Tachyon 0.99.5）
 *
 * 只使用 Tachyon 公开 API (tachyon.h)，无内部头文件依赖。
 * 相机模型与 OVITO TachyonRenderer 完全一致。
 *
 * ★ 0.99.5 API 变更说明：
 *   1. rt_rawimage_rgba32() 已移除 → 改用 rt_rawimage_rgb24()（24-bit RGB）
 *      渲染完毕后手动将 RGB24 扩展为 RGBA32（alpha=255）。
 *   2. colora 类型已移除 → rt_background() 接受 apicolor（无 alpha 通道）。
 *   3. rt_ambient_occlusion() 新增 maxdist 参数：
 *      rt_ambient_occlusion(scene, numsamples, maxdist, col)
 *      传入 RT_AO_MAXDIST_UNLIMITED 表示无限距离。
 *   4. rt_outputfile(scene, "") 仍然需要调用以禁用文件输出。
 *
 * ★ 新增功能：
 *   BoxEdgeData 现支持自定义颜色（r/g/b/a）和线宽（radius）。
 *   ParticleData 沿用 positions/colors/radii，接口不变。
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

namespace mdapy_tachyon {

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
    // 注：不用 M_PI 是因为 MSVC 默认不暴露它（要求 _USE_MATH_DEFINES），
    // 直接写常量更稳，不引入额外的宏依赖。
    double fieldOfView   = 40.0 * 3.14159265358979323846 / 180.0;

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
    double aoMaxDist    = RT_AO_MAXDIST_UNLIMITED;  // AO 最大距离，默认无限

    float bgR = 0.f, bgG = 0.f, bgB = 0.f, bgA = 1.f;  // bgA 用于最终 RGBA 输出
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
    // 颜色与线宽（可由 Python 层自定义）
    float r      = 1.f;
    float g      = 1.f;
    float b      = 1.f;
    float a      = 1.f;   // 透明度（0=透明，1=不透明）
    float radius = 0.05f; // 棱线圆柱半径
};

// ─── Bond 圆柱输入 ───────────────────────────────────────────────────────────
struct BondData {
    const double* points = nullptr;  // [M,2,3] float64，每根 bond 两端点
    const float*  colors = nullptr;  // [M,4] float32，每根圆柱的 RGBA 颜色
    size_t        count  = 0;
    float         radius = 0.1f;
};

// ─────────────────────────────────────────────────────────────────────────────
//  TachyonRenderer
// ─────────────────────────────────────────────────────────────────────────────
class TachyonRenderer {
public:
    TachyonRenderer() {
        rt_set_ui_message([](int, char*){});
        rt_set_ui_progress([](int){});
        rt_initialize_nompi();   // 0.99.5 推荐：不启用 MPI
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
        const BondData*     bonds = nullptr,
        const BoxEdgeData*  box = nullptr)
    {
        SceneHandle scene = rt_newscene();

        // ── 分辨率 ────────────────────────────────────────────────────────
        rt_resolution(scene, rp.width, rp.height);
        if (rp.numThreads > 0)
            rt_set_numthreads(scene, rp.numThreads);
        if (rp.antialiasingEnabled)
            rt_aa_maxsamples(scene, rp.antialiasingSamples);

        // ── 禁用文件输出（必须在设置 rawimage buffer 之前调用）─────────
        // rt_newscene() 默认 writeimagefile=1，不清零会触发内部文件写入
        rt_outputfile(scene, "");

        // ── 原始图像 buffer（RGB24，Tachyon 从下往上填充）────────────────
        // 0.99.5 不再提供 rgba32，改用 rgb24，之后手动扩展为 RGBA32
        const int npixels = rp.width * rp.height;
        std::vector<uint8_t> rgb24buf(npixels * 3, 0);
        rt_rawimage_rgb24(scene, rgb24buf.data());

        // ── 背景（0.99.5: rt_background 接受 apicolor，无 alpha）─────────
        apicolor bg;
        bg.r = rp.bgR;
        bg.g = rp.bgG;
        bg.b = rp.bgB;
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

        // ── Ambient Occlusion（0.99.5 新增 maxdist 参数）──────────────────
        if (rp.aoEnabled) {
            apicolor skycol;
            skycol.r = skycol.g = skycol.b = static_cast<float>(rp.aoBrightness);
            rt_rescale_lights(scene, 0.2f);
            rt_ambient_occlusion(scene,
                                 rp.aoSamples,
                                 static_cast<apiflt>(rp.aoMaxDist),
                                 skycol);
        }

        // ── 几何体 ────────────────────────────────────────────────────────
        addParticles(scene, pd);
        if (bonds && bonds->count > 0)
            addBonds(scene, *bonds);
        if (box && box->count > 0)
            addBoxEdges(scene, *box);

        // ── 渲染 ──────────────────────────────────────────────────────────
        rt_renderscene(scene);

        // ── RGB24 → RGBA32（垂直翻转 + 补 alpha）─────────────────────────
        // Tachyon 从底行到顶行填充，需要翻转为标准的从顶到底顺序
        const uint8_t bgAlpha = static_cast<uint8_t>(
            std::max(0.f, std::min(1.f, rp.bgA)) * 255.f + 0.5f);
        std::vector<uint8_t> result(npixels * 4);
        const int bpl_src = rp.width * 3;
        const int bpl_dst = rp.width * 4;
        for (int y = 0; y < rp.height; y++) {
            const uint8_t* src = rgb24buf.data() + y * bpl_src;
            uint8_t* dst = result.data() + (rp.height - 1 - y) * bpl_dst;
            for (int x = 0; x < rp.width; x++) {
                dst[x*4+0] = src[x*3+0];  // R
                dst[x*4+1] = src[x*3+1];  // G
                dst[x*4+2] = src[x*3+2];  // B
                dst[x*4+3] = bgAlpha;      // A（统一用背景 alpha）
            }
        }

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
            rt_camera_position(scene, tvec(pos), tvec(dir), tvec(up));
            rt_camera_zoom(scene, 0.5 / std::tan(cp.fieldOfView * 0.5) / zoomScale);
        } else {
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

        const Vec3 d = cp.direction.normalized();
        const Vec3 r = d.cross(cp.up.normalized()).normalized();
        const Vec3 u = r.cross(d).normalized();
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

    // ── 球形粒子 ──────────────────────────────────────────────────────────────
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

    // ── Bond 圆柱 ────────────────────────────────────────────────────────────
    static void addBonds(SceneHandle scene, const BondData& bd) {
        for (size_t i = 0; i < bd.count; i++) {
            const double* p1 = bd.points + i * 6;
            const double* p2 = bd.points + i * 6 + 3;
            const float alpha = bd.colors[i * 4 + 3];
            if (alpha <= 0.f) continue;

            void* tex = makeTex(scene,
                bd.colors[i * 4 + 0],
                bd.colors[i * 4 + 1],
                bd.colors[i * 4 + 2],
                alpha);

            const Vec3 a{p1[0], p1[1], p1[2]};
            const Vec3 b{p2[0], p2[1], p2[2]};
            const Vec3 axis = b - a;
            const Vec3 neg  = axis * (-1.0);
            rt_fcylinder(scene, tex, tvec(a), tvec(axis), bd.radius);
            rt_ring(scene, tex, tvec(a), tvec(neg),  0.f, bd.radius);
            rt_ring(scene, tex, tvec(b), tvec(axis), 0.f, bd.radius);
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
