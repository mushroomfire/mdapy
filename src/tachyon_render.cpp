/**
 * tachyon_render.cpp  —  mdapy Tachyon nanobind 绑定
 *
 * 暴露的 Python 接口：
 *   _tachyon.TachyonRenderer
 *   _tachyon.RenderParams
 *   _tachyon.CameraParams
 *   _tachyon.Vec3
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstring>
#include <vector>
#include <cstdint>
#include <stdexcept>

#include "tachyon_render.h"

namespace nb = nanobind;
using namespace mdapy;

// ndarray 类型别名
using PosArray = nb::ndarray<double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using ColArray = nb::ndarray<float,  nb::shape<-1, 4>, nb::c_contig, nb::device::cpu>;
using RadArray = nb::ndarray<float,  nb::shape<-1>,    nb::c_contig, nb::device::cpu>;
using BoxArray = nb::ndarray<double, nb::shape<-1, 2, 3>, nb::c_contig, nb::device::cpu>;

NB_MODULE(_tachyon, m) {
    m.doc() = "Tachyon ray-tracing renderer for mdapy";

    // ── Vec3 ─────────────────────────────────────────────────────────────
    nb::class_<Vec3>(m, "Vec3")
        .def(nb::init<double, double, double>(),
             nb::arg("x")=0., nb::arg("y")=0., nb::arg("z")=0.)
        .def_rw("x", &Vec3::x)
        .def_rw("y", &Vec3::y)
        .def_rw("z", &Vec3::z)
        .def("__repr__", [](const Vec3& v) {
            return "Vec3(" + std::to_string(v.x) + ", "
                           + std::to_string(v.y) + ", "
                           + std::to_string(v.z) + ")";
        });

    // ── CameraParams ─────────────────────────────────────────────────────
    nb::class_<CameraParams>(m, "CameraParams")
        .def(nb::init<>())
        .def_rw("is_perspective",  &CameraParams::isPerspective)
        .def_rw("field_of_view",   &CameraParams::fieldOfView)
        .def_rw("position",        &CameraParams::position)
        .def_rw("direction",       &CameraParams::direction)
        .def_rw("up",              &CameraParams::up)
        .def_rw("znear",           &CameraParams::znear)
        .def_rw("dof_enabled",     &CameraParams::dofEnabled)
        .def_rw("dof_focal_len",   &CameraParams::dofFocalLen)
        .def_rw("dof_aperture",    &CameraParams::dofAperture);

    // ── RenderParams ──────────────────────────────────────────────────────
    nb::class_<RenderParams>(m, "RenderParams")
        .def(nb::init<>())
        .def_rw("width",                  &RenderParams::width)
        .def_rw("height",                 &RenderParams::height)
        .def_rw("antialiasing_enabled",   &RenderParams::antialiasingEnabled)
        .def_rw("antialiasing_samples",   &RenderParams::antialiasingSamples)
        .def_rw("direct_light_enabled",   &RenderParams::directLightEnabled)
        .def_rw("shadows_enabled",        &RenderParams::shadowsEnabled)
        .def_rw("direct_light_intensity", &RenderParams::directLightIntensity)
        .def_rw("ao_enabled",             &RenderParams::aoEnabled)
        .def_rw("ao_samples",             &RenderParams::aoSamples)
        .def_rw("ao_brightness",          &RenderParams::aoBrightness)
        .def_rw("bg_r",                   &RenderParams::bgR)
        .def_rw("bg_g",                   &RenderParams::bgG)
        .def_rw("bg_b",                   &RenderParams::bgB)
        .def_rw("bg_a",                   &RenderParams::bgA)
        .def_rw("num_threads",            &RenderParams::numThreads);

    // ── TachyonRenderer ───────────────────────────────────────────────────
    nb::class_<TachyonRenderer>(m, "TachyonRenderer")
        .def(nb::init<>())
        .def("render",
            [](TachyonRenderer& self,
               const RenderParams& rp,
               const CameraParams& cp,
               PosArray positions,      // (N,3) float64
               ColArray colors,         // (N,4) float32
               RadArray radii,          // (N,)  float32
               nb::object box_obj)      // None | (M,2,3) float64
            -> nb::ndarray<nb::numpy, uint8_t, nb::shape<-1,-1,4>>
        {
            // ── 粒子（直接使用 Python 数组内存，生命期由 Python GC 保证）──
            const size_t N = positions.shape(0);
            if (colors.shape(0) != N || radii.shape(0) != N)
                throw std::runtime_error("positions/colors/radii size mismatch");

            ParticleData pd;
            pd.positions = positions.data();
            pd.colors    = colors.data();
            pd.radii     = radii.data();
            pd.count     = N;

            // ── 晶胞棱线：立即深拷贝到 std::vector ──────────────────────
            // ★ 关键：不能存 BoxArray 的 .data() 指针然后再渲染
            //   因为 nb::cast<BoxArray>(box_obj) 返回的局部对象析构后
            //   指针即悬空。改为拷贝到 std::vector 确保生命期安全。
            std::vector<double> box_buf;
            BoxEdgeData be{};
            be.r = be.g = be.b = be.a = 1.f;
            be.radius = 0.05f;

            bool hasBox = !box_obj.is_none();
            if (hasBox) {
                BoxArray ba = nb::cast<BoxArray>(box_obj);
                const size_t M = ba.shape(0);
                if (M > 0) {
                    // 深拷贝：ba 析构后 box_buf 仍然有效
                    box_buf.assign(ba.data(), ba.data() + M * 6);
                    be.points = box_buf.data();
                    be.count  = M;
                } else {
                    hasBox = false;
                }
            }

            // ── 渲染 ──────────────────────────────────────────────────────
            std::vector<uint8_t> raw = self.render(
                rp, cp, pd,
                hasBox ? &be : nullptr);

            // ── 把结果移交给 Python（通过 capsule 管理堆内存）────────────
            const size_t H = static_cast<size_t>(rp.height);
            const size_t W = static_cast<size_t>(rp.width);
            uint8_t* heap = new uint8_t[raw.size()];
            std::memcpy(heap, raw.data(), raw.size());
            nb::capsule owner(heap,
                [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
            size_t shape[3] = {H, W, 4};
            return nb::ndarray<nb::numpy, uint8_t, nb::shape<-1,-1,4>>(
                heap, 3, shape, owner);
        },
        nb::arg("render_params"),
        nb::arg("camera_params"),
        nb::arg("positions"),
        nb::arg("colors"),
        nb::arg("radii"),
        nb::arg("box_edges") = nb::none(),
        "Render sphere particles + optional box edges.\n\n"
        "Parameters\n"
        "----------\n"
        "positions : (N,3) float64\n"
        "colors    : (N,4) float32  RGBA in [0,1]\n"
        "radii     : (N,)  float32\n"
        "box_edges : (M,2,3) float64 or None\n\n"
        "Returns\n"
        "-------\n"
        "numpy.ndarray shape (H,W,4) uint8 RGBA");
}