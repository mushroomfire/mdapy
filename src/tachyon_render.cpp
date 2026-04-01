/**
 * tachyon_render.cpp  -  mdapy Tachyon nanobind bindings
 *
 * Exposes a unified Python interface for both CPU (Tachyon classic) and
 * GPU (TachyonOptiX, NVIDIA OptiX 7+) backends.
 *
 * Python classes exposed in _tachyon:
 *   Vec3
 *   CameraParams
 *   RenderParams
 *   TachyonRenderer          (CPU backend, always available)
 *   TachyonOptiXRenderer     (GPU backend, only when MDAPY_OPTIX=1)
 *   has_optix()              -> bool
 */

// ── STB image I/O ────────────────────────────────────────────────────────────
// Define the implementation exactly once here (before any other includes).
// stb_image_write uses only fopen/fwrite/fclose, which are cross-platform on
// Windows, Linux, and macOS.  No extra platform handling is needed.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cctype>
#include <cstring>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "tachyon_render.h"   // CPU renderer + shared data structs

#if defined(MDAPY_OPTIX)
#  include "tachyon_optix_render.h"   // GPU renderer
#endif

namespace nb = nanobind;
using namespace mdapy_tachyon;

// ---------------------------------------------------------------------------
// Image I/O helpers (stb_image / stb_image_write)
// ---------------------------------------------------------------------------
using ImgArray = nb::ndarray<uint8_t, nb::shape<-1,-1,4>, nb::c_contig, nb::device::cpu>;

// save_image(path, rgba_array)
// Format is determined by file extension: .png (RGBA), .jpg/.jpeg (RGB, quality 95),
// .bmp (RGB), .tga (RGBA).  Unknown extensions default to PNG.
static void save_image(const std::string &path, ImgArray img)
{
    const int H   = (int)img.shape(0);
    const int W   = (int)img.shape(1);
    const uint8_t *data = img.data();

    // Extract lower-case extension.
    std::string ext;
    auto dot = path.rfind('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot + 1);
        for (auto &c : ext) c = (char)std::tolower((unsigned char)c);
    }

    int ok = 0;
    if (ext == "jpg" || ext == "jpeg") {
        // JPEG does not support alpha; convert RGBA → RGB.
        std::vector<uint8_t> rgb(size_t(H) * W * 3);
        for (int i = 0; i < H * W; ++i) {
            rgb[i*3+0] = data[i*4+0];
            rgb[i*3+1] = data[i*4+1];
            rgb[i*3+2] = data[i*4+2];
        }
        ok = stbi_write_jpg(path.c_str(), W, H, 3, rgb.data(), 95);
    } else if (ext == "bmp") {
        // BMP variant used here is RGB-only.
        std::vector<uint8_t> rgb(size_t(H) * W * 3);
        for (int i = 0; i < H * W; ++i) {
            rgb[i*3+0] = data[i*4+0];
            rgb[i*3+1] = data[i*4+1];
            rgb[i*3+2] = data[i*4+2];
        }
        ok = stbi_write_bmp(path.c_str(), W, H, 3, rgb.data());
    } else if (ext == "tga") {
        ok = stbi_write_tga(path.c_str(), W, H, 4, data);
    } else {
        // PNG (default, including unknown extensions): full RGBA.
        ok = stbi_write_png(path.c_str(), W, H, 4, data, W * 4);
    }

    if (!ok)
        throw std::runtime_error("save_image: failed to write '" + path + "'");
}

// load_image(path) → numpy (H, W, 4) uint8 RGBA
static nb::ndarray<nb::numpy, uint8_t, nb::shape<-1,-1,4>>
load_image(const std::string &path)
{
    int W = 0, H = 0, n = 0;
    // Always request 4 channels (RGBA) regardless of source format.
    uint8_t *raw = stbi_load(path.c_str(), &W, &H, &n, 4);
    if (!raw) {
        const char *reason = stbi_failure_reason();
        throw std::runtime_error(
            "load_image: cannot read '" + path + "': " +
            (reason ? reason : "unknown error"));
    }

    const size_t nbytes = size_t(W) * H * 4;
    uint8_t *heap = new uint8_t[nbytes];
    std::memcpy(heap, raw, nbytes);
    stbi_image_free(raw);

    nb::capsule owner(heap,
        [](void *p) noexcept { delete[] static_cast<uint8_t *>(p); });
    size_t shape[3] = { size_t(H), size_t(W), 4 };
    return nb::ndarray<nb::numpy, uint8_t, nb::shape<-1,-1,4>>(
        heap, 3, shape, owner);
}

// ---------------------------------------------------------------------------
// ndarray type aliases (shared by both backends)
// ---------------------------------------------------------------------------
using PosArray = nb::ndarray<double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using ColArray = nb::ndarray<float,  nb::shape<-1, 4>, nb::c_contig, nb::device::cpu>;
using RadArray = nb::ndarray<float,  nb::shape<-1>,    nb::c_contig, nb::device::cpu>;
using BoxArray = nb::ndarray<double, nb::shape<-1, 2, 3>, nb::c_contig, nb::device::cpu>;

// ---------------------------------------------------------------------------
// Shared helper: parse box_obj + optional colour/radius params into
// BoxEdgeData and a backing buffer.  Returns hasBox flag.
// The box_buf must outlive the returned BoxEdgeData.
// ---------------------------------------------------------------------------
static bool parse_box(nb::object box_obj,
                      float box_r, float box_g, float box_b, float box_a,
                      float box_radius,
                      std::vector<double> &box_buf,
                      BoxEdgeData &be)
{
    be.r      = box_r;
    be.g      = box_g;
    be.b      = box_b;
    be.a      = box_a;
    be.radius = box_radius;

    if (box_obj.is_none()) return false;

    BoxArray ba = nb::cast<BoxArray>(box_obj);
    const size_t M = ba.shape(0);
    if (M == 0) return false;

    // Deep-copy: the temporary BoxArray will be destroyed after this call.
    box_buf.assign(ba.data(), ba.data() + M * 6);
    be.points = box_buf.data();
    be.count  = M;
    return true;
}

// ---------------------------------------------------------------------------
// Shared helper: wrap a std::vector<uint8_t> in a numpy array via capsule.
// ---------------------------------------------------------------------------
static nb::ndarray<nb::numpy, uint8_t, nb::shape<-1,-1,4>>
wrap_image(std::vector<uint8_t> raw, int height, int width)
{
    uint8_t *heap = new uint8_t[raw.size()];
    std::memcpy(heap, raw.data(), raw.size());
    nb::capsule owner(heap,
        [](void *p) noexcept { delete[] static_cast<uint8_t *>(p); });
    size_t shape[3] = { size_t(height), size_t(width), 4 };
    return nb::ndarray<nb::numpy, uint8_t, nb::shape<-1,-1,4>>(
        heap, 3, shape, owner);
}

// ---------------------------------------------------------------------------
// NB_MODULE
// ---------------------------------------------------------------------------
NB_MODULE(_tachyon, m) {
    m.doc() = "Tachyon ray-tracing renderer for mdapy  "
              "(CPU + optional GPU/OptiX backend)";

    // -----------------------------------------------------------------------
    // Vec3
    // -----------------------------------------------------------------------
    nb::class_<Vec3>(m, "Vec3",
        "3-component double-precision vector (world-space coordinates).")
        .def(nb::init<double, double, double>(),
             nb::arg("x") = 0., nb::arg("y") = 0., nb::arg("z") = 0.)
        .def_rw("x", &Vec3::x)
        .def_rw("y", &Vec3::y)
        .def_rw("z", &Vec3::z)
        .def("__repr__", [](const Vec3 &v) {
            return "Vec3(" + std::to_string(v.x) + ", "
                           + std::to_string(v.y) + ", "
                           + std::to_string(v.z) + ")";
        });

    // -----------------------------------------------------------------------
    // CameraParams
    // -----------------------------------------------------------------------
    nb::class_<CameraParams>(m, "CameraParams",
        "Camera parameters compatible with OVITO's ViewProjectionParameters.")
        .def(nb::init<>())
        .def_rw("is_perspective",  &CameraParams::isPerspective,
                "True = perspective projection, False = orthographic.")
        .def_rw("field_of_view",   &CameraParams::fieldOfView,
                "Perspective: vertical FOV in radians.  "
                "Orthographic: viewport half-height in world units.")
        .def_rw("position",        &CameraParams::position,
                "Camera position in world coordinates.")
        .def_rw("direction",       &CameraParams::direction,
                "Camera view direction (need not be normalised).")
        .def_rw("up",              &CameraParams::up,
                "Camera up vector (need not be normalised).")
        .def_rw("znear",           &CameraParams::znear,
                "Orthographic near-plane offset for cross-section clipping. "
                "Default 0 (disabled).")
        .def_rw("dof_enabled",     &CameraParams::dofEnabled,
                "Enable depth-of-field (perspective only).")
        .def_rw("dof_focal_len",   &CameraParams::dofFocalLen,
                "DoF focal distance in world units.")
        .def_rw("dof_aperture",    &CameraParams::dofAperture,
                "DoF aperture radius (CPU) / f-number denominator (GPU).");

    // -----------------------------------------------------------------------
    // RenderParams
    // -----------------------------------------------------------------------
    nb::class_<RenderParams>(m, "RenderParams",
        "Global rendering settings shared by both CPU and GPU backends.")
        .def(nb::init<>())
        .def_rw("width",                  &RenderParams::width)
        .def_rw("height",                 &RenderParams::height)
        .def_rw("antialiasing_enabled",   &RenderParams::antialiasingEnabled)
        .def_rw("antialiasing_samples",   &RenderParams::antialiasingSamples)
        .def_rw("direct_light_enabled",   &RenderParams::directLightEnabled)
        .def_rw("shadows_enabled",        &RenderParams::shadowsEnabled,
                "Enable hard shadows (CPU only; GPU always uses shadows).")
        .def_rw("direct_light_intensity", &RenderParams::directLightIntensity)
        .def_rw("ao_enabled",             &RenderParams::aoEnabled)
        .def_rw("ao_samples",             &RenderParams::aoSamples)
        .def_rw("ao_brightness",          &RenderParams::aoBrightness)
        .def_rw("ao_max_dist",            &RenderParams::aoMaxDist,
                "Maximum AO occlusion distance.  "
                "Default: unlimited (RT_AO_MAXDIST_UNLIMITED).")
        .def_rw("bg_r",                   &RenderParams::bgR)
        .def_rw("bg_g",                   &RenderParams::bgG)
        .def_rw("bg_b",                   &RenderParams::bgB)
        .def_rw("bg_a",                   &RenderParams::bgA,
                "Alpha of background pixels in the output RGBA image.")
        .def_rw("num_threads",            &RenderParams::numThreads,
                "CPU thread count (CPU backend only).  0 = Tachyon auto.");

    // -----------------------------------------------------------------------
    // Shared render lambda (avoids duplicating 50 lines for each backend).
    // BackendT must have: render(rp, cp, pd, box*) -> vector<uint8_t>
    // -----------------------------------------------------------------------
    // Typed render helper — avoids generic lambda so nanobind can resolve
    // operator() unambiguously.  One instantiation per backend class.
    auto do_render = [](auto *self_ptr,
               const RenderParams &rp,
               const CameraParams &cp,
               PosArray positions,
               ColArray colors,
               RadArray radii,
               nb::object box_obj,
               float box_radius,
               float box_r, float box_g, float box_b, float box_a)
            -> nb::ndarray<nb::numpy, uint8_t, nb::shape<-1,-1,4>>
    {
        const size_t N = positions.shape(0);
        if (colors.shape(0) != N || radii.shape(0) != N)
            throw std::runtime_error("positions/colors/radii size mismatch");

        ParticleData pd;
        pd.positions = positions.data();
        pd.colors    = colors.data();
        pd.radii     = radii.data();
        pd.count     = N;

        std::vector<double> box_buf;
        BoxEdgeData be{};
        bool hasBox = parse_box(box_obj,
            box_r, box_g, box_b, box_a, box_radius, box_buf, be);

        std::vector<uint8_t> raw = self_ptr->render(
            rp, cp, pd, hasBox ? &be : nullptr);

        return wrap_image(std::move(raw), rp.height, rp.width);
    };

    // Macro to bind render() for a concrete class — avoids auto &self ambiguity.
#define BIND_RENDER(CLS_VAR, CLASS_T)                              \
    (CLS_VAR).def("render",                                        \
        [do_render](CLASS_T &self,                                  \
               const RenderParams &rp,                             \
               const CameraParams &cp,                             \
               PosArray positions, ColArray colors, RadArray radii, \
               nb::object box_obj, float box_radius,               \
               float box_r, float box_g, float box_b, float box_a) \
            -> nb::ndarray<nb::numpy, uint8_t, nb::shape<-1,-1,4>> \
        {                                                           \
            return do_render(&self, rp, cp, positions, colors,     \
                radii, box_obj, box_radius,                        \
                box_r, box_g, box_b, box_a);                       \
        },                                                          \
        nb::arg("render_params"),  nb::arg("camera_params"),       \
        nb::arg("positions"),      nb::arg("colors"),              \
        nb::arg("radii"),                                           \
        nb::arg("box_edges")  = nb::none(),                        \
        nb::arg("box_radius") = 0.05f,                             \
        nb::arg("box_r") = 1.0f, nb::arg("box_g") = 1.0f,        \
        nb::arg("box_b") = 1.0f, nb::arg("box_a") = 1.0f,        \
        "Render sphere particles and optional simulation-cell box edges.\n\n" \
        "positions  : (N, 3) float64\n"                           \
        "colors     : (N, 4) float32  RGBA in [0, 1]\n"          \
        "radii      : (N,)   float32\n"                           \
        "box_edges  : (M, 2, 3) float64 or None\n"               \
        "Returns: numpy (H, W, 4) uint8 RGBA")

    // -----------------------------------------------------------------------
    // TachyonRenderer  (CPU backend)
    // -----------------------------------------------------------------------
    {
        auto cls = nb::class_<TachyonRenderer>(m, "TachyonRenderer",
            "CPU ray-tracing renderer using the classic Tachyon engine.\n"
            "Always available regardless of GPU hardware.");
        cls.def(nb::init<>());
        BIND_RENDER(cls, TachyonRenderer);
    }

    // -----------------------------------------------------------------------
    // TachyonOptiXRenderer  (GPU backend, only when compiled with OptiX)
    // -----------------------------------------------------------------------
#if defined(MDAPY_OPTIX)
    {
        auto cls = nb::class_<TachyonOptiXRenderer>(m, "TachyonOptiXRenderer",
            "GPU ray-tracing renderer using TachyonOptiX (NVIDIA OptiX 7+).\n"
            "Only available when mdapy was built with CUDA and OptiX support.\n"
            "Typically 5-20x faster than the CPU backend for large scenes.");
        cls.def(nb::init<>(),
            "Construct and initialise the OptiX context.\n"
            "Raises RuntimeError if no CUDA-capable GPU is available.");
        BIND_RENDER(cls, TachyonOptiXRenderer);
    }
    m.def("has_optix", []{ return true; },
          "Return True if this mdapy build includes GPU/OptiX support.");
#else
    m.def("has_optix", []{ return false; },
          "Return False: this mdapy build was compiled without OptiX support.");
#endif

    // -----------------------------------------------------------------------
    // Image I/O
    // -----------------------------------------------------------------------
    m.def("save_image", &save_image,
          nb::arg("path"), nb::arg("img"),
          "Save a (H, W, 4) uint8 RGBA image to *path*.\n\n"
          "Format is determined by the file extension:\n"
          "  .png          — RGBA (alpha preserved)\n"
          "  .jpg / .jpeg  — RGB  (alpha discarded, quality 95)\n"
          "  .bmp          — RGB\n"
          "  .tga          — RGBA\n"
          "  other         — treated as .png\n\n"
          "Cross-platform: uses stb_image_write (fopen/fwrite).");

    m.def("load_image", &load_image,
          nb::arg("path"),
          "Load an image from *path* and return a (H, W, 4) uint8 RGBA array.\n\n"
          "Supports PNG, JPEG, BMP, TGA, GIF, PSD, HDR, PIC, PNM (via stb_image).\n"
          "Raises RuntimeError if the file cannot be read.");

} // NB_MODULE