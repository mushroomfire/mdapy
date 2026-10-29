// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The rays of a picture traced with Metal: the shader follows render.cpp line
// by line, in the same single precision, with the same random numbers.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "visualization/render_gpu.h"

namespace mdapy::visualization::gpu {
namespace {

const char* const kShader = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Node {
    float lo[3];
    uint first;
    float hi[3];
    uint count;
};

struct Params {
    float center[3], forward[3], right[3], up[3];
    float half_width, half_height, aperture, focal_distance;
    float start, nearest, farthest, epsilon;
    float key[3], highlight[3], fill[3], key_tangent[3], key_bitangent[3];
    float ambient, diffuse, specular, shininess, metalness, fill_intensity, base_light;
    float ao_distance, ao_strength, shadow_strength, fog, light_size, cone;
    float background[3];
    int perspective, ao_samples, shadows, flat, toon_levels, bounces, samples;
    int width, height;
    uint refs;
};

constant uint kKindShift = 30;
constant uint kIndexMask = (1u << 30) - 1u;
constant uint kSphere = 0;
constant uint kCylinder = 1;
constant uint kTriangle = 2;

struct World {
    device const float* sphere_center;
    device const float* sphere_radius;
    device const float* sphere_color;
    device const int* sphere_id;
    device const float* cylinder_end;
    device const float* cylinder_radius;
    device const float* cylinder_color;
    device const int* cylinder_id;
    device const float* triangle_vertex;
    device const float* triangle_color;
    device const float* triangle_alpha;
    device const int* triangle_id;
    device const Node* nodes;
    device const uint* refs;
    uint count;
};

struct Ray {
    float3 origin, direction, inverse;
};

struct Hit {
    float t;
    uint ref;
};

float3 load3(device const float* p) { return float3(p[0], p[1], p[2]); }
float3 vec3(constant float* p) { return float3(p[0], p[1], p[2]); }

float3 unit(float3 a) {
    const float length = sqrt(dot(a, a));
    return length > 0 ? a * (1.0f / length) : a;
}

Ray make_ray(float3 origin, float3 direction) {
    Ray ray;
    ray.origin = origin;
    ray.direction = direction;
    for (int e = 0; e < 3; ++e) {
        const float d = fabs(direction[e]) < 1e-20f ? copysign(1e-20f, direction[e]) : direction[e];
        ray.inverse[e] = 1.0f / d;
    }
    return ray;
}

float intersect(thread const World& w, uint ref, thread const Ray& ray, float t_min, float t_max) {
    const uint i = ref & kIndexMask;
    const uint kind = ref >> kKindShift;
    if (kind == kSphere) {
        const float3 oc = ray.origin - load3(w.sphere_center + 3 * i);
        const float r = w.sphere_radius[i];
        const float b = dot(oc, ray.direction);
        const float c = dot(oc, oc) - r * r;
        const float disc = b * b - c;
        if (disc < 0) {
            return INFINITY;
        }
        const float s = sqrt(disc);
        float t = -b - s;
        if (t <= t_min) {
            t = -b + s;
        }
        return t > t_min && t < t_max ? t : INFINITY;
    }
    if (kind == kCylinder) {
        const float3 p0 = load3(w.cylinder_end + 6 * i);
        const float3 axis = load3(w.cylinder_end + 6 * i + 3) - p0;
        const float length = sqrt(dot(axis, axis));
        if (length == 0) {
            return INFINITY;
        }
        const float3 a = axis * (1.0f / length);
        const float3 oc = ray.origin - p0;
        const float3 d = ray.direction - a * dot(ray.direction, a);
        const float3 o = oc - a * dot(oc, a);
        const float r = w.cylinder_radius[i];
        const float qa = dot(d, d);
        if (qa < 1e-12f) {
            return INFINITY;
        }
        const float qb = dot(d, o);
        const float qc = dot(o, o) - r * r;
        const float disc = qb * qb - qa * qc;
        if (disc < 0) {
            return INFINITY;
        }
        const float s = sqrt(disc);
        const float roots[2] = {(-qb - s) / qa, (-qb + s) / qa};
        for (int k = 0; k < 2; ++k) {
            const float t = roots[k];
            if (t > t_min && t < t_max) {
                const float along = dot(oc + ray.direction * t, a);
                if (along >= 0 && along <= length) {
                    return t;
                }
            }
        }
        return INFINITY;
    }
    device const float* v = w.triangle_vertex + 9 * i;
    const float3 v0 = load3(v), e1 = load3(v + 3) - v0, e2 = load3(v + 6) - v0;
    const float3 p = cross(ray.direction, e2);
    const float det = dot(e1, p);
    if (fabs(det) < 1e-20f) {
        return INFINITY;
    }
    const float inverse = 1.0f / det;
    const float3 s = ray.origin - v0;
    const float u = dot(s, p) * inverse;
    if (u < 0 || u > 1) {
        return INFINITY;
    }
    const float3 q = cross(s, e1);
    const float wv = dot(ray.direction, q) * inverse;
    if (wv < 0 || u + wv > 1) {
        return INFINITY;
    }
    const float t = dot(e2, q) * inverse;
    return t > t_min && t < t_max ? t : INFINITY;
}

float enter(device const Node& node, thread const Ray& ray, float t_min, float t_max) {
    float t_in = t_min, t_out = t_max;
    for (int e = 0; e < 3; ++e) {
        float t0 = (node.lo[e] - ray.origin[e]) * ray.inverse[e];
        float t1 = (node.hi[e] - ray.origin[e]) * ray.inverse[e];
        if (t0 > t1) {
            const float swap = t0;
            t0 = t1;
            t1 = swap;
        }
        t_in = t0 > t_in ? t0 : t_in;
        t_out = t1 < t_out ? t1 : t_out;
    }
    return t_in <= t_out ? t_in : INFINITY;
}

void closest(thread const World& w, thread const Ray& ray, float t_min, thread Hit& hit) {
    if (w.count == 0) {
        return;
    }
    uint stack[96];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        device const Node& node = w.nodes[stack[--top]];
        if (node.count > 0) {
            for (uint k = node.first; k < node.first + node.count; ++k) {
                const float t = intersect(w, w.refs[k], ray, t_min, hit.t);
                if (t < hit.t) {
                    hit.t = t;
                    hit.ref = w.refs[k];
                }
            }
            continue;
        }
        const float near_left = enter(w.nodes[node.first], ray, t_min, hit.t);
        const float near_right = enter(w.nodes[node.first + 1], ray, t_min, hit.t);
        if (near_left <= near_right) {
            if (near_right < INFINITY) {
                stack[top++] = node.first + 1;
            }
            if (near_left < INFINITY) {
                stack[top++] = node.first;
            }
        } else {
            if (near_left < INFINITY) {
                stack[top++] = node.first;
            }
            if (near_right < INFINITY) {
                stack[top++] = node.first + 1;
            }
        }
    }
}

bool occluded(thread const World& w, thread const Ray& ray, float t_min, float t_max) {
    if (w.count == 0) {
        return false;
    }
    uint stack[96];
    int top = 0;
    stack[top++] = 0;
    while (top > 0) {
        device const Node& node = w.nodes[stack[--top]];
        if (enter(node, ray, t_min, t_max) == INFINITY) {
            continue;
        }
        if (node.count > 0) {
            for (uint k = node.first; k < node.first + node.count; ++k) {
                const uint ref = w.refs[k];
                if ((ref >> kKindShift) == kTriangle && w.triangle_alpha[ref & kIndexMask] < 1.0f) {
                    continue;
                }
                if (intersect(w, ref, ray, t_min, t_max) < t_max) {
                    return true;
                }
            }
            continue;
        }
        stack[top++] = node.first;
        stack[top++] = node.first + 1;
    }
    return false;
}

float3 normal_at(thread const World& w, uint ref, float3 point, float3 direction) {
    const uint i = ref & kIndexMask;
    const uint kind = ref >> kKindShift;
    float3 n;
    if (kind == kSphere) {
        n = unit(point - load3(w.sphere_center + 3 * i));
    } else if (kind == kCylinder) {
        const float3 p0 = load3(w.cylinder_end + 6 * i);
        const float3 a = unit(load3(w.cylinder_end + 6 * i + 3) - p0);
        const float3 rel = point - p0;
        n = unit(rel - a * dot(rel, a));
    } else {
        device const float* v = w.triangle_vertex + 9 * i;
        n = unit(cross(load3(v + 3) - load3(v), load3(v + 6) - load3(v)));
    }
    return dot(n, direction) > 0 ? n * -1.0f : n;
}

float3 color_of(thread const World& w, uint ref) {
    const uint i = ref & kIndexMask;
    const uint kind = ref >> kKindShift;
    if (kind == kSphere) {
        return load3(w.sphere_color + 3 * i);
    }
    if (kind == kCylinder) {
        return load3(w.cylinder_color + 3 * i);
    }
    return load3(w.triangle_color + 3 * i);
}

float alpha_of(thread const World& w, uint ref) {
    return (ref >> kKindShift) == kTriangle ? w.triangle_alpha[ref & kIndexMask] : 1.0f;
}

int2 id_of(thread const World& w, uint ref) {
    const uint i = ref & kIndexMask;
    const uint kind = ref >> kKindShift;
    const int id = kind == kSphere ? w.sphere_id[i] : kind == kCylinder ? w.cylinder_id[i] : w.triangle_id[i];
    return int2(int(kind), id);
}

struct Random {
    uint state;
};

Random random_from(uint seed) {
    Random r;
    r.state = seed * 747796405u + 2891336453u;
    return r;
}

float next(thread Random& r) {
    r.state = r.state * 747796405u + 2891336453u;
    uint word = ((r.state >> ((r.state >> 28u) + 4u)) ^ r.state) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word >> 8) * (1.0f / 16777216.0f);
}

uint mix_seed(uint a, uint b) {
    uint h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u + (a << 6) + (a >> 2));
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    return h;
}

// with count > 0, in the k-th of count rings of equal weight, turned by a
// golden angle each
float3 cosine(float3 n, thread Random& random, int k, int count, float turn) {
    const float3 helper = fabs(n[0]) < 0.9f ? float3(1, 0, 0) : float3(0, 1, 0);
    const float3 tangent = unit(cross(n, helper));
    const float3 bitangent = cross(n, tangent);
    const float u = count > 0 ? (float(k) + next(random)) / float(count) : next(random);
    const float phi = 6.2831853f * (count > 0 ? turn + 0.618034f * float(k) : next(random));
    const float r = sqrt(u);
    return tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + n * sqrt(max(0.0f, 1.0f - u));
}

float3 toward_light(constant Params& P, thread Random& random) {
    const float3 key = vec3(P.key);
    if (P.light_size <= 0) {
        return key;
    }
    const float z = 1.0f - next(random) * (1.0f - P.cone);
    const float phi = 6.2831853f * next(random);
    const float r = sqrt(max(0.0f, 1.0f - z * z));
    return vec3(P.key_tangent) * (r * cos(phi)) + vec3(P.key_bitangent) * (r * sin(phi)) + key * z;
}

// metals: less of the color reflected diffusely, the rest in the color of
// the highlights
float body(constant Params& P) { return 1.0f - 0.6f * clamp(P.metalness, 0.0f, 1.0f); }

float3 tint(constant Params& P, float3 base) {
    const float m = clamp(P.metalness, 0.0f, 1.0f);
    return float3(1.0f - m) + base * m;
}

// a highlight: the light from toward mirrored about n, seen along ray
float phong(constant Params& P, float3 n, float3 toward, float3 ray) {
    if (dot(n, toward) <= 0) {
        return 0.0f;
    }
    const float3 mirrored = n * (2.0f * dot(n, toward)) - toward;
    return P.specular * pow(max(0.0f, -dot(mirrored, ray)), P.shininess);
}

float3 shade(thread const World& w, constant Params& P, Hit hit, thread const Ray& ray, thread Random& random) {
    const float3 point = ray.origin + ray.direction * hit.t;
    const float3 n = normal_at(w, hit.ref, point, ray.direction);
    const float3 base = color_of(w, hit.ref);
    const float3 lifted = point + n * P.epsilon;
    float ao = 1.0f;
    if (P.ao_samples > 0 && P.ao_strength > 0) {
        const float3 helper = fabs(n[0]) < 0.9f ? float3(1, 0, 0) : float3(0, 1, 0);
        const float3 tangent = unit(cross(n, helper));
        const float3 bitangent = cross(n, tangent);
        const float turn = next(random);
        int blocked = 0;
        for (int k = 0; k < P.ao_samples; ++k) {
            const float u = (float(k) + next(random)) / float(P.ao_samples);
            const float phi = 6.2831853f * (turn + 0.618034f * float(k));
            const float r = sqrt(u);
            const float3 d = tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + n * sqrt(max(0.0f, 1.0f - u));
            blocked += occluded(w, make_ray(lifted, d), 0.0f, P.ao_distance) ? 1 : 0;
        }
        ao = 1.0f - P.ao_strength * float(blocked) / float(P.ao_samples);
    }
    if (P.flat) {
        return base * ao;
    }
    const float3 key = vec3(P.key), fill = vec3(P.fill);
    const float lit = max(0.0f, dot(n, key));
    float shadow = 1.0f;
    if (P.shadows && lit > 0) {
        const float3 l = toward_light(P, random);
        if (dot(n, l) <= 0 || occluded(w, make_ray(lifted, l), 0.0f, INFINITY)) {
            shadow = 1.0f - P.shadow_strength;
        }
    }
    float diffuse = lit * shadow + P.fill_intensity * max(0.0f, dot(n, fill));
    if (P.toon_levels > 0) {
        const float levels = float(P.toon_levels);
        diffuse = min(1.0f, floor(diffuse * levels + 0.5f) / levels);
    }
    float shine = 0;
    if (P.specular > 0) {
        shine = phong(P, n, vec3(P.highlight), ray.direction) * shadow;
        if (P.toon_levels > 0) {
            shine = shine > 0.5f * P.specular ? P.specular : 0.0f;
        }
    }
    const float light = P.base_light + P.ambient * ao + P.diffuse * diffuse * (0.5f + 0.5f * ao);
    return base * (body(P) * light) + tint(P, base) * shine;
}

float3 path(thread const World& w, constant Params& P, Hit first, Ray primary, thread Random& random, int sample,
            float turn) {
    float3 radiance = float3(0, 0, 0), weight = float3(1, 1, 1);
    Hit hit = first;
    Ray ray = primary;
    for (int bounce = 0;; ++bounce) {
        const float3 point = ray.origin + ray.direction * hit.t;
        const float3 n = normal_at(w, hit.ref, point, ray.direction);
        const float3 lifted = point + n * P.epsilon;
        const float3 color = color_of(w, hit.ref);
        const float3 reflected = weight * color * body(P);
        const float3 l = toward_light(P, random);
        const float lit = dot(n, l);
        if (lit > 0 && !occluded(w, make_ray(lifted, l), 0.0f, INFINITY)) {
            radiance = radiance + reflected * (P.diffuse * lit);
            if (bounce == 0 && P.specular > 0) {
                radiance = radiance + tint(P, color) * phong(P, n, vec3(P.highlight), ray.direction);
            }
        }
        const Ray next_ray = make_ray(lifted, cosine(n, random, sample, bounce == 0 ? P.samples : 0, turn));
        Hit following;
        following.t = INFINITY;
        following.ref = 0;
        closest(w, next_ray, 0.0f, following);
        if (following.t == INFINITY) {
            radiance = radiance + reflected * P.ambient;
            break;
        }
        if (bounce == P.bounces) {
            break;
        }
        weight = reflected;
        hit = following;
        ray = next_ray;
    }
    return radiance;
}

kernel void trace_subpixels(device const float* sphere_center [[buffer(0)]],
                            device const float* sphere_radius [[buffer(1)]],
                            device const float* sphere_color [[buffer(2)]],
                            device const int* sphere_id [[buffer(3)]],
                            device const float* cylinder_end [[buffer(4)]],
                            device const float* cylinder_radius [[buffer(5)]],
                            device const float* cylinder_color [[buffer(6)]],
                            device const int* cylinder_id [[buffer(7)]],
                            device const float* triangle_vertex [[buffer(8)]],
                            device const float* triangle_color [[buffer(9)]],
                            device const float* triangle_alpha [[buffer(10)]],
                            device const int* triangle_id [[buffer(11)]],
                            device const Node* nodes [[buffer(12)]],
                            device const uint* refs [[buffer(13)]],
                            constant Params& P [[buffer(14)]],
                            device float* rgba [[buffer(15)]],
                            device float* depth [[buffer(16)]],
                            device int2* ids [[buffer(17)]],
                            constant uint& first_row [[buffer(18)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint x = gid.x, y = gid.y + first_row;
    if (x >= uint(P.width) || y >= uint(P.height)) {
        return;
    }
    World w;
    w.sphere_center = sphere_center;
    w.sphere_radius = sphere_radius;
    w.sphere_color = sphere_color;
    w.sphere_id = sphere_id;
    w.cylinder_end = cylinder_end;
    w.cylinder_radius = cylinder_radius;
    w.cylinder_color = cylinder_color;
    w.cylinder_id = cylinder_id;
    w.triangle_vertex = triangle_vertex;
    w.triangle_color = triangle_color;
    w.triangle_alpha = triangle_alpha;
    w.triangle_id = triangle_id;
    w.nodes = nodes;
    w.refs = refs;
    w.count = P.refs;
    const uint p = y * uint(P.width) + x;
    const uint seed = mix_seed(x, y);
    const float turn = float(mix_seed(seed, 0x51ED27u) >> 8) * (1.0f / 16777216.0f);
    const float3 center = vec3(P.center), forward = vec3(P.forward), right = vec3(P.right), up = vec3(P.up);
    const float3 background = vec3(P.background);
    float3 color = float3(0, 0, 0);
    float coverage = 0.0f;
    float first_depth = INFINITY;
    int2 first_id = int2(-1, 0);
    for (int sample = 0; sample < P.samples; ++sample) {
        Random random = random_from(sample == 0 ? seed : mix_seed(seed, uint(sample)));
        const float jx = sample == 0 ? 0.5f : next(random);
        const float jy = sample == 0 ? 0.5f : next(random);
        const float px = (2.0f * (float(x) + jx) / float(P.width)) - 1.0f;
        const float py = 1.0f - (2.0f * (float(y) + jy) / float(P.height));
        Ray ray;
        if (P.perspective) {
            const float3 d = unit(forward + right * (px * P.half_width) + up * (py * P.half_height));
            ray = make_ray(center, d);
            if (P.aperture > 0) {
                const float3 sharp = center + d * (P.focal_distance / dot(d, forward));
                const float u = next(random), phi = 6.2831853f * next(random);
                const float r = P.aperture * sqrt(u);
                const float3 from = center + right * (r * cos(phi)) + up * (r * sin(phi));
                ray = make_ray(from, unit(sharp - from));
            }
        } else {
            ray = make_ray(center + right * (px * P.half_width) + up * (py * P.half_height) + forward * P.start, forward);
        }
        float remaining = 1.0f, t_min = 0.0f;
        for (int layer = 0; layer < 8 && remaining > 1e-3f; ++layer) {
            Hit hit;
            hit.t = INFINITY;
            hit.ref = 0;
            closest(w, ray, t_min, hit);
            if (hit.t == INFINITY) {
                break;
            }
            if (sample == 0 && layer == 0) {
                first_depth = hit.t + (P.perspective ? 0.0f : P.start);
                first_id = id_of(w, hit.ref);
            }
            float3 c = P.bounces > 0 ? path(w, P, hit, ray, random, sample, turn) : shade(w, P, hit, ray, random);
            if (P.fog > 0 && P.farthest > P.nearest) {
                const float f = P.fog * clamp((hit.t + P.start - P.nearest) / (P.farthest - P.nearest), 0.0f, 1.0f);
                c = c * (1.0f - f) + background * f;
            }
            const float a = alpha_of(w, hit.ref);
            color = color + c * (remaining * a);
            remaining *= 1.0f - a;
            t_min = hit.t + P.epsilon;
        }
        coverage += 1.0f - remaining;
    }
    color = color * (1.0f / float(P.samples));
    coverage /= float(P.samples);
    rgba[4 * p + 0] = color[0];
    rgba[4 * p + 1] = color[1];
    rgba[4 * p + 2] = color[2];
    rgba[4 * p + 3] = coverage;
    depth[p] = first_depth;
    ids[p] = first_id;
}
)METAL";

// The device, its queue and the compiled shader, made once.
struct Device {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    std::string error;
};

Device& shared() {
    static Device d;
    static std::once_flag once;
    std::call_once(once, [] {
        @autoreleasepool {
            d.device = MTLCreateSystemDefaultDevice();
            if (d.device == nil) {
                d.error = "no Metal device";
                return;
            }
            MTLCompileOptions* options = [MTLCompileOptions new];
            // the shader compares with infinity: no fast math
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            NSError* error = nil;
            id<MTLLibrary> library = [d.device newLibraryWithSource:@(kShader)
                                                            options:options
                                                              error:&error];
            if (library == nil) {
                d.error = std::string("the shader does not compile: ") +
                          error.localizedDescription.UTF8String;
                d.device = nil;
                return;
            }
            id<MTLFunction> function = [library newFunctionWithName:@"trace_subpixels"];
            d.pipeline = [d.device newComputePipelineStateWithFunction:function error:&error];
            if (d.pipeline == nil) {
                d.error = std::string("no pipeline: ") + error.localizedDescription.UTF8String;
                d.device = nil;
                return;
            }
            d.queue = [d.device newCommandQueue];
        }
    });
    return d;
}

}  // namespace

bool available() { return shared().device != nil; }

std::string device() {
    Device& d = shared();
    return d.device != nil ? std::string(d.device.name.UTF8String) + " (Metal)" : std::string();
}

void trace(const Geometry& g, const Params& params, float* rgba, float* depth, std::int64_t* ids,
           const Control& control) {
    Device& d = shared();
    if (d.device == nil) {
        throw std::runtime_error("no GPU: " + d.error);
    }
    const auto width = static_cast<std::size_t>(params.width);
    const auto height = static_cast<std::size_t>(params.height);
    const std::size_t subpixels = width * height;
    @autoreleasepool {
        // a buffer of the bytes given, never empty
        const auto buffer = [&](const void* data, std::size_t bytes) {
            static const std::uint64_t zero[2] = {0, 0};
            return [d.device newBufferWithBytes:(bytes > 0 ? data : zero)
                                         length:std::max<std::size_t>(bytes, sizeof(zero))
                                        options:MTLResourceStorageModeShared];
        };
        const std::size_t f = sizeof(float), i = sizeof(std::int32_t);
        NSArray<id<MTLBuffer>>* inputs = @[
            buffer(g.sphere_center, 3 * g.spheres * f), buffer(g.sphere_radius, g.spheres * f),
            buffer(g.sphere_color, 3 * g.spheres * f), buffer(g.sphere_id, g.spheres * i),
            buffer(g.cylinder_end, 6 * g.cylinders * f), buffer(g.cylinder_radius, g.cylinders * f),
            buffer(g.cylinder_color, 3 * g.cylinders * f), buffer(g.cylinder_id, g.cylinders * i),
            buffer(g.triangle_vertex, 9 * g.triangles * f), buffer(g.triangle_color, 3 * g.triangles * f),
            buffer(g.triangle_alpha, g.triangles * f), buffer(g.triangle_id, g.triangles * i),
            buffer(g.nodes, g.node_count * 32), buffer(g.refs, params.refs * sizeof(std::uint32_t))
        ];
        id<MTLBuffer> out_rgba = [d.device newBufferWithLength:4 * subpixels * f
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_depth = [d.device newBufferWithLength:subpixels * f
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_ids = [d.device newBufferWithLength:2 * subpixels * i
                                                      options:MTLResourceStorageModeShared];
        if (out_rgba == nil || out_depth == nil || out_ids == nil) {
            throw std::runtime_error("no GPU memory for the picture");
        }
        // rows in tiles: first of some million paths, then twice as many rows
        // after a tile shorter than 0.1 s and half as many after one longer
        // than 0.4 s, so the GPU is busy yet each dispatch short, with
        // cancellation and progress between them
        const std::size_t paths = width * static_cast<std::size_t>(std::max(1, params.samples)) *
                                  static_cast<std::size_t>(1 + params.bounces);
        std::size_t rows = std::min(height, std::max<std::size_t>(1, (std::size_t{1} << 21) / paths));
        const NSUInteger group_width = std::min<NSUInteger>(d.pipeline.threadExecutionWidth, width);
        const NSUInteger group_height = std::max<NSUInteger>(1, std::min<NSUInteger>(
            d.pipeline.maxTotalThreadsPerThreadgroup / group_width, 4));
        for (std::size_t row = 0; row < height;) {
            if (control.cancel != nullptr && control.cancel->cancelled()) {
                throw Cancelled();
            }
            const std::size_t count = std::min(rows, height - row);
            const auto first = static_cast<std::uint32_t>(row);
            const auto started = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> commands = [d.queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:d.pipeline];
            for (NSUInteger k = 0; k < inputs.count; ++k) {
                [encoder setBuffer:inputs[k] offset:0 atIndex:k];
            }
            [encoder setBytes:&params length:sizeof(Params) atIndex:14];
            [encoder setBuffer:out_rgba offset:0 atIndex:15];
            [encoder setBuffer:out_depth offset:0 atIndex:16];
            [encoder setBuffer:out_ids offset:0 atIndex:17];
            [encoder setBytes:&first length:sizeof(first) atIndex:18];
            [encoder dispatchThreads:MTLSizeMake(width, count, 1)
                threadsPerThreadgroup:MTLSizeMake(group_width, group_height, 1)];
            [encoder endEncoding];
            [commands commit];
            [commands waitUntilCompleted];
            if (commands.error != nil) {
                throw std::runtime_error(std::string("the GPU failed: ") +
                                         commands.error.localizedDescription.UTF8String);
            }
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            row += count;
            if (control.monitor &&
                !control.monitor(static_cast<std::int64_t>(row), static_cast<std::int64_t>(height))) {
                throw Cancelled();
            }
            if (seconds < 0.1) {
                rows = std::min(height, 2 * rows);
            } else if (seconds > 0.4) {
                rows = std::max<std::size_t>(1, rows / 2);
            }
        }
        std::memcpy(rgba, out_rgba.contents, 4 * subpixels * f);
        std::memcpy(depth, out_depth.contents, subpixels * f);
        const auto* pairs = static_cast<const std::int32_t*>(out_ids.contents);
        for (std::size_t p = 0; p < subpixels; ++p) {
            const std::int32_t kind = pairs[2 * p];
            ids[p] = kind < 0 ? -1 : (static_cast<std::int64_t>(kind) << 32) + pairs[2 * p + 1];
        }
    }
}

}  // namespace mdapy::visualization::gpu
