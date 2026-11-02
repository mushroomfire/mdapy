// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The rays of a picture traced with Direct3D 12: the shader of render_vulkan.comp
// in HLSL, line by line, in the same single precision, with the same random
// numbers. With RAY_QUERY, the ray tracing units of the GPU find the candidates
// in their own hierarchy of the bounding boxes of the primitives, and the
// shader intersects them as without. mdapy holds the DXIL of both; after a change
// compile them again with dxc of the Windows SDK 10.0.26100.0 (a test compares):
//
//     dxc -nologo -T cs_6_2 -E main -HV 2021 -Gis -O3 -WX -Fo render_d3d12.dxil render_d3d12.hlsl
//     dxc -nologo -T cs_6_5 -E main -HV 2021 -Gis -O3 -WX -DRAY_QUERY -Fo render_d3d12_rt.dxil render_d3d12.hlsl

struct Params {
    float3 center, forward, right, up;
    float half_width, half_height, aperture, focal_distance;
    float start, nearest, farthest, epsilon;
    float3 key, highlight, fill, key_tangent, key_bitangent;
    float ambient, diffuse, specular, shininess, metalness, fill_intensity, base_light;
    float ao_distance, ao_strength, shadow_strength, fog, light_size, cone;
    float3 background;
    int perspective, ao_samples, shadows, flat_shading, toon_levels, bounces, samples;
    int width, height;
    uint refs;
};

// where each array of the scene starts in words, and the rows of one dispatch
cbuffer Tile : register(b0) {
    uint A_sphere_center, A_sphere_radius, A_sphere_color, A_sphere_id;
    uint A_cylinder_end, A_cylinder_radius, A_cylinder_color, A_cylinder_id;
    uint A_triangle_vertex, A_triangle_color, A_triangle_alpha, A_triangle_id;
    uint A_refs;
    uint first_row;
    uint rows;
    uint geometry_kinds;  // of the geometries of the GPU's hierarchy, two bits each
};

// the scene, its parameters first; the references of the leaves are
// kind << 30 | index
ByteAddressBuffer scene : register(t0);
// the subpixels of the rows first_row to first_row + rows
RWStructuredBuffer<float4> rgba : register(u0);
RWStructuredBuffer<float> depths : register(u1);
RWStructuredBuffer<int2> ids : register(u2);
// the nodes of the hierarchy, two quads each: (lo, first) and (hi, count)
ByteAddressBuffer nodes : register(t1);
#ifdef RAY_QUERY
// the bounding boxes of the primitives
RaytracingAccelerationStructure boxes : register(t2);
#endif

static const uint kParams = 244;  // bytes, before the words of the scene
static const uint kKindShift = 30u;
static const uint kIndexMask = (1u << 30) - 1u;
static const uint kSphere = 0u;
static const uint kCylinder = 1u;
static const uint kTriangle = 2u;
static const float kInfinity = asfloat(0x7F800000u);
static const float kLargest = asfloat(0x7F7FFFFFu);

static Params P;

struct Ray {
    float3 origin, direction, inverse;
};

struct Hit {
    float t;
    uint ref;
};

uint word(uint i) { return scene.Load(kParams + 4u * i); }
float load(uint i) { return asfloat(word(i)); }
float3 load3(uint i) { return float3(load(i), load(i + 1u), load(i + 2u)); }

float3 unit(float3 a) {
    const float len = sqrt(dot(a, a));
    return len > 0.0 ? a * (1.0 / len) : a;
}

Ray make_ray(float3 origin, float3 direction) {
    Ray ray;
    ray.origin = origin;
    ray.direction = direction;
    for (int e = 0; e < 3; ++e) {
        const float negative = (asuint(direction[e]) & 0x80000000u) != 0u ? -1e-20 : 1e-20;
        const float d = abs(direction[e]) < 1e-20 ? negative : direction[e];
        ray.inverse[e] = 1.0 / d;
    }
    return ray;
}

float intersect(uint ref, Ray ray, float t_min, float t_max) {
    const uint i = ref & kIndexMask;
    const uint kind = ref >> kKindShift;
    if (kind == kSphere) {
        const float3 oc = ray.origin - load3(A_sphere_center + 3u * i);
        const float r = load(A_sphere_radius + i);
        const float b = dot(oc, ray.direction);
        const float c = dot(oc, oc) - r * r;
        const float disc = b * b - c;
        if (disc < 0.0) {
            return kInfinity;
        }
        const float s = sqrt(disc);
        float t = -b - s;
        if (t <= t_min) {
            t = -b + s;
        }
        return t > t_min && t < t_max ? t : kInfinity;
    }
    if (kind == kCylinder) {
        const float3 p0 = load3(A_cylinder_end + 6u * i);
        const float3 axis = load3(A_cylinder_end + 6u * i + 3u) - p0;
        const float len = sqrt(dot(axis, axis));
        if (len == 0.0) {
            return kInfinity;
        }
        const float3 a = axis * (1.0 / len);
        const float3 oc = ray.origin - p0;
        const float3 d = ray.direction - a * dot(ray.direction, a);
        const float3 o = oc - a * dot(oc, a);
        const float r = load(A_cylinder_radius + i);
        const float qa = dot(d, d);
        const float qb = dot(d, o);
        const float qc = dot(o, o) - r * r;
        const float disc = qb * qb - qa * qc;
        const bool across = qa >= 1e-12;
        if (across ? disc < 0.0 : qc > 0.0) {
            return kInfinity;
        }
        const float s = across ? sqrt(disc) : 0.0;
        if (across) {
            const float t = (-qb - s) / qa;
            if (t > t_min && t < t_max) {
                const float along = dot(oc + ray.direction * t, a);
                if (along >= 0.0 && along <= len) {
                    return t;
                }
            }
        }
        // in through the disk of an end
        const float da = dot(ray.direction, a);
        if (da != 0.0) {
            const float t = ((da > 0.0 ? 0.0 : len) - dot(oc, a)) / da;
            const float3 q = o + d * t;
            if (t > t_min && t < t_max && dot(q, q) <= r * r) {
                return t;
            }
        }
        if (across) {
            const float t = (-qb + s) / qa;
            if (t > t_min && t < t_max) {
                const float along = dot(oc + ray.direction * t, a);
                if (along >= 0.0 && along <= len) {
                    return t;
                }
            }
        }
        return kInfinity;
    }
    const uint v = A_triangle_vertex + 9u * i;
    const float3 v0 = load3(v), e1 = load3(v + 3u) - v0, e2 = load3(v + 6u) - v0;
    const float3 p = cross(ray.direction, e2);
    const float det = dot(e1, p);
    if (abs(det) < 1e-20) {
        return kInfinity;
    }
    const float inv = 1.0 / det;
    const float3 s = ray.origin - v0;
    const float u = dot(s, p) * inv;
    if (u < 0.0 || u > 1.0) {
        return kInfinity;
    }
    const float3 q = cross(s, e1);
    const float wv = dot(ray.direction, q) * inv;
    if (wv < 0.0 || u + wv > 1.0) {
        return kInfinity;
    }
    const float t = dot(e2, q) * inv;
    return t > t_min && t < t_max ? t : kInfinity;
}

#ifdef RAY_QUERY
// a candidate of a ray query as a reference: its geometry gives its kind
uint candidate(uint geometry, uint primitive) {
    return (((geometry_kinds >> (2u * geometry)) & 3u) << kKindShift) | primitive;
}

void closest(Ray ray, float t_min, inout Hit hit) {
    if (P.refs == 0u) {
        return;
    }
    RayDesc description;
    description.Origin = ray.origin;
    description.TMin = t_min;
    description.Direction = ray.direction;
    description.TMax = min(hit.t, kLargest);
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(boxes, RAY_FLAG_NONE, 0xFF, description);
    while (query.Proceed()) {
        const uint ref = candidate(query.CandidateGeometryIndex(), query.CandidatePrimitiveIndex());
        const float t = intersect(ref, ray, t_min, hit.t);
        if (t < hit.t) {
            hit.t = t;
            hit.ref = ref;
            query.CommitProceduralPrimitiveHit(t);
        }
    }
}

bool occluded(Ray ray, float t_min, float t_max) {
    if (P.refs == 0u) {
        return false;
    }
    RayDesc description;
    description.Origin = ray.origin;
    description.TMin = t_min;
    description.Direction = ray.direction;
    description.TMax = min(t_max, kLargest);
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(boxes, RAY_FLAG_NONE, 0xFF, description);
    while (query.Proceed()) {
        const uint ref = candidate(query.CandidateGeometryIndex(), query.CandidatePrimitiveIndex());
        if ((ref >> kKindShift) == kTriangle && load(A_triangle_alpha + (ref & kIndexMask)) < 1.0) {
            continue;
        }
        if (intersect(ref, ray, t_min, t_max) < t_max) {
            return true;
        }
    }
    return false;
}
#else
uint first_of(uint node) { return nodes.Load(32u * node + 12u); }
uint count_of(uint node) { return nodes.Load(32u * node + 28u); }

float enter(uint node, Ray ray, float t_min, float t_max) {
    const float3 lo = asfloat(nodes.Load3(32u * node)), hi = asfloat(nodes.Load3(32u * node + 16u));
    float t_in = t_min, t_out = t_max;
    for (int e = 0; e < 3; ++e) {
        float t0 = (lo[e] - ray.origin[e]) * ray.inverse[e];
        float t1 = (hi[e] - ray.origin[e]) * ray.inverse[e];
        if (t0 > t1) {
            const float swap = t0;
            t0 = t1;
            t1 = swap;
        }
        t_in = t0 > t_in ? t0 : t_in;
        t_out = t1 < t_out ? t1 : t_out;
    }
    return t_in <= t_out ? t_in : kInfinity;
}

void closest(Ray ray, float t_min, inout Hit hit) {
    if (P.refs == 0u) {
        return;
    }
    uint stack[96];
    int top = 0;
    stack[top++] = 0u;
    while (top > 0) {
        const uint node = stack[--top];
        const uint first = first_of(node), count = count_of(node);
        if (count > 0u) {
            for (uint k = first; k < first + count; ++k) {
                const uint ref = word(A_refs + k);
                const float t = intersect(ref, ray, t_min, hit.t);
                if (t < hit.t) {
                    hit.t = t;
                    hit.ref = ref;
                }
            }
            continue;
        }
        const float near_left = enter(first, ray, t_min, hit.t);
        const float near_right = enter(first + 1u, ray, t_min, hit.t);
        if (near_left <= near_right) {
            if (near_right < kInfinity) {
                stack[top++] = first + 1u;
            }
            if (near_left < kInfinity) {
                stack[top++] = first;
            }
        } else {
            if (near_left < kInfinity) {
                stack[top++] = first;
            }
            if (near_right < kInfinity) {
                stack[top++] = first + 1u;
            }
        }
    }
}

bool occluded(Ray ray, float t_min, float t_max) {
    if (P.refs == 0u) {
        return false;
    }
    uint stack[96];
    int top = 0;
    stack[top++] = 0u;
    while (top > 0) {
        const uint node = stack[--top];
        if (enter(node, ray, t_min, t_max) == kInfinity) {
            continue;
        }
        const uint first = first_of(node), count = count_of(node);
        if (count > 0u) {
            for (uint k = first; k < first + count; ++k) {
                const uint ref = word(A_refs + k);
                if ((ref >> kKindShift) == kTriangle && load(A_triangle_alpha + (ref & kIndexMask)) < 1.0) {
                    continue;
                }
                if (intersect(ref, ray, t_min, t_max) < t_max) {
                    return true;
                }
            }
            continue;
        }
        stack[top++] = first;
        stack[top++] = first + 1u;
    }
    return false;
}

#endif

float3 normal_at(uint ref, float3 where, float3 direction) {
    const uint i = ref & kIndexMask;
    const uint kind = ref >> kKindShift;
    float3 n;
    if (kind == kSphere) {
        n = unit(where - load3(A_sphere_center + 3u * i));
    } else if (kind == kCylinder) {
        const float3 p0 = load3(A_cylinder_end + 6u * i);
        const float3 axis = load3(A_cylinder_end + 6u * i + 3u) - p0;
        const float len = sqrt(dot(axis, axis));
        const float3 a = axis * (1.0 / len);
        const float3 rel = where - p0;
        const float along = dot(rel, a);
        const float3 radial = rel - a * along;
        const float r = load(A_cylinder_radius + i);
        // on the disk of an end, or on the side
        const bool disk = (along < 0.01 * r || along > len - 0.01 * r) && dot(radial, radial) < 0.98 * r * r;
        n = disk ? a : unit(radial);
    } else {
        const uint v = A_triangle_vertex + 9u * i;
        n = unit(cross(load3(v + 3u) - load3(v), load3(v + 6u) - load3(v)));
    }
    return dot(n, direction) > 0.0 ? n * -1.0 : n;
}

float3 color_of(uint ref) {
    const uint i = ref & kIndexMask;
    const uint kind = ref >> kKindShift;
    if (kind == kSphere) {
        return load3(A_sphere_color + 3u * i);
    }
    if (kind == kCylinder) {
        return load3(A_cylinder_color + 3u * i);
    }
    return load3(A_triangle_color + 3u * i);
}

float alpha_of(uint ref) {
    if ((ref >> kKindShift) != kTriangle) {
        return 1.0;
    }
    return load(A_triangle_alpha + (ref & kIndexMask));
}

int2 id_of(uint ref) {
    const uint i = ref & kIndexMask;
    const uint kind = ref >> kKindShift;
    const uint at = kind == kSphere ? A_sphere_id : kind == kCylinder ? A_cylinder_id : A_triangle_id;
    return int2(int(kind), int(word(at + i)));
}

uint random_from(uint seed) { return seed * 747796405u + 2891336453u; }

float next(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint value = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    value = (value >> 22u) ^ value;
    return float(value >> 8) * (1.0 / 16777216.0);
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
float3 cosine(float3 n, inout uint random, int k, int count, float turn) {
    const float3 helper = abs(n[0]) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
    const float3 tangent = unit(cross(n, helper));
    const float3 bitangent = cross(n, tangent);
    float u, angle;
    if (count > 0) {
        u = (float(k) + next(random)) / float(count);
        angle = turn + 0.618034 * float(k);
    } else {
        u = next(random);
        angle = next(random);
    }
    const float phi = 6.2831853 * angle;
    const float r = sqrt(u);
    return tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u));
}

float3 toward_light(inout uint random) {
    if (P.light_size <= 0.0) {
        return P.key;
    }
    const float z = 1.0 - next(random) * (1.0 - P.cone);
    const float phi = 6.2831853 * next(random);
    const float r = sqrt(max(0.0, 1.0 - z * z));
    return P.key_tangent * (r * cos(phi)) + P.key_bitangent * (r * sin(phi)) + P.key * z;
}

// metals: less of the color reflected diffusely, the rest in the color of
// the highlights
float body() { return 1.0 - 0.6 * saturate(P.metalness); }

float3 tint(float3 base) {
    const float m = saturate(P.metalness);
    return float3(1.0 - m, 1.0 - m, 1.0 - m) + base * m;
}

// a highlight: the light from toward mirrored about n, seen along ray
float phong(float3 n, float3 toward, float3 ray) {
    if (dot(n, toward) <= 0.0) {
        return 0.0;
    }
    const float3 mirrored = n * (2.0 * dot(n, toward)) - toward;
    return P.specular * pow(max(0.0, -dot(mirrored, ray)), P.shininess);
}

float3 shade(Hit hit, Ray ray, inout uint random) {
    const float3 where = ray.origin + ray.direction * hit.t;
    const float3 n = normal_at(hit.ref, where, ray.direction);
    const float3 base = color_of(hit.ref);
    const float3 lifted = where + n * P.epsilon;
    float ao = 1.0;
    if (P.ao_samples > 0 && P.ao_strength > 0.0) {
        const float3 helper = abs(n[0]) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
        const float3 tangent = unit(cross(n, helper));
        const float3 bitangent = cross(n, tangent);
        const float turn = next(random);
        int blocked = 0;
        for (int k = 0; k < P.ao_samples; ++k) {
            const float u = (float(k) + next(random)) / float(P.ao_samples);
            const float phi = 6.2831853 * (turn + 0.618034 * float(k));
            const float r = sqrt(u);
            const float3 d = tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u));
            blocked += occluded(make_ray(lifted, d), 0.0, P.ao_distance) ? 1 : 0;
        }
        ao = 1.0 - P.ao_strength * float(blocked) / float(P.ao_samples);
    }
    if (P.flat_shading != 0) {
        return base * ao;
    }
    const float lit_by = max(0.0, dot(n, P.key));
    float shadow = 1.0;
    if (P.shadows != 0 && lit_by > 0.0) {
        const float3 l = toward_light(random);
        if (dot(n, l) <= 0.0 || occluded(make_ray(lifted, l), 0.0, kInfinity)) {
            shadow = 1.0 - P.shadow_strength;
        }
    }
    float diffuse = lit_by * shadow + P.fill_intensity * max(0.0, dot(n, P.fill));
    if (P.toon_levels > 0) {
        const float levels = float(P.toon_levels);
        diffuse = min(1.0, floor(diffuse * levels + 0.5) / levels);
    }
    float shine = 0.0;
    if (P.specular > 0.0) {
        shine = phong(n, P.highlight, ray.direction) * shadow;
        if (P.toon_levels > 0) {
            shine = shine > 0.5 * P.specular ? P.specular : 0.0;
        }
    }
    const float light = P.base_light + P.ambient * ao + P.diffuse * diffuse * (0.5 + 0.5 * ao);
    return base * (body() * light) + tint(base) * shine;
}

float3 path(Hit first, Ray primary, inout uint random, int sample_index, float turn) {
    float3 radiance = float3(0, 0, 0), weight = float3(1, 1, 1);
    Hit hit = first;
    Ray ray = primary;
    for (int bounce = 0;; ++bounce) {
        const float3 where = ray.origin + ray.direction * hit.t;
        const float3 n = normal_at(hit.ref, where, ray.direction);
        const float3 lifted = where + n * P.epsilon;
        const float3 color = color_of(hit.ref);
        const float3 reflected = weight * color * body();
        const float3 l = toward_light(random);
        const float lit_by = dot(n, l);
        if (lit_by > 0.0 && !occluded(make_ray(lifted, l), 0.0, kInfinity)) {
            radiance = radiance + reflected * (P.diffuse * lit_by);
            if (bounce == 0 && P.specular > 0.0) {
                radiance = radiance + tint(color) * phong(n, P.highlight, ray.direction);
            }
        }
        const Ray next_ray = make_ray(lifted, cosine(n, random, sample_index, bounce == 0 ? P.samples : 0, turn));
        Hit following;
        following.t = kInfinity;
        following.ref = 0u;
        closest(next_ray, 0.0, following);
        if (following.t == kInfinity) {
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

[numthreads(8, 8, 1)]
void main(uint3 invocation : SV_DispatchThreadID) {
    P = scene.Load<Params>(0);
    const uint x = invocation.x, row = invocation.y;
    if (x >= uint(P.width) || row >= rows) {
        return;
    }
    const uint y = first_row + row;
    const uint seed = mix_seed(x, y);
    const float turn = float(mix_seed(seed, 0x51ED27u) >> 8) * (1.0 / 16777216.0);
    float3 color = float3(0, 0, 0);
    float coverage = 0.0;
    float first_depth = kInfinity;
    int2 first_id = int2(-1, 0);
    for (int sample_index = 0; sample_index < P.samples; ++sample_index) {
        uint random = random_from(sample_index == 0 ? seed : mix_seed(seed, uint(sample_index)));
        float jx = 0.5, jy = 0.5;
        if (sample_index != 0) {
            jx = next(random);
            jy = next(random);
        }
        const float px = (2.0 * (float(x) + jx) / float(P.width)) - 1.0;
        const float py = 1.0 - (2.0 * (float(y) + jy) / float(P.height));
        Ray ray;
        if (P.perspective != 0) {
            const float3 d = unit(P.forward + P.right * (px * P.half_width) + P.up * (py * P.half_height));
            ray = make_ray(P.center, d);
            if (P.aperture > 0.0) {
                const float3 sharp = P.center + d * (P.focal_distance / dot(d, P.forward));
                const float u = next(random);
                const float phi = 6.2831853 * next(random);
                const float r = P.aperture * sqrt(u);
                const float3 from = P.center + P.right * (r * cos(phi)) + P.up * (r * sin(phi));
                ray = make_ray(from, unit(sharp - from));
            }
        } else {
            ray = make_ray(P.center + P.right * (px * P.half_width) + P.up * (py * P.half_height) + P.forward * P.start,
                           P.forward);
        }
        float remaining = 1.0, t_min = 0.0;
        for (int layer = 0; layer < 8 && remaining > 1e-3; ++layer) {
            Hit hit;
            hit.t = kInfinity;
            hit.ref = 0u;
            closest(ray, t_min, hit);
            if (hit.t == kInfinity) {
                break;
            }
            if (sample_index == 0 && layer == 0) {
                first_depth = hit.t + (P.perspective != 0 ? 0.0 : P.start);
                first_id = id_of(hit.ref);
            }
            float3 c;
            if (P.bounces > 0) {
                c = path(hit, ray, random, sample_index, turn);
            } else {
                c = shade(hit, ray, random);
            }
            if (P.fog > 0.0 && P.farthest > P.nearest) {
                const float f = P.fog * saturate((hit.t + P.start - P.nearest) / (P.farthest - P.nearest));
                c = c * (1.0 - f) + P.background * f;
            }
            const float a = alpha_of(hit.ref);
            color = color + c * (remaining * a);
            remaining *= 1.0 - a;
            t_min = hit.t + P.epsilon;
        }
        coverage += 1.0 - remaining;
    }
    color = color * (1.0 / float(P.samples));
    coverage /= float(P.samples);
    const uint p = row * uint(P.width) + x;
    rgba[p] = float4(color, coverage);
    depths[p] = first_depth;
    ids[p] = first_id;
}
