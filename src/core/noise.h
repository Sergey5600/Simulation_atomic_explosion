#pragma once
// Table-free gradient noise usable in host and device code.

#include <cstdint>

#include "core/math.h"

namespace sim::noise {

SIM_INLINE std::uint32_t hash_u32(std::uint32_t x) {
    // lowbias32 by Chris Wellons
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

SIM_INLINE std::uint32_t hash3(int x, int y, int z, std::uint32_t seed) {
    return hash_u32(std::uint32_t(x) * 0x8da6b343U ^ hash_u32(std::uint32_t(y) * 0xd8163841U ^
                    hash_u32(std::uint32_t(z) * 0xcb1ab31fU ^ seed)));
}

// One of 12 edge gradients of a cube, dotted with (x, y, z).
SIM_INLINE float grad_dot(std::uint32_t h, float x, float y, float z) {
    switch (h % 12u) {
        case 0: return x + y;
        case 1: return -x + y;
        case 2: return x - y;
        case 3: return -x - y;
        case 4: return x + z;
        case 5: return -x + z;
        case 6: return x - z;
        case 7: return -x - z;
        case 8: return y + z;
        case 9: return -y + z;
        case 10: return y - z;
        default: return -y - z;
    }
}

SIM_INLINE float fade(float t) { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); }

// Perlin-style gradient noise, roughly in [-1, 1].
SIM_INLINE float perlin(Float3 p, std::uint32_t seed) {
    const float fx = floorf(p.x), fy = floorf(p.y), fz = floorf(p.z);
    const int ix = int(fx), iy = int(fy), iz = int(fz);
    const float x = p.x - fx, y = p.y - fy, z = p.z - fz;
    const float u = fade(x), v = fade(y), w = fade(z);

    const float n000 = grad_dot(hash3(ix, iy, iz, seed), x, y, z);
    const float n100 = grad_dot(hash3(ix + 1, iy, iz, seed), x - 1.f, y, z);
    const float n010 = grad_dot(hash3(ix, iy + 1, iz, seed), x, y - 1.f, z);
    const float n110 = grad_dot(hash3(ix + 1, iy + 1, iz, seed), x - 1.f, y - 1.f, z);
    const float n001 = grad_dot(hash3(ix, iy, iz + 1, seed), x, y, z - 1.f);
    const float n101 = grad_dot(hash3(ix + 1, iy, iz + 1, seed), x - 1.f, y, z - 1.f);
    const float n011 = grad_dot(hash3(ix, iy + 1, iz + 1, seed), x, y - 1.f, z - 1.f);
    const float n111 = grad_dot(hash3(ix + 1, iy + 1, iz + 1, seed), x - 1.f, y - 1.f, z - 1.f);

    const float nx00 = lerpf(n000, n100, u), nx10 = lerpf(n010, n110, u);
    const float nx01 = lerpf(n001, n101, u), nx11 = lerpf(n011, n111, u);
    return lerpf(lerpf(nx00, nx10, v), lerpf(nx01, nx11, v), w);
}

// Fractal sum of octaves; result roughly in [-1, 1].
SIM_INLINE float fbm(Float3 p, int octaves, float lacunarity, float gain, std::uint32_t seed) {
    float sum = 0.f, amp = 0.5f, norm = 0.f;
    for (int o = 0; o < octaves; ++o) {
        sum += amp * perlin(p, seed + std::uint32_t(o) * 1013u);
        norm += amp;
        p = p * lacunarity;
        amp *= gain;
    }
    return norm > 0.f ? sum / norm : 0.f;
}

}  // namespace sim::noise
