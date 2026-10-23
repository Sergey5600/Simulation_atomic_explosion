#pragma once
// Small vector types usable in host and device code.

#include <cmath>
#include <cstdint>

#include "core/backend.h"

namespace sim {

struct Int3 {
    int x = 0, y = 0, z = 0;
    SIM_INLINE std::int64_t volume() const { return std::int64_t(x) * y * z; }
    SIM_INLINE bool operator==(const Int3& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct Float3 {
    float x = 0.f, y = 0.f, z = 0.f;
};

SIM_INLINE Float3 make_f3(float x, float y, float z) { return Float3{x, y, z}; }
SIM_INLINE Float3 operator+(Float3 a, Float3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
SIM_INLINE Float3 operator-(Float3 a, Float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
SIM_INLINE Float3 operator*(Float3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
SIM_INLINE Float3 operator*(float s, Float3 a) { return {a.x * s, a.y * s, a.z * s}; }
SIM_INLINE Float3 operator/(Float3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
SIM_INLINE float dot(Float3 a, Float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
SIM_INLINE float length(Float3 a) { return sqrtf(dot(a, a)); }

SIM_INLINE float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
SIM_INLINE float lerpf(float a, float b, float t) { return a + (b - a) * t; }
SIM_INLINE float smoothstepf(float e0, float e1, float x) {
    float t = clampf((x - e0) / (e1 - e0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

}  // namespace sim
