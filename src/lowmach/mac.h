#pragma once
// Staggered (MAC) velocity grid and interpolation helpers.
//
// "Grid units": a position g corresponds to world point origin + g * dx, so
// cell (i, j, k) has its centre at (i + 0.5, j + 0.5, k + 0.5),
// u(i, j, k) sits on the x-face at (i, j + 0.5, k + 0.5)   dims (nx+1, ny, nz),
// v(i, j, k) sits on the y-face at (i + 0.5, j, k + 0.5)   dims (nx, ny+1, nz),
// w(i, j, k) sits on the z-face at (i + 0.5, j + 0.5, k)   dims (nx, ny, nz+1).
// w(i, j, 0) is the ground and stays zero.

#include <algorithm>
#include <cstdint>

#include "core/grid.h"

namespace sim {

SIM_INLINE Int3 u_dims(Int3 n) { return {n.x + 1, n.y, n.z}; }
SIM_INLINE Int3 v_dims(Int3 n) { return {n.x, n.y + 1, n.z}; }
SIM_INLINE Int3 w_dims(Int3 n) { return {n.x, n.y, n.z + 1}; }

inline std::size_t max_face_count(Int3 n) {
    return std::size_t(std::max({u_dims(n).volume(), v_dims(n).volume(), w_dims(n).volume()}));
}

// Read-only view of a staggered scalar array for interpolation.
struct SampleView {
    const float* data = nullptr;
    Int3 dims;
    Float3 offset;  // grid-unit position of sample (0, 0, 0)

    SIM_INLINE float at(int i, int j, int k) const {
        i = i < 0 ? 0 : (i >= dims.x ? dims.x - 1 : i);
        j = j < 0 ? 0 : (j >= dims.y ? dims.y - 1 : j);
        k = k < 0 ? 0 : (k >= dims.z ? dims.z - 1 : k);
        return data[i + std::int64_t(dims.x) * (j + std::int64_t(dims.y) * k)];
    }

    // Trilinear interpolation at grid-unit position g, clamped to the array.
    SIM_INLINE float sample(Float3 g) const {
        const float x = g.x - offset.x, y = g.y - offset.y, z = g.z - offset.z;
        const float fx = floorf(x), fy = floorf(y), fz = floorf(z);
        const int i = int(fx), j = int(fy), k = int(fz);
        const float tx = x - fx, ty = y - fy, tz = z - fz;
        const float c00 = lerpf(at(i, j, k), at(i + 1, j, k), tx);
        const float c10 = lerpf(at(i, j + 1, k), at(i + 1, j + 1, k), tx);
        const float c01 = lerpf(at(i, j, k + 1), at(i + 1, j, k + 1), tx);
        const float c11 = lerpf(at(i, j + 1, k + 1), at(i + 1, j + 1, k + 1), tx);
        return lerpf(lerpf(c00, c10, ty), lerpf(c01, c11, ty), tz);
    }

    // Min and max of the 8 samples surrounding g (MacCormack limiter).
    SIM_INLINE void minmax(Float3 g, float& lo, float& hi) const {
        const int i = int(floorf(g.x - offset.x)), j = int(floorf(g.y - offset.y)), k = int(floorf(g.z - offset.z));
        lo = hi = at(i, j, k);
        for (int c = 1; c < 8; ++c) {
            const float v = at(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
            lo = fminf(lo, v);
            hi = fmaxf(hi, v);
        }
    }
};

constexpr Float3 kCellOffset{0.5f, 0.5f, 0.5f};
constexpr Float3 kUOffset{0.f, 0.5f, 0.5f};
constexpr Float3 kVOffset{0.5f, 0.f, 0.5f};
constexpr Float3 kWOffset{0.5f, 0.5f, 0.f};

struct MacView {
    SampleView u, v, w;

    SIM_INLINE Float3 sample(Float3 g) const { return {u.sample(g), v.sample(g), w.sample(g)}; }

    // Second-order (midpoint) trace of a point over dt; dt_over_dx converts m/s to grid units.
    SIM_INLINE Float3 trace(Float3 g, float dt_over_dx) const {
        const Float3 v1 = sample(g);
        const Float3 mid = g - v1 * (0.5f * dt_over_dx);
        const Float3 v2 = sample(mid);
        return g - v2 * dt_over_dx;
    }
};

// Owning staggered velocity. Every component is allocated with max_face_count so that
// component buffers and advection scratch buffers can be swapped freely.
struct MacField {
    Int3 dims;  // cell counts
    Buffer<float> u, v, w;

    MacField() = default;
    MacField(Int3 n, Device dev) : dims(n), u(dev, max_face_count(n)), v(dev, max_face_count(n)), w(dev, max_face_count(n)) {}
    std::size_t memory_bytes() const { return u.bytes() + v.bytes() + w.bytes(); }

    FieldView u_view() { return {u.data(), u_dims(dims)}; }
    FieldView v_view() { return {v.data(), v_dims(dims)}; }
    FieldView w_view() { return {w.data(), w_dims(dims)}; }
    MacView sampler() const {
        return {{u.data(), u_dims(dims), kUOffset}, {v.data(), v_dims(dims), kVOffset}, {w.data(), w_dims(dims), kWOffset}};
    }
    void swap(MacField& o) noexcept {
        std::swap(dims, o.dims);
        std::swap(u, o.u);
        std::swap(v, o.v);
        std::swap(w, o.w);
    }
};

}  // namespace sim
