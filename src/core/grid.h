#pragma once
// Uniform cell-centred grids and scalar fields.
//
// World space: metres, Z up (matches Blender), ground plane at z = 0.
// Cell (i, j, k) covers [origin + (i, j, k) * dx, origin + (i+1, j+1, k+1) * dx].
// Memory layout: x fastest, then y, then z.

#include <cstdint>
#include <string>

#include "core/math.h"
#include "core/memory.h"

namespace sim {

struct GridInfo {
    Int3 dims;
    float dx = 1.f;   // voxel edge length, metres
    Float3 origin;    // world position of the min corner of cell (0, 0, 0)

    SIM_INLINE std::int64_t count() const { return dims.volume(); }
    SIM_INLINE std::int64_t index(int i, int j, int k) const {
        return i + std::int64_t(dims.x) * (j + std::int64_t(dims.y) * k);
    }
    SIM_INLINE Float3 cell_center(int i, int j, int k) const {
        return {origin.x + (i + 0.5f) * dx, origin.y + (j + 0.5f) * dx, origin.z + (k + 0.5f) * dx};
    }
    SIM_INLINE Float3 extent() const { return {dims.x * dx, dims.y * dx, dims.z * dx}; }
};

// Non-owning view passed by value into kernels.
struct FieldView {
    float* data = nullptr;
    Int3 dims;

    SIM_INLINE float& operator()(int i, int j, int k) const {
        return data[i + std::int64_t(dims.x) * (j + std::int64_t(dims.y) * k)];
    }
};

class ScalarField {
public:
    ScalarField() = default;
    ScalarField(std::string name, const GridInfo& grid, Device dev)
        : name_(std::move(name)), grid_(grid), buf_(dev, std::size_t(grid.count())) {}

    FieldView view() { return FieldView{buf_.data(), grid_.dims}; }
    const GridInfo& grid() const { return grid_; }
    const std::string& name() const { return name_; }
    Device device() const { return buf_.device(); }
    Buffer<float>& buffer() { return buf_; }
    const Buffer<float>& buffer() const { return buf_; }

private:
    std::string name_;
    GridInfo grid_;
    Buffer<float> buf_;
};

}  // namespace sim
