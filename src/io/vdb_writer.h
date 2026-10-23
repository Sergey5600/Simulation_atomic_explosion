#pragma once
// Writes host-side dense fields into sparse OpenVDB files readable by Blender.

#include <map>
#include <string>
#include <vector>

#include "core/grid.h"

namespace sim {

struct VdbScalarInput {
    std::string name;
    const float* data = nullptr;  // dense, GridInfo layout
};

struct VdbVectorInput {
    std::string name;
    const float* x = nullptr;
    const float* y = nullptr;
    const float* z = nullptr;
    float scale = 1.f;  // multiplier applied on export (e.g. unit conversion)
};

struct VdbFrame {
    GridInfo grid;
    std::vector<VdbScalarInput> scalars;
    std::vector<VdbVectorInput> vectors;  // stored only where any scalar is active
    std::map<std::string, float> metadata;
};

struct VdbWriteOptions {
    float world_scale = 1.f;
    bool half_float = true;
    float tolerance = 1e-4f;
};

struct VdbWriteStats {
    std::size_t active_voxels = 0;  // union over scalar grids
    std::size_t file_bytes = 0;
};

VdbWriteStats write_vdb(const std::string& path, const VdbFrame& frame, const VdbWriteOptions& opt);

}  // namespace sim
