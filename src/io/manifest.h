#pragma once
// manifest.json next to the exported frames: per-frame timing and lighting
// data consumed by tools/blender/import_explosion.py.

#include <string>
#include <vector>

#include "core/math.h"

namespace sim {

struct ManifestFrame {
    int number = 0;            // 1-based, matches the file name
    std::string file;
    double sim_time = 0.0;     // s
    double sim_dt = 0.0;       // sim seconds covered by this frame (for slow-motion aware motion blur)
    float voxel_size = 0.f;    // in exported units
    Float3 domain_min;         // simulation domain bounds, exported units
    Float3 domain_max;
    Float3 cloud_min;          // bounds of the visible volume, exported units
    Float3 cloud_max;
    Float3 light_position;     // emission-weighted centre, exported units
    double light_power = 0.0;  // relative radiated power (arbitrary units)
    std::size_t active_voxels = 0;
    std::size_t file_bytes = 0;
    float rest_weight = -1.f;    // cross-fade weight of grid rest_a (rest_b gets 1 - w); < 0 without rest grids
    float vorticity_ref = 0.f;   // 1/s, RMS vorticity of the visible volume (normalises the detail amount)
};

struct Manifest {
    std::string scene_name;
    std::string prefix;
    double fps = 24.0;
    float world_scale = 1.f;
    std::vector<std::string> fields;
    std::vector<ManifestFrame> frames;
};

void write_manifest(const std::string& path, const Manifest& m);

}  // namespace sim
