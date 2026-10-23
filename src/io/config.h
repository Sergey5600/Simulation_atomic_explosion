#pragma once
// Scene configuration loaded from a TOML file (see scenes/*.toml).

#include <cstdint>
#include <string>
#include <vector>

#include "core/grid.h"
#include "core/schedule.h"

namespace sim {

struct OutputConfig {
    std::string dir = "out";
    std::string prefix = "explosion";
    double fps = 24.0;
    float world_scale = 1.f;          // metres -> exported units
    bool half_float = true;           // store float grids as 16-bit in the .vdb
    float prune_tolerance = 1e-4f;    // values with |v| <= tolerance become background
    std::vector<std::string> fields;  // exported grids
    int preview_every = 0;            // write a PNG preview every N frames (0 = off)
    ExportSchedule schedule;
};

// Parameters of the analytic M0 pipeline-test fireball (not a simulation).
struct TestFireballConfig {
    float max_radius = 150.f;        // m
    float formation_time = 1.f;      // s, time to reach max_radius
    float rise_speed = 40.f;         // m/s, buoyant rise after formation
    float peak_temperature = 8000.f; // K
    float cooling_time = 1.5f;       // s, e-folding time of the temperature
    float noise_frequency = 0.02f;   // 1/m
    float noise_amplitude = 0.25f;   // relative boundary displacement
};

struct SceneConfig {
    std::string source_path;
    std::string name = "scene";
    std::string type = "test_fireball";
    std::string device = "auto";
    std::uint32_t seed = 1;
    GridInfo grid;
    OutputConfig output;
    TestFireballConfig test_fireball;
};

SceneConfig load_scene(const std::string& path);

}  // namespace sim
