#pragma once
// Scene configuration loaded from a TOML file (see scenes/*.toml).

#include <cstdint>
#include <string>
#include <vector>

#include "core/grid.h"
#include "core/schedule.h"
#include "lowmach/solver.h"
#include "physics/atmosphere.h"
#include "physics/wind.h"

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

// Initial state of the buoyant phase (scene type "fireball_rise"): the fireball once
// the shock has detached and its pressure is back to ambient.
struct FireballConfig {
    float start_time = 1.f;            // s after detonation
    float radius = 200.f;              // m
    float center_height = 0.f;         // m; 0 = surface burst (hemisphere on the ground)
    float vertical_scale = 1.f;        // < 1 flattens the fireball
    float temperature = 4000.f;        // K in the core
    float edge_width = 0.25f;          // fraction of the radius blending into ambient air
    float noise_amplitude = 0.12f;     // relative boundary displacement (seeds the instabilities)
    float noise_frequency = 0.012f;    // 1/m
    float flame_temperature = 6000.f;  // K at which the exported flame field is 1
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
    AtmosphereConfig atmosphere;
    WindProfile wind;
    FireballConfig fireball;
    RadiationParams radiation;
    SolverParams solver;
};

SceneConfig load_scene(const std::string& path);

}  // namespace sim
