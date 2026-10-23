#pragma once
// Scene configuration loaded from a TOML file (see scenes/*.toml).

#include <cstdint>
#include <string>
#include <vector>

#include "core/grid.h"
#include "compressible/euler.h"
#include "core/schedule.h"
#include "lowmach/solver.h"
#include "upres/rest_fields.h"
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
    int slices_every = 0;             // write density/vorticity slice PNGs every N frames (0 = off)
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

// Dust present when the buoyant phase starts (1 s): soil engulfed by the fireball and
// the skirt raised by the blast wave along the ground.
struct DustInitConfig {
    float fireball_loading = 0.02f;  // kg dust / kg air inside the fireball
    float fireball_split[kDustBins] = {0.5f, 0.3f, 0.2f};
    float skirt_radius = 700.f;      // m, reach of the dust raised by the shock
    float skirt_height = 120.f;      // m at ground zero, thinning outwards
    float skirt_loading = 0.01f;     // kg/kg near the ground
    float skirt_split[kDustBins] = {0.4f, 0.45f, 0.15f};
    float noise_frequency = 0.008f;  // 1/m
};

// Growing domain: when the cloud reaches `trigger` of the domain height or half-width,
// the voxel size and the domain double (fine detail early, the whole cloud later).
struct DomainConfig {
    bool expand = true;
    float trigger = 0.8f;
    int max_expansions = 4;
    float smoke_threshold = 0.02f;  // smoke fraction that counts as cloud
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
    DustParams dust;
    DustInitConfig dust_init;
    DomainConfig domain;
    BlastParams blast;
    HumidityProfile humidity;
    RestParams detail;     // rest fields for render-time detail, computed when output.fields has "rest"
    bool profile = false;  // per-stage step timing in the log (set from the command line)
};

SceneConfig load_scene(const std::string& path);

}  // namespace sim
