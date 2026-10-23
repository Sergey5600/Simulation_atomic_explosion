#include "io/config.h"

#include <stdexcept>
#include <toml++/toml.hpp>

namespace sim {

namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("scene config: " + what); }

template <class T>
T get_or(const toml::table& t, const char* key, T fallback) {
    if (const auto* node = t.get(key)) {
        if (auto v = node->value<T>()) return *v;
        fail(std::string("key '") + key + "' has the wrong type");
    }
    return fallback;
}

template <class T>
T require(const toml::table& t, const char* key) {
    const auto* node = t.get(key);
    if (!node) fail(std::string("missing required key '") + key + "'");
    if (auto v = node->value<T>()) return *v;
    fail(std::string("key '") + key + "' has the wrong type");
}

const toml::table* subtable(const toml::table& t, const char* key) {
    const auto* node = t.get(key);
    if (!node) return nullptr;
    if (!node->is_table()) fail(std::string("'") + key + "' must be a table");
    return node->as_table();
}

std::vector<double> number_array(const toml::table& t, const char* key, std::size_t expected) {
    const auto* arr = t.get_as<toml::array>(key);
    if (!arr) fail(std::string("missing array '") + key + "'");
    if (arr->size() != expected) fail(std::string("array '") + key + "' must have " + std::to_string(expected) + " entries");
    std::vector<double> out;
    for (const auto& el : *arr) {
        if (auto v = el.value<double>()) out.push_back(*v);
        else fail(std::string("array '") + key + "' must contain numbers");
    }
    return out;
}

void parse_grid(const toml::table& t, GridInfo& g) {
    const auto res = number_array(t, "resolution", 3);
    g.dims = Int3{int(res[0]), int(res[1]), int(res[2])};
    if (g.dims.x <= 0 || g.dims.y <= 0 || g.dims.z <= 0) fail("grid.resolution must be positive");
    g.dx = float(require<double>(t, "voxel_size"));
    if (g.dx <= 0.f) fail("grid.voxel_size must be > 0");
    if (t.contains("origin")) {
        const auto o = number_array(t, "origin", 3);
        g.origin = Float3{float(o[0]), float(o[1]), float(o[2])};
    } else {
        // Centred horizontally around ground zero, resting on the ground.
        g.origin = Float3{-0.5f * g.dims.x * g.dx, -0.5f * g.dims.y * g.dx, 0.f};
    }
}

void parse_schedule(const toml::table& out, ExportSchedule& s) {
    s.start_time = get_or<double>(out, "start_time", 0.0);
    const auto* arr = out.get_as<toml::array>("schedule");
    if (!arr || arr->empty()) fail("output.schedule must contain at least one [[output.schedule]] segment");
    for (const auto& node : *arr) {
        const auto* seg_t = node.as_table();
        if (!seg_t) fail("output.schedule entries must be tables");
        ScheduleSegment seg;
        const std::string mode = get_or<std::string>(*seg_t, "mode", "constant");
        seg.frames = int(require<std::int64_t>(*seg_t, "frames"));
        if (mode == "constant") {
            seg.mode = ScheduleSegment::Mode::Constant;
            seg.dt_start = seg.dt_end = require<double>(*seg_t, "dt");
        } else if (mode == "ramp") {
            seg.mode = ScheduleSegment::Mode::Ramp;
            seg.dt_start = require<double>(*seg_t, "dt_start");
            seg.dt_end = require<double>(*seg_t, "dt_end");
        } else {
            fail("unknown schedule mode '" + mode + "' (expected constant or ramp)");
        }
        if (seg.frames <= 0 || seg.dt_start <= 0.0 || seg.dt_end <= 0.0) fail("schedule segment needs frames > 0 and dt > 0");
        s.segments.push_back(seg);
    }
}

void parse_output(const toml::table& t, OutputConfig& o) {
    o.dir = get_or<std::string>(t, "dir", o.dir);
    o.prefix = get_or<std::string>(t, "prefix", o.prefix);
    o.fps = get_or<double>(t, "fps", o.fps);
    o.world_scale = float(get_or<double>(t, "world_scale", o.world_scale));
    o.half_float = get_or<bool>(t, "half_float", o.half_float);
    o.prune_tolerance = float(get_or<double>(t, "prune_tolerance", o.prune_tolerance));
    o.preview_every = int(get_or<std::int64_t>(t, "preview_every", o.preview_every));
    o.slices_every = int(get_or<std::int64_t>(t, "slices_every", o.slices_every));
    if (const auto* arr = t.get_as<toml::array>("fields")) {
        for (const auto& el : *arr) {
            if (auto v = el.value<std::string>()) o.fields.push_back(*v);
            else fail("output.fields must contain strings");
        }
    } else {
        o.fields = {"density", "temperature", "flame", "velocity"};
    }
    if (o.fps <= 0.0) fail("output.fps must be > 0");
    if (o.world_scale <= 0.f) fail("output.world_scale must be > 0");
    parse_schedule(t, o.schedule);
}

void parse_test_fireball(const toml::table& t, TestFireballConfig& c) {
    c.max_radius = float(get_or<double>(t, "max_radius", c.max_radius));
    c.formation_time = float(get_or<double>(t, "formation_time", c.formation_time));
    c.rise_speed = float(get_or<double>(t, "rise_speed", c.rise_speed));
    c.peak_temperature = float(get_or<double>(t, "peak_temperature", c.peak_temperature));
    c.cooling_time = float(get_or<double>(t, "cooling_time", c.cooling_time));
    c.noise_frequency = float(get_or<double>(t, "noise_frequency", c.noise_frequency));
    c.noise_amplitude = float(get_or<double>(t, "noise_amplitude", c.noise_amplitude));
}

void parse_atmosphere(const toml::table& t, AtmosphereConfig& c) {
    c.ground_altitude = float(get_or<double>(t, "ground_altitude", c.ground_altitude));
    c.ground_temperature = float(get_or<double>(t, "ground_temperature", c.ground_temperature));
    c.lapse_rate = float(get_or<double>(t, "lapse_rate", c.lapse_rate));
    c.tropopause_altitude = float(get_or<double>(t, "tropopause_altitude", c.tropopause_altitude));
    c.sea_level_pressure = float(get_or<double>(t, "sea_level_pressure", c.sea_level_pressure));
}

void parse_wind(const toml::table& t, WindProfile& w) {
    w.heading_deg = float(get_or<double>(t, "heading", w.heading_deg));
    if (const auto* arr = t.get_as<toml::array>("profile")) {
        for (const auto& el : *arr) {
            const auto* pair = el.as_array();
            if (!pair || pair->size() != 2) fail("wind.profile entries must be [height, speed] pairs");
            const auto z = (*pair)[0].value<double>(), s = (*pair)[1].value<double>();
            if (!z || !s) fail("wind.profile entries must be numbers");
            if (!w.points.empty() && *z <= w.points.back().first) fail("wind.profile heights must increase");
            w.points.emplace_back(float(*z), float(*s));
        }
    }
}

void parse_fireball(const toml::table& t, FireballConfig& c) {
    c.start_time = float(get_or<double>(t, "start_time", c.start_time));
    c.radius = float(get_or<double>(t, "radius", c.radius));
    c.center_height = float(get_or<double>(t, "center_height", c.center_height));
    c.vertical_scale = float(get_or<double>(t, "vertical_scale", c.vertical_scale));
    c.temperature = float(get_or<double>(t, "temperature", c.temperature));
    c.edge_width = float(get_or<double>(t, "edge_width", c.edge_width));
    c.noise_amplitude = float(get_or<double>(t, "noise_amplitude", c.noise_amplitude));
    c.noise_frequency = float(get_or<double>(t, "noise_frequency", c.noise_frequency));
    c.flame_temperature = float(get_or<double>(t, "flame_temperature", c.flame_temperature));
    if (c.radius <= 0.f || c.temperature <= 0.f || c.edge_width <= 0.f || c.vertical_scale <= 0.f) {
        fail("fireball: radius, temperature, edge_width and vertical_scale must be > 0");
    }
}

void parse_radiation(const toml::table& t, RadiationParams& c) {
    c.absorption = float(get_or<double>(t, "absorption", c.absorption));
    c.reference_temperature = float(get_or<double>(t, "reference_temperature", c.reference_temperature));
    c.exponent = float(get_or<double>(t, "exponent", c.exponent));
}

void bins(const toml::table& t, const char* key, float (&out)[kDustBins]) {
    if (!t.contains(key)) return;
    const auto v = number_array(t, key, kDustBins);
    for (int b = 0; b < kDustBins; ++b) out[b] = float(v[std::size_t(b)]);
}

void parse_dust(const toml::table& t, DustParams& d, DustInitConfig& init) {
    d.enabled = get_or<bool>(t, "enabled", d.enabled);
    bins(t, "settling", d.settling);
    bins(t, "optical_weight", d.optical_weight);
    d.erosion = float(get_or<double>(t, "erosion", d.erosion));
    d.threshold_friction_velocity = float(get_or<double>(t, "threshold_friction_velocity", d.threshold_friction_velocity));
    d.roughness_length = float(get_or<double>(t, "roughness_length", d.roughness_length));
    bins(t, "erosion_split", d.erosion_split);
    d.mass_loading = get_or<bool>(t, "mass_loading", d.mass_loading);
    d.render_scale = float(get_or<double>(t, "render_scale", d.render_scale));
    init.fireball_loading = float(get_or<double>(t, "fireball_loading", init.fireball_loading));
    bins(t, "fireball_split", init.fireball_split);
    init.skirt_radius = float(get_or<double>(t, "skirt_radius", init.skirt_radius));
    init.skirt_height = float(get_or<double>(t, "skirt_height", init.skirt_height));
    init.skirt_loading = float(get_or<double>(t, "skirt_loading", init.skirt_loading));
    bins(t, "skirt_split", init.skirt_split);
    init.noise_frequency = float(get_or<double>(t, "noise_frequency", init.noise_frequency));
    if (d.roughness_length <= 0.f) fail("dust.roughness_length must be > 0");
}

void parse_domain(const toml::table& t, DomainConfig& d) {
    d.expand = get_or<bool>(t, "expand", d.expand);
    d.trigger = float(get_or<double>(t, "trigger", d.trigger));
    d.max_expansions = int(get_or<std::int64_t>(t, "max_expansions", d.max_expansions));
    d.smoke_threshold = float(get_or<double>(t, "smoke_threshold", d.smoke_threshold));
    if (d.trigger <= 0.f || d.trigger > 1.f) fail("domain.trigger must be in (0, 1]");
}

void parse_blast(const toml::table& t, BlastParams& b) {
    b.start_time = float(get_or<double>(t, "start_time", b.start_time));
    b.yield_kt = float(get_or<double>(t, "yield_kt", b.yield_kt));
    b.hydro_fraction = float(get_or<double>(t, "hydro_fraction", b.hydro_fraction));
    b.deposit_radius = float(get_or<double>(t, "deposit_radius", b.deposit_radius));
    b.noise_amplitude = float(get_or<double>(t, "noise_amplitude", b.noise_amplitude));
    b.noise_frequency = float(get_or<double>(t, "noise_frequency", b.noise_frequency));
    b.handoff_time = float(get_or<double>(t, "handoff_time", b.handoff_time));
    b.gamma = float(get_or<double>(t, "gamma", b.gamma));
    b.cfl = float(get_or<double>(t, "cfl", b.cfl));
    b.radiation_temperature_cap = float(get_or<double>(t, "radiation_temperature_cap", b.radiation_temperature_cap));
    b.radiation_length = float(get_or<double>(t, "radiation_length", b.radiation_length));
    b.fireball_temperature = float(get_or<double>(t, "fireball_temperature", b.fireball_temperature));
    b.fireball_dust_loading = float(get_or<double>(t, "fireball_dust_loading", b.fireball_dust_loading));
    bins(t, "fireball_dust_split", b.fireball_dust_split);
    b.blast_erosion = float(get_or<double>(t, "erosion", b.blast_erosion));
    if (b.yield_kt <= 0.f || b.deposit_radius <= 0.f || b.handoff_time <= b.start_time) {
        fail("blast: yield_kt and deposit_radius must be > 0 and handoff_time > start_time");
    }
}

void parse_humidity(const toml::table& t, HumidityProfile& h) {
    h.enabled = get_or<bool>(t, "enabled", true);
    h.render_scale = float(get_or<double>(t, "render_scale", h.render_scale));
    if (const auto* arr = t.get_as<toml::array>("profile")) {
        for (const auto& el : *arr) {
            const auto* pair = el.as_array();
            if (!pair || pair->size() != 2) fail("humidity.profile entries must be [height, relative humidity] pairs");
            const auto z = (*pair)[0].value<double>(), rh = (*pair)[1].value<double>();
            if (!z || !rh) fail("humidity.profile entries must be numbers");
            if (*rh < 0.0 || *rh > 1.0) fail("humidity.profile relative humidity must be between 0 and 1");
            if (!h.points.empty() && *z <= h.points.back().first) fail("humidity.profile heights must increase");
            h.points.emplace_back(float(*z), float(*rh));
        }
    }
}

void parse_solver(const toml::table& t, SolverParams& c) {
    c.cfl = float(get_or<double>(t, "cfl", c.cfl));
    c.max_dt = float(get_or<double>(t, "max_dt", c.max_dt));
    c.reflection = get_or<bool>(t, "reflection", c.reflection);
    c.vorticity_confinement = float(get_or<double>(t, "vorticity_confinement", c.vorticity_confinement));
    c.smagorinsky = float(get_or<double>(t, "smagorinsky", c.smagorinsky));
    c.turbulent_prandtl = float(get_or<double>(t, "turbulent_prandtl", c.turbulent_prandtl));
    c.pressure_tolerance = float(get_or<double>(t, "pressure_tolerance", c.pressure_tolerance));
    c.pressure_max_iterations = int(get_or<std::int64_t>(t, "pressure_max_iterations", c.pressure_max_iterations));
    if (c.cfl <= 0.f || c.max_dt <= 0.f) fail("solver: cfl and max_dt must be > 0");
}

}  // namespace

SceneConfig load_scene(const std::string& path) {
    toml::table root;
    try {
        root = toml::parse_file(path);
    } catch (const toml::parse_error& e) {
        const auto& src = e.source();
        fail(path + ":" + std::to_string(src.begin.line) + ":" + std::to_string(src.begin.column) + ": " +
             std::string(e.description()));
    }

    SceneConfig cfg;
    cfg.source_path = path;
    if (const auto* scene = subtable(root, "scene")) {
        cfg.name = get_or<std::string>(*scene, "name", cfg.name);
        cfg.type = get_or<std::string>(*scene, "type", cfg.type);
        cfg.device = get_or<std::string>(*scene, "device", cfg.device);
        cfg.seed = std::uint32_t(get_or<std::int64_t>(*scene, "seed", cfg.seed));
    }
    const auto* grid = subtable(root, "grid");
    if (!grid) fail("missing [grid] table");
    parse_grid(*grid, cfg.grid);

    const auto* output = subtable(root, "output");
    if (!output) fail("missing [output] table");
    parse_output(*output, cfg.output);

    if (const auto* tf = subtable(root, "test_fireball")) parse_test_fireball(*tf, cfg.test_fireball);
    if (const auto* t = subtable(root, "atmosphere")) parse_atmosphere(*t, cfg.atmosphere);
    if (const auto* t = subtable(root, "wind")) parse_wind(*t, cfg.wind);
    if (const auto* t = subtable(root, "fireball")) parse_fireball(*t, cfg.fireball);
    if (const auto* t = subtable(root, "radiation")) parse_radiation(*t, cfg.radiation);
    if (const auto* t = subtable(root, "solver")) parse_solver(*t, cfg.solver);
    if (const auto* t = subtable(root, "dust")) parse_dust(*t, cfg.dust, cfg.dust_init);
    if (const auto* t = subtable(root, "domain")) parse_domain(*t, cfg.domain);
    if (const auto* t = subtable(root, "blast")) parse_blast(*t, cfg.blast);
    if (const auto* t = subtable(root, "humidity")) parse_humidity(*t, cfg.humidity);
    return cfg;
}

}  // namespace sim
