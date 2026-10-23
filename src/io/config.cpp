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
    return cfg;
}

}  // namespace sim
