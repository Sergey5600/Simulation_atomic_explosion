#include "app/run.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "core/backend.h"
#include "core/grid.h"
#include "core/log.h"
#include "io/config.h"
#include "io/manifest.h"
#include "io/vdb_writer.h"
#include "preview/preview.h"
#include "scenes/scene.h"

namespace sim {

namespace {

bool wants(const std::vector<std::string>& fields, const char* name) {
    return std::find(fields.begin(), fields.end(), name) != fields.end();
}

void validate_fields(const std::vector<std::string>& fields) {
    static const char* known[] = {"density", "dust", "condensate", "temperature", "flame", "velocity", "vorticity"};
    for (const std::string& f : fields) {
        if (std::none_of(std::begin(known), std::end(known), [&](const char* k) { return f == k; })) {
            throw std::runtime_error("unknown output field '" + f + "'");
        }
    }
}

std::string frame_file_name(const std::string& prefix, int number) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "_%04d.vdb", number);
    return prefix + buf;
}

// Emission-weighted centre and total emitted power (arbitrary units), for the Blender light rig.
void measure_light(const GridInfo& g, const std::vector<float>& flame, Float3& center, double& power) {
    double sum = 0.0, cx = 0.0, cy = 0.0, cz = 0.0;
    for (int k = 0; k < g.dims.z; ++k)
        for (int j = 0; j < g.dims.y; ++j)
            for (int i = 0; i < g.dims.x; ++i) {
                const double f = flame[std::size_t(g.index(i, j, k))];
                if (f <= 0.0) continue;
                const Float3 p = g.cell_center(i, j, k);
                sum += f;
                cx += f * p.x;
                cy += f * p.y;
                cz += f * p.z;
            }
    power = sum * double(g.dx) * g.dx * g.dx;
    center = sum > 0.0 ? Float3{float(cx / sum), float(cy / sum), float(cz / sum)} : Float3{};
}

// Bounds of the cells that would be visible in a render.
void measure_cloud(const GridInfo& g, const std::vector<float>& density, const std::vector<float>& dust,
                   const std::vector<float>& flame, Float3& lo, Float3& hi) {
    int i0 = g.dims.x, j0 = g.dims.y, k0 = g.dims.z, i1 = -1, j1 = -1, k1 = -1;
    for (int k = 0; k < g.dims.z; ++k)
        for (int j = 0; j < g.dims.y; ++j)
            for (int i = 0; i < g.dims.x; ++i) {
                const std::size_t idx = std::size_t(g.index(i, j, k));
                if (density[idx] <= 1e-3f && dust[idx] <= 1e-3f && flame[idx] <= 1e-4f) continue;
                i0 = std::min(i0, i), j0 = std::min(j0, j), k0 = std::min(k0, k);
                i1 = std::max(i1, i), j1 = std::max(j1, j), k1 = std::max(k1, k);
            }
    if (i1 < 0) {
        lo = hi = Float3{};
        return;
    }
    lo = g.origin + Float3{float(i0), float(j0), float(k0)} * g.dx;
    hi = g.origin + Float3{float(i1 + 1), float(j1 + 1), float(k1 + 1)} * g.dx;
}

}  // namespace

int run_scene(const RunOptions& opt) {
    SceneConfig cfg = load_scene(opt.scene_path);
    if (!opt.device.empty()) cfg.device = opt.device;
    if (!opt.output_dir.empty()) cfg.output.dir = opt.output_dir;
    if (opt.preview_every >= 0) cfg.output.preview_every = opt.preview_every;
    if (opt.slices_every >= 0) cfg.output.slices_every = opt.slices_every;
    cfg.profile = opt.profile;
    validate_fields(cfg.output.fields);

    const Device dev = parse_device(cfg.device);
    const GridInfo& g = cfg.grid;
    log_info("scene '%s' (%s) from %s", cfg.name.c_str(), cfg.type.c_str(), cfg.source_path.c_str());
    log_info("device: %s", device_description(dev).c_str());
    log_info("grid: %d x %d x %d cells, voxel %.3g m, domain %.0f x %.0f x %.0f m", g.dims.x, g.dims.y, g.dims.z,
             g.dx, g.extent().x, g.extent().y, g.extent().z);

    std::vector<double> times = cfg.output.schedule.frame_times();
    log_info("schedule: %s", describe(cfg.output.schedule, cfg.output.fps).c_str());
    if (opt.max_frames >= 0 && std::size_t(opt.max_frames) < times.size()) times.resize(std::size_t(opt.max_frames));

    const std::filesystem::path out_dir(cfg.output.dir);
    std::filesystem::create_directories(out_dir);
    const std::filesystem::path preview_dir = out_dir / "preview";
    if (cfg.output.preview_every > 0) std::filesystem::create_directories(preview_dir);
    const std::filesystem::path slice_dir = out_dir / "slices";
    if (cfg.output.slices_every > 0) std::filesystem::create_directories(slice_dir);

    std::unique_ptr<Scene> scene = make_scene(cfg, dev);
    log_info("memory: simulation %s (export reuses its scratch)", format_bytes(double(scene->memory_bytes())).c_str());
    if (dev == Device::CUDA) {
        const DeviceMemoryInfo mem = device_memory_info(dev);
        log_info("GPU memory: %s free of %s", format_bytes(double(mem.free_bytes)).c_str(),
                 format_bytes(double(mem.total_bytes)).c_str());
    }

    const std::size_t n = std::size_t(g.count());
    const bool want_velocity = wants(cfg.output.fields, "velocity");
    const bool want_vorticity = wants(cfg.output.fields, "vorticity") || cfg.output.slices_every > 0;
    // Density, dust, cloud water, temperature and flame are always needed (light, bounds, previews).
    std::vector<float> h_density(n), h_dust(n), h_cond(n), h_temperature(n), h_flame(n);
    std::vector<float> h_vx(want_velocity ? n : 0), h_vy(want_velocity ? n : 0), h_vz(want_velocity ? n : 0);
    std::vector<float> h_vort(want_vorticity ? n : 0);

    Manifest manifest;
    manifest.scene_name = cfg.name;
    manifest.prefix = cfg.output.prefix;
    manifest.fps = cfg.output.fps;
    manifest.world_scale = cfg.output.world_scale;
    manifest.fields = cfg.output.fields;

    VdbWriteOptions vdb_opt;
    vdb_opt.world_scale = cfg.output.world_scale;
    vdb_opt.half_float = cfg.output.half_float;
    vdb_opt.tolerance = cfg.output.prune_tolerance;

    const Timer total;
    double t_sim = 0.0, t_io = 0.0;
    std::size_t total_bytes = 0;
    for (std::size_t f = 0; f < times.size(); ++f) {
        const int number = int(f) + 1;
        const double t = times[f];
        // Sim time covered by one video frame around this one: drives motion blur in slow motion.
        const double sim_dt = times.size() < 2 ? 1.0 / cfg.output.fps
                              : f == 0      ? times[1] - times[0]
                                            : times[f] - times[f - 1];

        Timer step;
        scene->advance_to(t);
        device_synchronize(dev);
        const double frame_sim = step.seconds();
        t_sim += frame_sim;

        step.reset();
        scene->export_field(ExportField::Density, h_density.data());
        scene->export_field(ExportField::Dust, h_dust.data());
        scene->export_field(ExportField::Condensate, h_cond.data());
        scene->export_field(ExportField::Temperature, h_temperature.data());
        scene->export_field(ExportField::Flame, h_flame.data());
        if (want_velocity) {
            scene->export_field(ExportField::VelX, h_vx.data());
            scene->export_field(ExportField::VelY, h_vy.data());
            scene->export_field(ExportField::VelZ, h_vz.data());
        }
        if (want_vorticity) scene->export_field(ExportField::Vorticity, h_vort.data());

        const GridInfo fg = scene->grid();  // the domain may have grown
        VdbFrame frame;
        frame.grid = fg;
        if (wants(cfg.output.fields, "density")) frame.scalars.push_back({"density", h_density.data()});
        if (wants(cfg.output.fields, "dust")) frame.scalars.push_back({"dust", h_dust.data()});
        if (wants(cfg.output.fields, "condensate")) frame.scalars.push_back({"condensate", h_cond.data()});
        if (wants(cfg.output.fields, "temperature")) frame.scalars.push_back({"temperature", h_temperature.data()});
        if (wants(cfg.output.fields, "flame")) frame.scalars.push_back({"flame", h_flame.data()});
        if (wants(cfg.output.fields, "vorticity")) frame.scalars.push_back({"vorticity", h_vort.data()});
        if (want_velocity) {
            // Exported in world units per *video* second, so Blender's motion blur matches the slow motion.
            const float scale = float(cfg.output.world_scale * sim_dt * cfg.output.fps);
            frame.vectors.push_back({"velocity", h_vx.data(), h_vy.data(), h_vz.data(), scale});
        }
        frame.metadata["sim_time"] = float(t);
        frame.metadata["sim_dt"] = float(sim_dt);

        const std::string file = frame_file_name(cfg.output.prefix, number);
        const VdbWriteStats stats = write_vdb((out_dir / file).string(), frame, vdb_opt);
        total_bytes += stats.file_bytes;

        ManifestFrame mf;
        mf.number = number;
        mf.file = file;
        mf.sim_time = t;
        mf.sim_dt = sim_dt;
        mf.voxel_size = fg.dx * cfg.output.world_scale;
        mf.domain_min = fg.origin * cfg.output.world_scale;
        mf.domain_max = (fg.origin + fg.extent()) * cfg.output.world_scale;
        measure_light(fg, h_flame, mf.light_position, mf.light_power);
        measure_cloud(fg, h_density, h_dust, h_flame, mf.cloud_min, mf.cloud_max);
        {
            Float3 lo, hi;  // cloud water counts as visible too
            measure_cloud(fg, h_cond, h_cond, h_cond, lo, hi);
            if (hi.z > lo.z) {
                if (mf.cloud_max.z > mf.cloud_min.z) {
                    mf.cloud_min = Float3{std::min(lo.x, mf.cloud_min.x), std::min(lo.y, mf.cloud_min.y), std::min(lo.z, mf.cloud_min.z)};
                    mf.cloud_max = Float3{std::max(hi.x, mf.cloud_max.x), std::max(hi.y, mf.cloud_max.y), std::max(hi.z, mf.cloud_max.z)};
                } else {
                    mf.cloud_min = lo;
                    mf.cloud_max = hi;
                }
            }
        }
        mf.cloud_min = mf.cloud_min * cfg.output.world_scale;
        mf.cloud_max = mf.cloud_max * cfg.output.world_scale;
        mf.light_position = mf.light_position * cfg.output.world_scale;
        mf.active_voxels = stats.active_voxels;
        mf.file_bytes = stats.file_bytes;
        manifest.frames.push_back(mf);
        write_manifest((out_dir / "manifest.json").string(), manifest);

        if (cfg.output.preview_every > 0 && (f % std::size_t(cfg.output.preview_every) == 0 || f + 1 == times.size())) {
            char name[32];
            std::snprintf(name, sizeof(name), "preview_%04d.png", number);
            PreviewInput pin{fg, h_density.data(), h_temperature.data(), h_flame.data(), h_dust.data(), h_cond.data()};
            render_preview_png((preview_dir / name).string(), pin, PreviewOptions{});
        }
        if (cfg.output.slices_every > 0 && (f % std::size_t(cfg.output.slices_every) == 0 || f + 1 == times.size())) {
            char name[32];
            std::snprintf(name, sizeof(name), "slice_%04d.png", number);
            const SliceInput sin{fg, h_density.data(), h_dust.data(), h_cond.data(), h_vort.data()};
            render_slice_png((slice_dir / name).string(), sin);
        }
        t_io += step.seconds();

        log_info("frame %4d/%zu  t=%9.4f s  sim %s  active=%9zu voxels  file=%s  %s", number, times.size(), t,
                 format_seconds(frame_sim).c_str(), stats.active_voxels, format_bytes(double(stats.file_bytes)).c_str(),
                 scene->status().c_str());
        const std::string prof = scene->profile_report();
        if (!prof.empty()) log_info("  profile: %s", prof.c_str());
    }

    log_info("done: %zu frames in %s (compute %s, export %s), %s written to %s", times.size(),
             format_seconds(total.seconds()).c_str(), format_seconds(t_sim).c_str(), format_seconds(t_io).c_str(),
             format_bytes(double(total_bytes)).c_str(), out_dir.string().c_str());
    return 0;
}

}  // namespace sim
