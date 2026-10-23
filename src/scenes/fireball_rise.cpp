#include "scenes/fireball_rise.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "core/log.h"
#include "core/noise.h"
#include "core/parallel.h"
#include "lowmach/solver.h"

namespace sim {

void init_fireball(LowMachSolver& solver, const FireballConfig& cfg, std::uint32_t seed) {
    const GridInfo g = solver.grid();
    FieldView th = solver.theta(), sm = solver.smoke();
    const float* exner = solver.ambient().exner.data();
    const float* t_amb = solver.ambient().temperature.data();
    const float radius = cfg.radius, cz = cfg.center_height, vscale = cfg.vertical_scale;
    const float t_core = cfg.temperature, edge = cfg.edge_width;
    const float amp = cfg.noise_amplitude, freq = cfg.noise_frequency;

    parallel_for(solver.device(), g.dims, SIM_LAMBDA(int i, int j, int k) {
        const Float3 p = g.cell_center(i, j, k);
        const Float3 rel = make_f3(p.x / radius, p.y / radius, (p.z - cz) / (radius * vscale));
        const float d = length(rel) + amp * noise::fbm(p * freq, 5, 2.0f, 0.5f, seed);
        const float s = smoothstepf(1.f, 1.f - edge, d);
        if (s <= 0.f) return;
        const float ta = t_amb[k];
        const float t = ta + (t_core - ta) * s;
        th(i, j, k) = t / exner[k];
        sm(i, j, k) = s;
    });
}

void init_dust(LowMachSolver& solver, const FireballConfig& fireball, const DustInitConfig& cfg, std::uint32_t seed) {
    if (!solver.has_dust()) return;
    const GridInfo g = solver.grid();
    const float radius = fireball.radius, cz = fireball.center_height, vscale = fireball.vertical_scale;
    const float edge = fireball.edge_width, amp = fireball.noise_amplitude, freq = fireball.noise_frequency;
    const float skirt_r = cfg.skirt_radius, skirt_h = cfg.skirt_height, skirt_freq = cfg.noise_frequency;
    for (int b = 0; b < kDustBins; ++b) {
        FieldView d = solver.dust(b);
        const float fb_load = cfg.fireball_loading * cfg.fireball_split[b];
        const float sk_load = cfg.skirt_loading * cfg.skirt_split[b];
        parallel_for(solver.device(), g.dims, SIM_LAMBDA(int i, int j, int k) {
            const Float3 p = g.cell_center(i, j, k);
            // Same displaced boundary as the temperature in init_fireball.
            const Float3 rel = make_f3(p.x / radius, p.y / radius, (p.z - cz) / (radius * vscale));
            const float dist = length(rel) + amp * noise::fbm(p * freq, 5, 2.0f, 0.5f, seed);
            float y = fb_load * smoothstepf(1.f, 1.f - edge, dist);
            // Skirt: a ground-hugging layer, thickest at ground zero, with a ragged top.
            const float r = sqrtf(p.x * p.x + p.y * p.y) / skirt_r;
            if (r < 1.f && skirt_h > 0.f) {
                const float n = noise::fbm(make_f3(p.x, p.y, p.z * 2.f) * skirt_freq, 4, 2.1f, 0.55f, seed + 911u);
                const float h = skirt_h * sqrtf(1.f - r * r) * (0.75f + 0.5f * n);
                y += sk_load * smoothstepf(h, 0.5f * h, p.z) * smoothstepf(1.f, 0.7f, r);
            }
            d(i, j, k) += y;
        });
    }
}

RisePhase::RisePhase(const SceneConfig& cfg, Device dev, const Atmosphere& atm)
    : cfg_(cfg), solver_(dev, cfg.grid, atm, cfg.solver, cfg.radiation, cfg.dust, cfg.humidity) {
    solver_.reset_to_ambient(cfg.wind);
    if (cfg.domain.expand && !LowMachSolver::can_coarsen(cfg.grid.dims)) {
        log_warn("domain.expand needs grid x, y divisible by 4 and z by 2: the domain will not grow");
    }
}

void RisePhase::advance_to(double t) {
    if (t < time_ - 1e-9) throw std::runtime_error("frame time is before the start of the buoyant phase");
    steps_ = 0;
    iterations_ = 0;
    projections_ = 0;
    residual_ = 0.0;
    while (time_ < t - 1e-9) {
        double dt = solver_.suggest_dt();
        const double remaining = t - time_;
        // Land exactly on the frame time without a tiny last step.
        if (dt >= remaining) dt = remaining;
        else if (dt > 0.5 * remaining) dt = 0.5 * remaining;
        const StepStats s = solver_.step(dt);
        time_ += dt;
        maybe_grow_domain();
        ++steps_;
        iterations_ += s.pcg_iterations;
        projections_ += s.projections;
        residual_ = std::max(residual_, s.max_residual);
        last_dt_ = dt;
    }
}

void RisePhase::fill_export(ExportBuffers& out) {
    solver_.fill_export(out.density.view(), out.dust.view(), out.condensate.view(), out.temperature.view(),
                        out.flame.view(), out.vel_x.view(), out.vel_y.view(), out.vel_z.view(),
                        cfg_.fireball.flame_temperature);
}

std::string RisePhase::status() {
    const CloudDiagnostics d = solver_.diagnostics();
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "steps=%d dt=%.3fs pcg=%.1f it/solve res=%.1e | Tmax=%.0fK top=%.0fm centroid=%.0fm "
                  "|u|max=%.0fm/s at z=%.0fm dust=%.0ft cloud water=%.0ft",
                  steps_, last_dt_, projections_ ? double(iterations_) / projections_ : 0.0, residual_,
                  double(solver_.max_temperature()), double(d.top), double(d.centroid_z), double(d.max_speed),
                  double(d.max_speed_z), d.dust_tonnes, solver_.condensate_tonnes());
    return buf;
}

void RisePhase::maybe_grow_domain() {
    const DomainConfig& d = cfg_.domain;
    if (!d.expand || expansions_ >= d.max_expansions || !LowMachSolver::can_coarsen(cfg_.grid.dims)) return;
    if (++steps_since_check_ < 5) return;
    steps_since_check_ = 0;
    const GridInfo g = solver_.grid();
    const CloudExtent e = solver_.cloud_extent(d.smoke_threshold);
    const float height = g.dims.z * g.dx;
    const float half_width = 0.5f * std::min(g.dims.x, g.dims.y) * g.dx;
    if (e.top < d.trigger * height && e.radius < d.trigger * half_width) return;
    solver_.coarsen_domain();
    ++expansions_;
    const GridInfo ng = solver_.grid();
    log_info("domain grown (cloud top %.0f m, radius %.0f m): voxel %.3g m, domain %.0f x %.0f x %.0f m", e.top,
             e.radius, ng.dx, ng.extent().x, ng.extent().y, ng.extent().z);
}

namespace {

class FireballRiseScene final : public Scene {
public:
    FireballRiseScene(const SceneConfig& cfg, Device dev) : atmosphere_(cfg.atmosphere), rise_(cfg, dev, atmosphere_) {
        init_fireball(rise_.solver(), cfg.fireball, cfg.seed);
        init_dust(rise_.solver(), cfg.fireball, cfg.dust_init, cfg.seed);
        rise_.set_time(cfg.fireball.start_time);
        const AtmosphereState ground = atmosphere_.at(0.0);
        log_info("atmosphere at ground: %.1f K, %.0f Pa, %.3f kg/m^3", ground.temperature, ground.pressure,
                 ground.density);
        log_info("fireball: radius %.0f m, core %.0f K, starts at t = %.2f s", cfg.fireball.radius,
                 cfg.fireball.temperature, cfg.fireball.start_time);
    }

    GridInfo grid() const override { return rise_.grid(); }
    void advance_to(double t) override { rise_.advance_to(t); }
    void fill_export(ExportBuffers& out) override { rise_.fill_export(out); }
    std::string status() override { return rise_.status(); }
    std::size_t memory_bytes() const override { return rise_.memory_bytes(); }

private:
    Atmosphere atmosphere_;
    RisePhase rise_;
};

}  // namespace

std::unique_ptr<Scene> make_fireball_rise_scene(const SceneConfig& cfg, Device dev) {
    return std::make_unique<FireballRiseScene>(cfg, dev);
}

}  // namespace sim
