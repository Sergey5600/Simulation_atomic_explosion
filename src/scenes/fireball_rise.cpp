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

namespace {

class FireballRiseScene final : public Scene {
public:
    FireballRiseScene(const SceneConfig& cfg, Device dev)
        : cfg_(cfg), atmosphere_(cfg.atmosphere), solver_(dev, cfg.grid, atmosphere_, cfg.solver, cfg.radiation) {
        solver_.reset_to_ambient(cfg.wind);
        init_fireball(solver_, cfg.fireball, cfg.seed);
        time_ = cfg.fireball.start_time;
        const AtmosphereState ground = atmosphere_.at(0.0);
        log_info("atmosphere at ground: %.1f K, %.0f Pa, %.3f kg/m^3", ground.temperature, ground.pressure,
                 ground.density);
        log_info("fireball: radius %.0f m, core %.0f K, starts at t = %.2f s", cfg.fireball.radius,
                 cfg.fireball.temperature, cfg.fireball.start_time);
    }

    void advance_to(double t) override {
        if (t < time_ - 1e-9) {
            throw std::runtime_error("fireball_rise: frame time is before fireball.start_time; set "
                                     "output.start_time >= fireball.start_time");
        }
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
            ++steps_;
            iterations_ += s.pcg_iterations;
            projections_ += s.projections;
            residual_ = std::max(residual_, s.max_residual);
            last_dt_ = dt;
        }
    }

    void fill_export(ExportBuffers& out) override {
        solver_.fill_export(out.density.view(), out.temperature.view(), out.flame.view(), out.vel_x.view(),
                            out.vel_y.view(), out.vel_z.view(), cfg_.fireball.flame_temperature);
    }

    std::string status() override {
        const CloudDiagnostics d = solver_.diagnostics();
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "steps=%d dt=%.3fs pcg=%.1f it/solve res=%.1e | Tmax=%.0fK top=%.0fm centroid=%.0fm "
                      "|u|max=%.0fm/s at z=%.0fm",
                      steps_, last_dt_, projections_ ? double(iterations_) / projections_ : 0.0, residual_,
                      double(solver_.max_temperature()), double(d.top), double(d.centroid_z), double(d.max_speed),
                      double(d.max_speed_z));
        return buf;
    }

    std::size_t memory_bytes() const override { return solver_.memory_bytes(); }

private:
    SceneConfig cfg_;
    Atmosphere atmosphere_;
    LowMachSolver solver_;
    double time_ = 0.0;
    int steps_ = 0, iterations_ = 0, projections_ = 0;
    double residual_ = 0.0, last_dt_ = 0.0;
};

}  // namespace

std::unique_ptr<Scene> make_fireball_rise_scene(const SceneConfig& cfg, Device dev) {
    return std::make_unique<FireballRiseScene>(cfg, dev);
}

}  // namespace sim
