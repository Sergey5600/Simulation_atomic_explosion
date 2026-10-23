#include "scenes/surface_burst.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "compressible/euler.h"
#include "core/log.h"
#include "scenes/fireball_rise.h"

namespace sim {

namespace {

class SurfaceBurstScene final : public Scene {
public:
    SurfaceBurstScene(const SceneConfig& cfg, Device dev) : cfg_(cfg), dev_(dev), atmosphere_(cfg.atmosphere) {
        blast_ = std::make_unique<CompressibleSolver>(dev, cfg.grid, atmosphere_, cfg.blast, cfg.dust, cfg.humidity);
        if (cfg.profile) blast_->profiler().enable(dev);
        blast_->reset_to_ambient();
        blast_->deposit_energy(cfg.seed);
        time_ = cfg.blast.start_time;
        log_info("surface burst: %.1f kt (%.0f %% into the blast), starts at t = %.4f s, handoff to the buoyant "
                 "phase at t = %.2f s",
                 cfg.blast.yield_kt, 100.0 * cfg.blast.hydro_fraction, cfg.blast.start_time, cfg.blast.handoff_time);
    }

    GridInfo grid() const override { return blast_ ? blast_->grid() : rise_->grid(); }

    void advance_to(double t) override {
        if (t < time_ - 1e-9) throw std::runtime_error("surface_burst: frame time is before blast.start_time");
        steps_ = 0;
        const Timer timer;
        if (blast_) {
            const double handoff = cfg_.blast.handoff_time;
            const double target = std::min(t, handoff);
            while (time_ < target - 1e-12) {
                double dt = blast_->suggest_dt();
                const double remaining = target - time_;
                if (dt >= remaining) dt = remaining;
                else if (dt > 0.5 * remaining) dt = 0.5 * remaining;
                blast_->step(dt);
                time_ += dt;
                ++steps_;
                last_dt_ = dt;
            }
            device_synchronize(dev_);
            step_ms_ = steps_ > 0 ? 1e3 * timer.seconds() / steps_ : 0.0;
            if (t >= handoff - 1e-12) handoff_to_rise();
        }
        if (rise_) rise_->advance_to(t);
    }

    void export_field(ExportField f, float* host) override {
        if (blast_) blast_->export_field_to_host(f, host, cfg_.fireball.flame_temperature);
        else rise_->export_field(f, host);
    }

    std::string status() override {
        if (!blast_) return "[rise] " + rise_->status();
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "[blast] steps=%d %.0fms/step dt=%.2es | Tmax=%.0fK p/p0 max=%.2f shock r=%.0fm |u|max=%.0fm/s",
                      steps_, step_ms_, last_dt_, double(blast_->max_temperature()), double(blast_->max_pressure_ratio()),
                      double(blast_->shock_radius()), double(blast_->max_speed()));
        return buf;
    }

    std::size_t memory_bytes() const override { return blast_ ? blast_->memory_bytes() : rise_->memory_bytes(); }
    std::string profile_report() override {
        return blast_ ? blast_->profiler().report_and_reset() : rise_->profile_report();
    }

private:
    void handoff_to_rise() {
        log_info("handoff at t = %.3f s: max T %.0f K, p/p0 max %.2f, shock at %.0f m", time_,
                 double(blast_->max_temperature()), double(blast_->max_pressure_ratio()), double(blast_->shock_radius()));
        CompressibleSolver::HandoffState h = blast_->handoff_state();
        const GridInfo g = blast_->grid();
        blast_.reset();  // free the GPU memory before allocating stage B

        // The compressible stage ignores the wind; add it back now.
        const float heading = cfg_.wind.heading_deg * 3.14159265f / 180.f;
        for (int k = 0; k < g.dims.z; ++k) {
            const float s = cfg_.wind.speed_at(g.origin.z + (k + 0.5f) * g.dx);
            const float wu = s * std::cos(heading), wv = s * std::sin(heading);
            for (int j = 0; j < g.dims.y; ++j)
                for (int i = 0; i < g.dims.x; ++i) {
                    const std::size_t idx = std::size_t(g.index(i, j, k));
                    h.u[idx] += wu;
                    h.v[idx] += wv;
                }
        }
        rise_ = std::make_unique<RisePhase>(cfg_, dev_, atmosphere_);
        rise_->solver().load_cell_state(h.temperature, h.smoke, h.dust, h.u, h.v, h.w, &h.vapor, &h.condensate);
        rise_->set_time(time_);
    }

    SceneConfig cfg_;
    Device dev_;
    Atmosphere atmosphere_;
    std::unique_ptr<CompressibleSolver> blast_;
    std::unique_ptr<RisePhase> rise_;
    double time_ = 0.0, last_dt_ = 0.0, step_ms_ = 0.0;
    int steps_ = 0;
};

}  // namespace

std::unique_ptr<Scene> make_surface_burst_scene(const SceneConfig& cfg, Device dev) {
    return std::make_unique<SurfaceBurstScene>(cfg, dev);
}

}  // namespace sim
