#include "scenes/scene.h"

#include <algorithm>
#include <stdexcept>

#include "scenes/fireball_rise.h"
#include "scenes/surface_burst.h"
#include "scenes/test_fireball.h"

namespace sim {

namespace {

class TestFireballScene final : public Scene {
public:
    TestFireballScene(const SceneConfig& cfg, Device dev)
        : cfg_(cfg),
          density_("density", cfg.grid, dev),
          temperature_("temperature", cfg.grid, dev),
          flame_("flame", cfg.grid, dev),
          vx_("vel_x", cfg.grid, dev),
          vy_("vel_y", cfg.grid, dev),
          vz_("vel_z", cfg.grid, dev) {}

    void advance_to(double t) override {
        time_ = t;
        generated_ = false;
    }
    GridInfo grid() const override { return cfg_.grid; }

    void export_field(ExportField f, float* host) override {
        const std::size_t n = std::size_t(cfg_.grid.count());
        if (!generated_) {
            TestFireballFields fields{&density_, &temperature_, &flame_, &vx_, &vy_, &vz_};
            generate_test_fireball(cfg_.test_fireball, cfg_.seed, time_, fields);
            generated_ = true;
        }
        const ScalarField* src = nullptr;
        switch (f) {
            case ExportField::Density: src = &density_; break;
            case ExportField::Temperature: src = &temperature_; break;
            case ExportField::Flame: src = &flame_; break;
            case ExportField::VelX: src = &vx_; break;
            case ExportField::VelY: src = &vy_; break;
            case ExportField::VelZ: src = &vz_; break;
            default: break;  // no dust, cloud water or vorticity in the analytic ball
        }
        if (src) src->buffer().download(host, n);
        else std::fill(host, host + n, 0.f);
    }

private:
    SceneConfig cfg_;
    ScalarField density_, temperature_, flame_, vx_, vy_, vz_;
    double time_ = 0.0;
    bool generated_ = false;
};

}  // namespace

std::unique_ptr<Scene> make_scene(const SceneConfig& cfg, Device dev) {
    if (cfg.type == "test_fireball") return std::make_unique<TestFireballScene>(cfg, dev);
    if (cfg.type == "fireball_rise") return make_fireball_rise_scene(cfg, dev);
    if (cfg.type == "surface_burst") return make_surface_burst_scene(cfg, dev);
    throw std::runtime_error("unknown scene type '" + cfg.type +
                             "' (available: test_fireball, fireball_rise, surface_burst)");
}

}  // namespace sim
