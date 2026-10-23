#include "scenes/scene.h"

#include <stdexcept>

#include "scenes/fireball_rise.h"
#include "scenes/surface_burst.h"
#include "scenes/test_fireball.h"

namespace sim {

namespace {

class TestFireballScene final : public Scene {
public:
    TestFireballScene(const SceneConfig& cfg) : cfg_(cfg) {}

    void advance_to(double t) override { time_ = t; }
    GridInfo grid() const override { return cfg_.grid; }

    void fill_export(ExportBuffers& out) override {
        TestFireballFields f{&out.density, &out.temperature, &out.flame, &out.vel_x, &out.vel_y, &out.vel_z};
        generate_test_fireball(cfg_.test_fireball, cfg_.seed, time_, f);
        out.dust.buffer().zero();
        out.condensate.buffer().zero();
    }

private:
    SceneConfig cfg_;
    double time_ = 0.0;
};

}  // namespace

std::unique_ptr<Scene> make_scene(const SceneConfig& cfg, Device dev) {
    if (cfg.type == "test_fireball") return std::make_unique<TestFireballScene>(cfg);
    if (cfg.type == "fireball_rise") return make_fireball_rise_scene(cfg, dev);
    if (cfg.type == "surface_burst") return make_surface_burst_scene(cfg, dev);
    throw std::runtime_error("unknown scene type '" + cfg.type +
                             "' (available: test_fireball, fireball_rise, surface_burst)");
}

}  // namespace sim
