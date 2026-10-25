#include <doctest/doctest.h>

#include <string>

#include "io/config.h"

using namespace sim;

TEST_CASE("test_fireball scene parses") {
    const SceneConfig cfg = load_scene(std::string(SIM_SOURCE_DIR) + "/scenes/test_fireball.toml");
    CHECK(cfg.type == "test_fireball");
    CHECK(cfg.grid.dims == Int3{160, 160, 192});
    CHECK(cfg.grid.dx == doctest::Approx(4.f));
    CHECK(cfg.grid.origin.x == doctest::Approx(-320.f));
    CHECK(cfg.grid.origin.z == doctest::Approx(0.f));
    CHECK(cfg.output.schedule.segments.size() == 2);
    CHECK(cfg.output.schedule.total_frames() == 120);
    CHECK(cfg.output.fields.size() == 4);
}

TEST_CASE("missing file reports an error") {
    CHECK_THROWS(load_scene(std::string(SIM_SOURCE_DIR) + "/scenes/does_not_exist.toml"));
}

TEST_CASE("fireball scenes parse") {
    for (const char* name : {"/scenes/fireball_rise_preview.toml", "/scenes/surface_20kt.toml"}) {
        CAPTURE(name);
        const SceneConfig cfg = load_scene(std::string(SIM_SOURCE_DIR) + name);
        CHECK(cfg.type == "fireball_rise");
        CHECK(cfg.output.schedule.start_time >= cfg.fireball.start_time);
        CHECK(cfg.grid.dims.x % 16 == 0);
        CHECK(cfg.grid.dims.z % 16 == 0);
        CHECK(cfg.domain.expand);
        CHECK(cfg.dust.enabled);
        CHECK(cfg.wind.points.size() == 4);
        CHECK(cfg.atmosphere.ground_altitude == doctest::Approx(1200.f));
        CHECK(cfg.solver.reflection);
    }
}

TEST_CASE("surface burst scenes parse") {
    for (const char* name : {"/scenes/surface_burst_20kt.toml", "/scenes/surface_burst_preview.toml",
                             "/scenes/surface_burst_20kt_max.toml", "/scenes/surface_burst_20kt_ultra.toml",
                             "/scenes/fireball_closeup.toml", "/scenes/mushroom_20kt.toml", "/scenes/sand_20kt.toml"}) {
        CAPTURE(name);
        const SceneConfig cfg = load_scene(std::string(SIM_SOURCE_DIR) + name);
        CHECK(cfg.type == "surface_burst");
        CHECK(cfg.output.schedule.start_time >= cfg.blast.start_time - 1e-6);
        CHECK(cfg.blast.yield_kt == doctest::Approx(20.f));
        CHECK(cfg.blast.handoff_time > cfg.blast.start_time);
        CHECK(cfg.dust_init.skirt_loading == 0.f);
        CHECK(cfg.grid.dims.x % 16 == 0);
        CHECK(cfg.grid.dims.z % 16 == 0);
        CHECK(cfg.solver.confinement_hot_boost > 0.f);
        CHECK(cfg.dust.ground.variation > 0.f);
    }
}
