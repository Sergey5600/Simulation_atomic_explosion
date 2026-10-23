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
