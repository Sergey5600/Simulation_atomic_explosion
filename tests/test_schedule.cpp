#include <doctest/doctest.h>

#include <cmath>

#include "core/schedule.h"

using namespace sim;

TEST_CASE("constant schedule is evenly spaced") {
    ExportSchedule s;
    s.start_time = 1.0;
    s.segments.push_back({ScheduleSegment::Mode::Constant, 5, 0.5, 0.5});
    const auto t = s.frame_times();
    REQUIRE(t.size() == 5);
    CHECK(t.front() == doctest::Approx(1.0));
    CHECK(t.back() == doctest::Approx(3.0));
}

TEST_CASE("ramp eases from slow motion into real time") {
    ExportSchedule s;
    s.start_time = 0.002;
    s.segments.push_back({ScheduleSegment::Mode::Ramp, 240, 0.00025, 1.0 / 24.0});
    s.segments.push_back({ScheduleSegment::Mode::Constant, 24, 1.0 / 24.0, 1.0 / 24.0});
    const auto t = s.frame_times();
    REQUIRE(t.size() == 264);
    REQUIRE(s.total_frames() == 264);
    for (std::size_t i = 1; i < t.size(); ++i) REQUIRE(t[i] > t[i - 1]);
    // Steps grow monotonically across the ramp and stay constant afterwards.
    for (std::size_t i = 2; i < 240; ++i) REQUIRE(t[i] - t[i - 1] >= t[i - 1] - t[i - 2] - 1e-12);
    CHECK(t[1] - t[0] == doctest::Approx(0.00025 * std::pow((1.0 / 24.0) / 0.00025, 1.0 / 239.0)));
    CHECK(t[239] - t[238] == doctest::Approx(1.0 / 24.0));
    CHECK(t[263] - t[262] == doctest::Approx(1.0 / 24.0));
    // The ramp covers roughly the first two seconds of the explosion.
    CHECK(t[239] > 1.5);
    CHECK(t[239] < 2.5);
}
