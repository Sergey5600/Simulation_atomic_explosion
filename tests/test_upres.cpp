#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "lowmach/mac.h"
#include "physics/ground.h"
#include "upres/rest_fields.h"

using namespace sim;

namespace {

Device test_device() { return cuda_available() ? Device::CUDA : Device::CPU; }

GridInfo sim_grid() {
    GridInfo g;
    g.dims = Int3{32, 16, 16};
    g.dx = 10.f;
    g.origin = Float3{-160.f, -80.f, 0.f};
    return g;
}

}  // namespace

TEST_CASE("rest fields start at the cell positions and follow a uniform flow") {
    const Device dev = test_device();
    const GridInfo g = sim_grid();
    RestParams p;
    p.feature_voxels = 2.f;  // L = 20 m
    p.growth_ratio = 1e6f;  // no reset during the test
    p.clock_origin = 0.37f;  // keeps t = 1 .. 11 s away from a period boundary
    RestFields rest(dev, g, p);
    rest.reset(1.0);

    const GridInfo& rg = rest.grid();
    CHECK(rg.dims.x == 16);
    CHECK(rg.dx == doctest::Approx(20.f));
    const float L = 20.f;
    {
        const std::vector<float> x = rest.component(0, 0).to_host();
        const Float3 c = rg.cell_center(5, 3, 2);
        CHECK(x[std::size_t(rg.index(5, 3, 2))] == doctest::Approx(c.x / L));
    }

    // 4 m/s along x for 10 s: material moves 40 m, so each point now holds the coordinate
    // of the point 40 m upstream.
    MacField vel(g.dims, dev);
    std::vector<float> u(vel.u.size(), 4.f);
    vel.u.upload(u.data(), u.size());
    for (int s = 0; s < 5; ++s) rest.advect(vel.sampler(), 2.0, 1.0 + 2.0 * (s + 1));

    for (int f = 0; f < 2; ++f) {
        const std::vector<float> x = rest.component(f, 0).to_host();
        const std::vector<float> z = rest.component(f, 2).to_host();
        const Float3 c = rg.cell_center(10, 4, 4);
        CHECK(x[std::size_t(rg.index(10, 4, 4))] == doctest::Approx((c.x - 40.f) / L).epsilon(1e-4));
        CHECK(z[std::size_t(rg.index(10, 4, 4))] == doctest::Approx(c.z / L).epsilon(1e-4));
        // Air entering at the upstream edge starts at its own position (one step = 8 m back).
        CHECK(x[std::size_t(rg.index(0, 4, 4))] == doctest::Approx((rg.cell_center(0, 4, 4).x - 8.f) / L).epsilon(1e-4));
    }
}

TEST_CASE("rest fields cross-fade on a logarithmic clock") {
    const GridInfo g = sim_grid();
    RestParams p;
    p.growth_ratio = 2.f;
    p.clock_origin = 1.f;
    RestFields rest(Device::CPU, g, p);
    // Field A resets at t = 1, 2, 4, 8 ... (weight 0), field B at sqrt(2) times those.
    CHECK(rest.weight_a(2.0) == doctest::Approx(0.f).epsilon(1e-6));
    CHECK(rest.weight_a(4.0 * std::sqrt(2.0)) == doctest::Approx(1.f).epsilon(1e-5));
    const float w = rest.weight_a(3.0);
    CHECK(w > 0.f);
    CHECK(w < 1.f);

    // A reset moves the field back to the current positions.
    rest.reset(1.5);
    MacField vel(g.dims, Device::CPU);
    std::vector<float> u(vel.u.size(), 5.f);
    vel.u.upload(u.data(), u.size());
    rest.advect(vel.sampler(), 0.4, 1.9);  // clock still in A's period: A keeps its shift
    rest.advect(vel.sampler(), 0.2, 2.1);  // A's reset at t = 2
    const GridInfo& rg = rest.grid();
    const std::vector<float> ax = rest.component(0, 0).to_host();
    const std::vector<float> bx = rest.component(1, 0).to_host();
    const std::size_t idx = std::size_t(rg.index(8, 4, 4));
    const float L = p.feature_voxels * g.dx;
    CHECK(ax[idx] == doctest::Approx(rg.cell_center(8, 4, 4).x / L));
    CHECK(bx[idx] == doctest::Approx((rg.cell_center(8, 4, 4).x - 3.f) / L).epsilon(1e-4));
}

TEST_CASE("rest fields follow the domain when it doubles") {
    const GridInfo g = sim_grid();
    RestParams p;
    RestFields rest(Device::CPU, g, p);
    rest.reset(1.0);
    GridInfo ng = g;
    ng.dx = 2.f * g.dx;
    ng.origin = Float3{-320.f, -160.f, 0.f};
    rest.coarsen(ng);
    const GridInfo& rg = rest.grid();
    CHECK(rg.dx == doctest::Approx(40.f));
    // Unmoved material keeps its coordinates everywhere: inside the old domain (sampled) and
    // outside (set from the position).
    const std::vector<float> x = rest.component(0, 0).to_host();
    const float L = p.feature_voxels * g.dx;
    for (int i : {1, 8, 14}) {
        CHECK(x[std::size_t(rg.index(i, 4, 2))] == doctest::Approx(rg.cell_center(i, 4, 2).x / L).epsilon(1e-5));
    }
}

TEST_CASE("patchy ground keeps the mean dust lifting and has bare patches") {
    double sum = 0.0;
    int bare = 0, n = 0;
    float lo = 1e9f, hi = 0.f;
    for (int j = 0; j < 200; ++j)
        for (int i = 0; i < 200; ++i) {
            const float e = ground_erodibility(i * 25.f, j * 25.f, 0.8f, 1.f / 250.f, 7u);
            sum += e;
            bare += e == 0.f;
            lo = std::min(lo, e);
            hi = std::max(hi, e);
            ++n;
        }
    CHECK(sum / n == doctest::Approx(1.0).epsilon(0.15));
    CHECK(lo >= 0.f);
    CHECK(bare > 0);       // hard ground between the sand
    CHECK(hi > 1.5f);      // and loose patches that give more
    CHECK(ground_erodibility(10.f, 20.f, 0.f, 1.f / 250.f, 7u) == 1.f);
}

TEST_CASE("rest distortion is zero after a reset and measures the shear since") {
    const Device dev = test_device();
    const GridInfo g = sim_grid();
    RestParams p;
    p.feature_voxels = 2.f;
    p.growth_ratio = 1e6f;  // no reset during the test
    p.clock_origin = 0.37f;
    RestFields rest(dev, g, p);
    rest.reset(1.0);
    const GridInfo& rg = rest.grid();
    std::vector<float> d(std::size_t(rg.count()));
    rest.distortion_to_host(0, d.data());
    CHECK(*std::max_element(d.begin(), d.end()) < 1e-4f);

    // Shear u = s z for t seconds: the rest map x0 = x - s t z has |J|^2 = 3 + (s t)^2, so the
    // distortion is (s t)^2 / 3.
    const float s = 0.05f;
    MacField vel(g.dims, dev);
    const Int3 nu = u_dims(g.dims);
    std::vector<float> u(vel.u.size());
    for (int k = 0; k < nu.z; ++k)
        for (int j = 0; j < nu.y; ++j)
            for (int i = 0; i < nu.x; ++i) u[std::size_t(i + nu.x * (j + std::int64_t(nu.y) * k))] = s * (k + 0.5f) * g.dx;
    vel.u.upload(u.data(), u.size());
    for (int step = 0; step < 10; ++step) rest.advect(vel.sampler(), 1.0, 1.0 + (step + 1));
    rest.distortion_to_host(0, d.data());
    const float expected = (s * 10.f) * (s * 10.f) / 3.f;
    // Away from the inflow edge and the top and bottom.
    const float v = d[std::size_t(rg.index(rg.dims.x - 3, rg.dims.y / 2, rg.dims.z / 2))];
    MESSAGE("distortion " << v << ", expected " << expected);
    CHECK(v == doctest::Approx(expected).epsilon(0.15));
}
