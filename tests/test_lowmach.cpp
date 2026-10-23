#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "core/field_ops.h"
#include "core/memory.h"
#include "lowmach/advection.h"
#include "lowmach/pressure.h"
#include "lowmach/solver.h"
#include "physics/atmosphere.h"
#include "physics/constants.h"
#include "scenes/fireball_rise.h"

using namespace sim;

namespace {

Device test_device() { return cuda_available() ? Device::CUDA : Device::CPU; }

std::vector<float> host(const Buffer<float>& b) { return b.to_host(); }

// Divergence of a MAC field, computed on the host.
std::vector<float> divergence(Int3 n, float dx, const std::vector<float>& u, const std::vector<float>& v,
                              const std::vector<float>& w) {
    std::vector<float> d(std::size_t(n.volume()));
    const Int3 nu = u_dims(n), nv = v_dims(n), nw = w_dims(n);
    auto iu = [&](int i, int j, int k) { return std::size_t(i + nu.x * (j + std::int64_t(nu.y) * k)); };
    auto iv = [&](int i, int j, int k) { return std::size_t(i + nv.x * (j + std::int64_t(nv.y) * k)); };
    auto iw = [&](int i, int j, int k) { return std::size_t(i + nw.x * (j + std::int64_t(nw.y) * k)); };
    for (int k = 0; k < n.z; ++k)
        for (int j = 0; j < n.y; ++j)
            for (int i = 0; i < n.x; ++i)
                d[std::size_t(i + n.x * (j + std::int64_t(n.y) * k))] =
                    (u[iu(i + 1, j, k)] - u[iu(i, j, k)] + v[iv(i, j + 1, k)] - v[iv(i, j, k)] + w[iw(i, j, k + 1)] -
                     w[iw(i, j, k)]) /
                    dx;
    return d;
}

AtmosphereConfig sea_level_standard() {
    AtmosphereConfig c;
    c.ground_altitude = 0.f;
    c.ground_temperature = 288.15f;
    return c;
}

}  // namespace

TEST_CASE("atmosphere reproduces the standard atmosphere") {
    const Atmosphere atm(sea_level_standard());
    const AtmosphereState s0 = atm.at(0.0);
    CHECK(s0.temperature == doctest::Approx(288.15));
    CHECK(s0.pressure == doctest::Approx(101325.0));
    CHECK(s0.density == doctest::Approx(1.225).epsilon(0.002));
    const AtmosphereState s11 = atm.at(11000.0);
    CHECK(s11.temperature == doctest::Approx(216.65));
    CHECK(s11.pressure == doctest::Approx(22632.0).epsilon(0.005));
    const AtmosphereState s20 = atm.at(20000.0);
    CHECK(s20.pressure == doctest::Approx(5474.9).epsilon(0.01));
}

TEST_CASE("desert atmosphere is hydrostatic and stably stratified") {
    AtmosphereConfig cfg;  // defaults: 1200 m, 303 K
    const Atmosphere atm(cfg);
    CHECK(atm.at(0.0).pressure == doctest::Approx(87700.0).epsilon(0.01));
    double prev_theta = 0.0;
    for (double z = 0.0; z < 15000.0; z += 250.0) {
        const AtmosphereState a = atm.at(z), b = atm.at(z + 1.0);
        // dp/dz = -rho g
        CHECK((b.pressure - a.pressure) == doctest::Approx(-a.density * phys::kGravity).epsilon(0.01));
        CHECK(a.theta >= prev_theta);
        prev_theta = a.theta;
    }
}

TEST_CASE("reductions agree with the host") {
    GridInfo g;
    g.dims = Int3{17, 9, 6};
    ScalarField f("f", g, test_device());
    fill_linear_index(f);
    const double n = double(g.count());
    CHECK(sum(f) == doctest::Approx(n * (n - 1) / 2));
    CHECK(max_abs(f) == doctest::Approx(n - 1));
    axpy(f, -2.f, f);  // f = -index
    CHECK(max_abs(f) == doctest::Approx(n - 1));
}

TEST_CASE("projection enforces div(u) = S with a 25:1 density contrast") {
    const Device dev = test_device();
    const Int3 n{32, 32, 32};
    const float dx = 10.f;
    GridInfo g;
    g.dims = n;
    g.dx = dx;

    // Light (hot) sphere in heavy air.
    std::vector<float> rho(std::size_t(n.volume()), 1.2f), src(std::size_t(n.volume()), 0.f);
    for (int k = 0; k < n.z; ++k)
        for (int j = 0; j < n.y; ++j)
            for (int i = 0; i < n.x; ++i) {
                const float r = std::sqrt(float((i - 16) * (i - 16) + (j - 16) * (j - 16) + (k - 10) * (k - 10)));
                const std::size_t idx = std::size_t(g.index(i, j, k));
                if (r < 7.f) {
                    rho[idx] = 0.048f;
                    src[idx] = -0.05f;  // contracting while it cools
                }
            }
    Buffer<float> rho_d(dev, rho.size()), src_d(dev, src.size());
    rho_d.upload(rho.data(), rho.size());
    src_d.upload(src.data(), src.size());

    MacField vel(n, dev);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);
    auto randomize = [&](Buffer<float>& b) {
        std::vector<float> h(b.size());
        for (float& x : h) x = dist(rng);
        b.upload(h.data(), h.size());
    };
    randomize(vel.u);
    randomize(vel.v);
    randomize(vel.w);
    {  // impermeable ground
        std::vector<float> w = host(vel.w);
        for (int j = 0; j < n.y; ++j)
            for (int i = 0; i < n.x; ++i) w[std::size_t(i + n.x * j)] = 0.f;
        vel.w.upload(w.data(), w.size());
    }

    double err0 = 0.0;
    {
        const std::vector<float> d0 = divergence(n, dx, host(vel.u), host(vel.v), host(vel.w));
        for (std::size_t i = 0; i < d0.size(); ++i) err0 = std::max(err0, double(std::abs(d0[i] - src[i])));
    }

    PressureSolver ps(dev, n, dx);
    CHECK(ps.levels() >= 3);
    ps.set_density({rho_d.data(), n});
    const ProjectionStats st = ps.project(vel, {src_d.data(), n}, 1e-6f, 300);
    MESSAGE("MGPCG iterations: " << st.iterations << ", residual " << st.initial_residual << " -> " << st.final_residual);
    CHECK(st.iterations > 0);
    CHECK(st.iterations < 60);

    const std::vector<float> d = divergence(n, dx, host(vel.u), host(vel.v), host(vel.w));
    double err = 0.0;
    for (std::size_t i = 0; i < d.size(); ++i) err = std::max(err, double(std::abs(d[i] - src[i])));
    MESSAGE("max |div - S|: " << err0 << " -> " << err);
    CHECK(err < 1e-4 * err0);

    const std::vector<float> w = host(vel.w);
    for (int j = 0; j < n.y; ++j)
        for (int i = 0; i < n.x; ++i) REQUIRE(w[std::size_t(i + n.x * j)] == 0.f);

    // Projecting again barely changes the velocity.
    const std::vector<float> u1 = host(vel.u);
    ps.project(vel, {src_d.data(), n}, 1e-6f, 300);
    const std::vector<float> u2 = host(vel.u);
    float change = 0.f;
    for (std::size_t i = 0; i < u1.size(); ++i) change = std::max(change, std::abs(u2[i] - u1[i]));
    CHECK(change < 1e-3f);
}

TEST_CASE("MacCormack advection translates without new extrema") {
    const Device dev = test_device();
    const Int3 n{48, 16, 16};
    const float dx = 1.f;
    GridInfo g;
    g.dims = n;
    g.dx = dx;

    MacField vel(n, dev);
    {
        std::vector<float> u(vel.u.size(), 2.f);
        vel.u.upload(u.data(), u.size());
    }
    std::vector<float> blob(std::size_t(n.volume()));
    double mass0 = 0.0, cx0 = 0.0;
    for (int k = 0; k < n.z; ++k)
        for (int j = 0; j < n.y; ++j)
            for (int i = 0; i < n.x; ++i) {
                const float r2 = float((i - 12) * (i - 12) + (j - 8) * (j - 8) + (k - 8) * (k - 8));
                const float v = std::exp(-r2 / 8.f);
                blob[std::size_t(g.index(i, j, k))] = v;
                mass0 += v;
                cx0 += v * (i + 0.5);
            }
    cx0 /= mass0;
    Buffer<float> a(dev, blob.size()), b(dev, blob.size());
    a.upload(blob.data(), blob.size());
    AdvectionScratch scratch(n, dev);
    for (int s = 0; s < 6; ++s) {  // 6 steps of 1.5 s at 2 m/s = 18 cells
        advect_maccormack(dev, vel.sampler(), 1.5f, dx, SampleView{a.data(), n, kCellOffset}, {b.data(), n}, scratch);
        std::swap(a, b);
    }
    const std::vector<float> out = host(a);
    double mass = 0.0, cx = 0.0;
    float vmax = 0.f, vmin = 1.f;
    for (int k = 0; k < n.z; ++k)
        for (int j = 0; j < n.y; ++j)
            for (int i = 0; i < n.x; ++i) {
                const float v = out[std::size_t(g.index(i, j, k))];
                mass += v;
                cx += v * (i + 0.5);
                vmax = std::max(vmax, v);
                vmin = std::min(vmin, v);
            }
    cx /= mass;
    CHECK(cx - cx0 == doctest::Approx(18.0).epsilon(0.02));
    CHECK(vmax <= 1.f + 1e-5f);
    CHECK(vmin >= -1e-6f);
    CHECK(mass == doctest::Approx(mass0).epsilon(0.05));
}

TEST_CASE("atmosphere at rest stays at rest") {
    GridInfo g;
    g.dims = Int3{16, 16, 32};
    g.dx = 200.f;
    g.origin = Float3{-1600.f, -1600.f, 0.f};
    const Atmosphere atm{AtmosphereConfig{}};
    LowMachSolver solver(test_device(), g, atm, SolverParams{}, RadiationParams{});
    solver.reset_to_ambient(WindProfile{});
    const float t0 = solver.max_temperature();
    for (int s = 0; s < 5; ++s) solver.step(1.0);
    CHECK(solver.max_speed() < 1e-3f);
    CHECK(solver.max_temperature() == doctest::Approx(t0).epsilon(1e-5));
}

TEST_CASE("wind profile is carried without decay") {
    GridInfo g;
    g.dims = Int3{16, 16, 16};
    g.dx = 100.f;
    g.origin = Float3{-800.f, -800.f, 0.f};
    const Atmosphere atm{AtmosphereConfig{}};
    LowMachSolver solver(test_device(), g, atm, SolverParams{}, RadiationParams{});
    WindProfile wind;
    wind.heading_deg = 0.f;
    wind.points = {{0.f, 2.f}, {1600.f, 6.f}};
    solver.reset_to_ambient(wind);
    for (int s = 0; s < 10; ++s) solver.step(solver.suggest_dt());
    const std::vector<float> u = solver.velocity().u.to_host();
    const Int3 nu = u_dims(g.dims);
    const float expected = wind.speed_at(g.origin.z + 8.5f * g.dx);
    CHECK(u[std::size_t(8 + nu.x * (8 + std::int64_t(nu.y) * 8))] == doctest::Approx(expected).epsilon(0.02));
}

TEST_CASE("hot fireball rises and cools") {
    struct Variant {
        bool reflection;
        float smagorinsky;
    };
    for (const Variant var : {Variant{true, 0.f}, Variant{false, 0.f}, Variant{true, 0.17f}}) {
        CAPTURE(var.reflection);
        CAPTURE(var.smagorinsky);
        GridInfo g;
        g.dims = Int3{24, 24, 48};
        g.dx = 25.f;
        g.origin = Float3{-300.f, -300.f, 0.f};
        const Atmosphere atm{AtmosphereConfig{}};
        SolverParams params;
        params.reflection = var.reflection;
        params.smagorinsky = var.smagorinsky;
        LowMachSolver solver(test_device(), g, atm, params, RadiationParams{});
        solver.reset_to_ambient(WindProfile{});
        FireballConfig fb;
        fb.radius = 120.f;
        fb.temperature = 3000.f;
        init_fireball(solver, fb, 3);

        auto hot_centroid_z = [&]() {
            ScalarField d("d", g, solver.device()), du("du", g, solver.device()), t("t", g, solver.device()),
                f("f", g, solver.device()), vx("vx", g, solver.device()), vy("vy", g, solver.device()),
                vz("vz", g, solver.device());
            solver.fill_export(d.view(), du.view(), t.view(), f.view(), vx.view(), vy.view(), vz.view(), 6000.f);
            const std::vector<float> dens = d.buffer().to_host();
            double m = 0.0, z = 0.0;
            for (int k = 0; k < g.dims.z; ++k)
                for (int j = 0; j < g.dims.y; ++j)
                    for (int i = 0; i < g.dims.x; ++i) {
                        const double v = dens[std::size_t(g.index(i, j, k))];
                        m += v;
                        z += v * (k + 0.5) * g.dx;
                    }
            return m > 0.0 ? z / m : 0.0;
        };

        const double z0 = hot_centroid_z();
        const float t_start = solver.max_temperature();
        double t = 0.0;
        while (t < 10.0) {
            const double dt = std::min(solver.suggest_dt(), 10.0 - t);
            const StepStats s = solver.step(dt);
            REQUIRE(std::isfinite(solver.max_speed()));
            CHECK(s.pcg_iterations < 100 * s.projections);
            t += dt;
        }
        const double z1 = hot_centroid_z();
        MESSAGE("smoke centroid " << z0 << " m -> " << z1 << " m, Tmax " << t_start << " -> "
                                  << solver.max_temperature() << " K, |u|max " << solver.max_speed() << " m/s");
        CHECK(z1 > z0 + 150.0);
        CHECK(solver.max_temperature() < t_start);
        CHECK(solver.max_speed() > 20.f);
        CHECK(solver.max_speed() < 400.f);
    }
}

namespace {

// Dust-weighted mean height and total dust (sum of mixing ratios) of one bin.
void dust_moments(LowMachSolver& solver, int bin, double& total, double& mean_z) {
    const GridInfo& g = solver.grid();
    const FieldView v = solver.dust(bin);
    std::vector<float> h(std::size_t(g.count()));
    device_copy(h.data(), Device::CPU, v.data, solver.device(), h.size() * sizeof(float));
    total = 0.0;
    double z = 0.0;
    for (int k = 0; k < g.dims.z; ++k)
        for (int j = 0; j < g.dims.y; ++j)
            for (int i = 0; i < g.dims.x; ++i) {
                const double y = h[std::size_t(g.index(i, j, k))];
                total += y;
                z += y * (k + 0.5) * g.dx;
            }
    mean_z = total > 0.0 ? z / total : 0.0;
}

GridInfo dust_grid() {
    GridInfo g;
    g.dims = Int3{16, 16, 32};
    g.dx = 20.f;
    g.origin = Float3{-160.f, -160.f, 0.f};
    return g;
}

}  // namespace

TEST_CASE("dust settles at its fall speed and deposits on the ground") {
    const GridInfo g = dust_grid();
    const Atmosphere atm{AtmosphereConfig{}};
    DustParams dp;
    dp.erosion = 0.f;
    LowMachSolver solver(test_device(), g, atm, SolverParams{}, RadiationParams{}, dp);
    solver.reset_to_ambient(WindProfile{});
    FireballConfig no_fireball;
    DustInitConfig init;
    init.fireball_loading = 0.f;
    init.skirt_radius = 1e6f;  // a uniform layer over the whole domain
    init.skirt_height = 300.f;
    init.skirt_loading = 0.001f;  // light enough not to drive a density current
    init_dust(solver, no_fireball, init, 1);

    double total0[kDustBins], z0[kDustBins];
    for (int b = 0; b < kDustBins; ++b) dust_moments(solver, b, total0[b], z0[b]);
    const double t_end = 20.0;
    for (double t = 0.0; t < t_end - 1e-9; t += 0.5) solver.step(0.5);
    double total[kDustBins], z[kDustBins];
    for (int b = 0; b < kDustBins; ++b) dust_moments(solver, b, total[b], z[b]);
    MESSAGE("fine: " << z0[0] << " -> " << z[0] << " m, coarse: " << z0[2] << " -> " << z[2] << " m, coarse left "
                     << total[2] / total0[2]);
    // Fine dust (2 cm/s) barely moves; coarse dust (2.5 m/s) falls ~50 m and partly lands.
    CHECK(std::abs(z[0] - z0[0]) < 5.0);
    CHECK(z0[2] - z[2] > 15.0);
    CHECK(total[2] < 0.95 * total0[2]);
    CHECK(total[0] == doctest::Approx(total0[0]).epsilon(0.03));
}

TEST_CASE("strong wind lifts sand from the ground, a light breeze does not") {
    const GridInfo g = dust_grid();
    const Atmosphere atm{AtmosphereConfig{}};
    for (const float speed : {2.f, 60.f}) {
        CAPTURE(speed);
        LowMachSolver solver(test_device(), g, atm, SolverParams{}, RadiationParams{}, DustParams{});
        WindProfile wind;
        wind.points = {{0.f, speed}, {1000.f, speed}};
        solver.reset_to_ambient(wind);
        for (int s = 0; s < 10; ++s) solver.step(0.25);
        double total = 0.0, mean = 0.0;
        dust_moments(solver, 0, total, mean);
        if (speed < 10.f) CHECK(total == 0.0);
        else CHECK(total > 1e-4);
    }
}

TEST_CASE("dust loading weighs the rising fireball down") {
    double rise[2];
    for (int heavy = 0; heavy < 2; ++heavy) {
        const GridInfo g = dust_grid();
        const Atmosphere atm{AtmosphereConfig{}};
        LowMachSolver solver(test_device(), g, atm, SolverParams{}, RadiationParams{}, DustParams{});
        solver.reset_to_ambient(WindProfile{});
        FireballConfig fb;
        fb.radius = 80.f;
        fb.temperature = 1200.f;
        fb.noise_amplitude = 0.f;
        init_fireball(solver, fb, 5);
        DustInitConfig init;
        init.fireball_loading = heavy ? 2.0f : 0.f;  // more dust than air by mass
        init.fireball_split[0] = 1.f;
        init.fireball_split[1] = init.fireball_split[2] = 0.f;
        init.skirt_loading = 0.f;
        init_dust(solver, fb, init, 5);
        for (int s = 0; s < 20; ++s) solver.step(0.25);
        rise[heavy] = solver.diagnostics().centroid_z;
    }
    MESSAGE("smoke centroid after 5 s: clean " << rise[0] << " m, dusty " << rise[1] << " m");
    CHECK(rise[1] < rise[0] - 10.0);
}

TEST_CASE("growing the domain keeps the cloud and doubles the extent") {
    GridInfo g;
    g.dims = Int3{32, 32, 32};
    g.dx = 10.f;
    g.origin = Float3{-160.f, -160.f, 0.f};
    const Atmosphere atm{AtmosphereConfig{}};
    LowMachSolver solver(test_device(), g, atm, SolverParams{}, RadiationParams{}, DustParams{});
    WindProfile wind;
    wind.points = {{0.f, 3.f}, {1000.f, 3.f}};
    solver.reset_to_ambient(wind);
    FireballConfig fb;
    fb.radius = 60.f;
    fb.temperature = 1500.f;
    init_fireball(solver, fb, 2);
    for (int s = 0; s < 4; ++s) solver.step(0.2);

    auto smoke_mass = [&]() {
        const GridInfo& gg = solver.grid();
        std::vector<float> h(std::size_t(gg.count()));
        device_copy(h.data(), Device::CPU, solver.smoke().data, solver.device(), h.size() * sizeof(float));
        double m = 0.0;
        for (float v : h) m += v;
        return m * double(gg.dx) * gg.dx * gg.dx;
    };
    const double m0 = smoke_mass();
    const CloudExtent e0 = solver.cloud_extent(0.02f);
    solver.coarsen_domain();
    const GridInfo& ng = solver.grid();
    CHECK(ng.dx == doctest::Approx(20.f));
    CHECK(ng.origin.x == doctest::Approx(-320.f));
    CHECK(ng.origin.z == doctest::Approx(0.f));
    CHECK(smoke_mass() == doctest::Approx(m0).epsilon(1e-4));
    const CloudExtent e1 = solver.cloud_extent(0.02f);
    CHECK(e1.top >= e0.top - 20.f);
    CHECK(std::isfinite(solver.max_speed()));
    CHECK(solver.max_speed() < 200.f);
    // The new outer region carries the ambient wind.
    const std::vector<float> u = solver.velocity().u.to_host();
    const Int3 nu = u_dims(ng.dims);
    CHECK(u[std::size_t(1 + nu.x * (1 + std::int64_t(nu.y) * 20))] == doctest::Approx(3.f).epsilon(0.05));
    // And it keeps running.
    for (int s = 0; s < 4; ++s) solver.step(solver.suggest_dt());
    CHECK(std::isfinite(solver.max_speed()));
}
