#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "compressible/euler.h"
#include "core/backend.h"
#include "core/memory.h"
#include "io/config.h"
#include "scenes/scene.h"

using namespace sim;

namespace {

Device test_device() { return cuda_available() ? Device::CUDA : Device::CPU; }

std::vector<float> download(CompressibleSolver& s, int v) {
    const GridInfo& g = s.grid();
    std::vector<float> h(std::size_t(g.count()));
    device_copy(h.data(), Device::CPU, s.var(v).data, s.device(), h.size() * sizeof(float));
    return h;
}

BlastParams quiet_params() {
    BlastParams p;
    p.gravity = 0.f;
    p.well_balanced = false;
    p.radiation_length = 1e12f;  // no radiative loss
    p.blast_erosion = 0.f;
    return p;
}

}  // namespace

TEST_CASE("Sod shock tube matches the exact solution") {
    GridInfo g;
    g.dims = Int3{200, 4, 4};
    g.dx = 0.005f;  // 1 m tube
    const Atmosphere atm{AtmosphereConfig{}};
    CompressibleSolver s(test_device(), g, atm, quiet_params(), DustParams{});
    s.set_uniform(1.f, 1e5f);

    // Left: rho 1, p 1e5; right: rho 0.125, p 1e4 (the classic Sod problem scaled to SI).
    std::vector<float> rho(std::size_t(g.count())), e(std::size_t(g.count()));
    for (int k = 0; k < g.dims.z; ++k)
        for (int j = 0; j < g.dims.y; ++j)
            for (int i = 0; i < g.dims.x; ++i) {
                const bool left = i < g.dims.x / 2;
                rho[std::size_t(g.index(i, j, k))] = left ? 1.f : 0.125f;
                e[std::size_t(g.index(i, j, k))] = (left ? 1e5f : 1e4f) / 0.4f;
            }
    s.u_[0].upload(rho.data(), rho.size());
    s.u_[4].upload(e.data(), e.size());

    // Non-dimensional time 0.2 = t * sqrt(p_ref / rho_ref) / L.
    const double t_end = 0.2 / std::sqrt(1e5);
    double t = 0.0;
    while (t < t_end - 1e-12) {
        const double dt = std::min(s.suggest_dt(), t_end - t);
        s.step(dt);
        t += dt;
    }
    const std::vector<float> r = download(s, 0);
    auto at = [&](double x) { return double(r[std::size_t(g.index(int(x / g.dx), 1, 1))]); };
    // Exact: rarefaction fan 0.263-0.486, contact 0.685, shock 0.850 (x measured from the left end).
    CHECK(at(0.10) == doctest::Approx(1.0).epsilon(0.02));
    CHECK(at(0.60) == doctest::Approx(0.42632).epsilon(0.05));
    CHECK(at(0.77) == doctest::Approx(0.26557).epsilon(0.05));
    CHECK(at(0.92) == doctest::Approx(0.125).epsilon(0.02));
}

TEST_CASE("compressible atmosphere at rest stays at rest") {
    GridInfo g;
    g.dims = Int3{8, 8, 64};
    g.dx = 20.f;
    g.origin = Float3{-80.f, -80.f, 0.f};
    const Atmosphere atm{AtmosphereConfig{}};
    BlastParams p;
    p.blast_erosion = 0.f;
    CompressibleSolver s(test_device(), g, atm, p, DustParams{});
    s.reset_to_ambient();
    for (int i = 0; i < 50; ++i) s.step(s.suggest_dt());
    CHECK(s.max_speed() < 1e-3f);
    CHECK(s.max_pressure_ratio() == doctest::Approx(1.0).epsilon(1e-5));
}

TEST_CASE("blast wave radius follows Sedov-Taylor and energy is conserved") {
    GridInfo g;
    g.dims = Int3{64, 64, 32};
    g.dx = 6.f;
    g.origin = Float3{-192.f, -192.f, 0.f};
    const Atmosphere atm{AtmosphereConfig{}};
    BlastParams p = quiet_params();
    p.yield_kt = 1.f;
    p.hydro_fraction = 1.f;
    p.deposit_radius = 20.f;
    p.noise_amplitude = 0.f;
    CompressibleSolver s(test_device(), g, atm, p, DustParams{});
    const double rho = 1.0, p0 = 87715.0;
    s.set_uniform(float(rho), float(p0));  // no gravity here, so no stratification either
    const double e0 = s.total_energy();
    s.deposit_energy(1);
    const double deposited = s.total_energy() - e0;
    CHECK(deposited == doctest::Approx(4.184e12).epsilon(1e-3));

    // Early enough for a strong shock (overpressure ~10x ambient) that is still inside the domain.
    const double t_end = 0.05;
    double t = 0.0;
    while (t < t_end - 1e-12) {
        const double dt = std::min(s.suggest_dt(), t_end - t);
        s.step(dt);
        t += dt;
    }
    // A hemisphere on a reflecting ground behaves like a sphere with twice the energy.
    const double sedov = 1.15 * std::pow(2.0 * deposited * t_end * t_end / rho, 0.2);
    const double r = s.shock_radius();
    MESSAGE("shock radius " << r << " m, Sedov-Taylor " << sedov << " m");
    CHECK(r == doctest::Approx(sedov).epsilon(0.15));
    // Nothing has left the domain yet, so the total energy is unchanged.
    CHECK(s.total_energy() - e0 == doctest::Approx(deposited).epsilon(0.02));
    CHECK(s.max_temperature() > 1000.f);
}

TEST_CASE("surface burst scene hands off to the buoyant phase") {
    SceneConfig cfg;
    cfg.type = "surface_burst";
    cfg.device = cuda_available() ? "cuda" : "cpu";
    cfg.grid.dims = Int3{32, 32, 32};
    cfg.grid.dx = 10.f;
    cfg.grid.origin = Float3{-160.f, -160.f, 0.f};
    cfg.blast.yield_kt = 0.02f;
    cfg.blast.deposit_radius = 25.f;
    cfg.blast.start_time = 0.001f;
    cfg.blast.handoff_time = 0.3f;
    cfg.wind.points = {{0.f, 2.f}, {1000.f, 4.f}};
    auto scene = make_scene(cfg, parse_device(cfg.device));
    ExportBuffers out(cfg.grid, parse_device(cfg.device));

    scene->advance_to(0.1);
    scene->fill_export(out);
    CHECK(scene->status().rfind("[blast]", 0) == 0);
    {
        const std::vector<float> f = out.flame.buffer().to_host();
        float fmax = 0.f;
        bool finite = true;
        for (float v : f) {
            finite = finite && std::isfinite(v);
            fmax = std::max(fmax, v);
        }
        CHECK(finite);
        CHECK(fmax > 0.f);  // the fireball glows
    }

    scene->advance_to(0.6);
    scene->fill_export(out);
    CHECK(scene->status().rfind("[rise]", 0) == 0);
    const std::vector<float> t = out.temperature.buffer().to_host();
    float tmax = 0.f;
    bool finite = true;
    for (float v : t) {
        finite = finite && std::isfinite(v);
        tmax = std::max(tmax, v);
    }
    CHECK(finite);
    CHECK(tmax > 400.f);
}

TEST_CASE("the blast's negative phase condenses humid air (Wilson cloud)") {
    GridInfo g;
    g.dims = Int3{48, 48, 24};
    g.dx = 8.f;
    g.origin = Float3{-192.f, -192.f, 0.f};
    const Atmosphere atm{AtmosphereConfig{}};
    BlastParams p;
    p.yield_kt = 0.3f;
    p.deposit_radius = 20.f;
    p.blast_erosion = 0.f;
    HumidityProfile h;
    h.enabled = true;
    h.points = {{0.f, 0.97f}, {1000.f, 0.97f}};
    CompressibleSolver s(test_device(), g, atm, p, DustParams{}, h);
    s.reset_to_ambient();
    s.deposit_energy(3);
    double peak = 0.0;
    for (double t = 0.0; t < 0.25;) {
        const double dt = s.suggest_dt();
        s.step(dt);
        t += dt;
        const std::vector<float> qc = download(s, CompressibleSolver::kVapor + 1);
        double sum = 0.0;
        for (float v : qc) sum += v;
        peak = std::max(peak, sum);
    }
    MESSAGE("peak cloud water (sum of rho qc): " << peak);
    CHECK(peak > 1e-4);
}
