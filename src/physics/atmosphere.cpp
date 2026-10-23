#include "physics/atmosphere.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "physics/constants.h"

namespace sim {

namespace {
constexpr double kStratopauseAltitude = 20000.0;  // m, isothermal layer top
constexpr double kStratosphereLapse = -0.001;      // K/m (temperature rises above 20 km)
constexpr double kStandardSeaLevelTemperature = 288.15;
}  // namespace

Atmosphere::Atmosphere(const AtmosphereConfig& cfg) : cfg_(cfg) {
    if (cfg.ground_temperature <= 100.f || cfg.lapse_rate < 0.f || cfg.sea_level_pressure <= 0.f) {
        throw std::invalid_argument("atmosphere: invalid configuration");
    }
    const double g = phys::kGravity, R = phys::kGasConstantAir;
    // Ground pressure from the standard atmosphere at that altitude; the local
    // (hotter) temperature then shapes the profile above the ground.
    const double L0 = 0.0065;
    ground_pressure_ = cfg.sea_level_pressure *
                       std::pow(1.0 - L0 * cfg.ground_altitude / kStandardSeaLevelTemperature, g / (R * L0));
}

double Atmosphere::temperature_at_altitude(double h) const {
    const double hg = cfg_.ground_altitude;
    const double ht = std::max<double>(cfg_.tropopause_altitude, hg);
    if (h <= ht) return cfg_.ground_temperature - cfg_.lapse_rate * (h - hg);
    const double tt = cfg_.ground_temperature - cfg_.lapse_rate * (ht - hg);
    if (h <= kStratopauseAltitude) return tt;
    return tt - kStratosphereLapse * (h - kStratopauseAltitude);
}

double Atmosphere::pressure_at_altitude(double h) const {
    const double g = phys::kGravity, R = phys::kGasConstantAir;
    const double hg = cfg_.ground_altitude;
    const double ht = std::max<double>(cfg_.tropopause_altitude, hg);
    const double tg = cfg_.ground_temperature;
    const double L = cfg_.lapse_rate;

    auto lapse_layer = [&](double p0, double t0, double lapse, double dh) {
        if (std::abs(lapse) < 1e-9) return p0 * std::exp(-g * dh / (R * t0));
        return p0 * std::pow((t0 - lapse * dh) / t0, g / (R * lapse));
    };

    if (h <= ht) return lapse_layer(ground_pressure_, tg, L, h - hg);
    const double tt = tg - L * (ht - hg);
    const double pt = lapse_layer(ground_pressure_, tg, L, ht - hg);
    if (h <= kStratopauseAltitude) return pt * std::exp(-g * (h - ht) / (R * tt));
    const double ps = pt * std::exp(-g * (kStratopauseAltitude - ht) / (R * tt));
    return lapse_layer(ps, tt, kStratosphereLapse, h - kStratopauseAltitude);
}

AtmosphereState Atmosphere::at(double z) const {
    const double h = cfg_.ground_altitude + std::max(z, 0.0);
    AtmosphereState s;
    s.temperature = temperature_at_altitude(h);
    s.pressure = pressure_at_altitude(h);
    s.density = s.pressure / (phys::kGasConstantAir * s.temperature);
    s.exner = std::pow(s.pressure / phys::kReferencePressure, double(phys::kKappa));
    s.theta = s.temperature / s.exner;
    return s;
}

AtmosphereColumns::AtmosphereColumns(const Atmosphere& atm, const GridInfo& grid, Device dev)
    : temperature(dev, std::size_t(grid.dims.z)),
      pressure(dev, std::size_t(grid.dims.z)),
      density(dev, std::size_t(grid.dims.z)),
      theta(dev, std::size_t(grid.dims.z)),
      exner(dev, std::size_t(grid.dims.z)) {
    const std::size_t nz = std::size_t(grid.dims.z);
    std::vector<float> t(nz), p(nz), rho(nz), th(nz), ex(nz);
    for (std::size_t k = 0; k < nz; ++k) {
        const AtmosphereState s = atm.at(grid.origin.z + (double(k) + 0.5) * grid.dx);
        t[k] = float(s.temperature);
        p[k] = float(s.pressure);
        rho[k] = float(s.density);
        th[k] = float(s.theta);
        ex[k] = float(s.exner);
    }
    temperature.upload(t.data(), nz);
    pressure.upload(p.data(), nz);
    density.upload(rho.data(), nz);
    theta.upload(th.data(), nz);
    exner.upload(ex.data(), nz);
    ground_density = float(atm.at(0.0).density);
}

}  // namespace sim
