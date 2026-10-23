#pragma once
// Background atmosphere: a US Standard Atmosphere 1976 style layered profile
// with an adjustable ground altitude and ground temperature (hot desert).

#include <vector>

#include "core/grid.h"

namespace sim {

struct AtmosphereConfig {
    float ground_altitude = 1200.f;       // m above sea level (Nevada desert)
    float ground_temperature = 303.15f;   // K at ground level
    float lapse_rate = 0.0065f;           // K/m in the troposphere
    float tropopause_altitude = 11000.f;  // m above sea level
    float sea_level_pressure = 101325.f;  // Pa
};

struct AtmosphereState {
    double temperature = 0.0;  // K
    double pressure = 0.0;     // Pa
    double density = 0.0;      // kg/m^3
    double theta = 0.0;        // potential temperature, K
    double exner = 0.0;        // (p / p_ref)^(R/c_p), so T = theta * exner
};

class Atmosphere {
public:
    explicit Atmosphere(const AtmosphereConfig& cfg);

    // z: height above the ground in metres.
    AtmosphereState at(double z) const;
    const AtmosphereConfig& config() const { return cfg_; }

private:
    double temperature_at_altitude(double h) const;
    double pressure_at_altitude(double h) const;

    AtmosphereConfig cfg_;
    double ground_pressure_ = 0.0;
};

// Per-level profiles sampled at the cell centres of a grid (k = 0 .. nz-1),
// uploaded to the compute device for use in kernels.
struct AtmosphereColumns {
    Buffer<float> temperature;
    Buffer<float> pressure;
    Buffer<float> density;
    Buffer<float> theta;
    Buffer<float> exner;
    float ground_density = 1.f;

    AtmosphereColumns(const Atmosphere& atm, const GridInfo& grid, Device dev);
};

}  // namespace sim
