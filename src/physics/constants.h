#pragma once
// Physical constants (SI units).

namespace sim::phys {

constexpr float kGravity = 9.80665f;          // m/s^2
constexpr float kGasConstantAir = 287.05f;    // J/(kg K), dry air
constexpr float kHeatCapacityAir = 1004.7f;   // J/(kg K), c_p of dry air
constexpr float kGamma = 1.4f;                // c_p / c_v
constexpr float kKappa = kGasConstantAir / kHeatCapacityAir;  // R / c_p
constexpr float kStefanBoltzmann = 5.670374e-8f;  // W/(m^2 K^4)
constexpr float kReferencePressure = 100000.f;    // Pa, for potential temperature

}  // namespace sim::phys
