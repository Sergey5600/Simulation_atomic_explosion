#pragma once
// Low-Mach-number, variable-density solver for the buoyant phase of the
// explosion (from about one second after detonation, once the shock has left
// the fireball and the pressure is close to ambient).
//
// State: potential temperature theta (K), smoke mass fraction (0..1) and the
// staggered velocity. Density follows from the ideal gas law at the ambient
// hydrostatic pressure p0(z): rho = p0 / (R T), T = theta * exner(z). Unlike the
// Boussinesq approximation this stays valid for gas 10-30 times lighter than air.
//
//   Du/Dt = -(1/rho) grad p' + g (theta - theta_a) / theta_a z  (+ vorticity confinement)
//   div u = (D ln T / Dt)_radiation + w g / (gamma R T_a)        (heat loss, expansion with height)
//   Dtheta/Dt = radiative cooling only (adiabatic motion conserves theta)

#include <memory>

#include "core/grid.h"
#include "lowmach/advection.h"
#include "lowmach/mac.h"
#include "lowmach/pressure.h"
#include "physics/atmosphere.h"
#include "physics/wind.h"

namespace sim {

struct RadiationParams {
    float absorption = 3e-4f;             // 1/m, Planck-mean absorption of the hot gas at reference_temperature
    float reference_temperature = 3000.f; // K
    float exponent = 2.f;                 // absorption ~ (T / T_ref)^exponent
};

struct SolverParams {
    float cfl = 1.f;
    float max_dt = 0.1f;               // s
    bool reflection = true;            // advection-reflection (Zehnder et al. 2018): keeps vortices alive
    float vorticity_confinement = 0.15f;
    float pressure_tolerance = 1e-4f;  // relative residual
    int pressure_max_iterations = 200;
};

struct CloudDiagnostics {
    float top = 0.f;         // m, highest cell with smoke > 0.05
    float centroid_z = 0.f;  // m, smoke-weighted mean height
    float max_speed = 0.f;   // m/s
    float max_speed_z = 0.f; // m, height of the fastest cell
};

struct StepStats {
    double dt = 0.0;
    int projections = 0;
    int pcg_iterations = 0;
    double max_residual = 0.0;  // 1/s, RMS divergence error after projection
};

class LowMachSolver {
public:
    LowMachSolver(Device dev, const GridInfo& grid, const Atmosphere& atm, const SolverParams& params,
                  const RadiationParams& radiation);

    // Ambient atmosphere at rest plus the wind profile; no smoke.
    void reset_to_ambient(const WindProfile& wind);

    // Largest stable time step for the current state (capped by params.max_dt).
    double suggest_dt();
    StepStats step(double dt);

    // Cell-centred export fields: density = smoke mass concentration relative to ground air,
    // temperature in K (0 where ambient), flame = (T / flame_temperature)^4 above ambient.
    void fill_export(FieldView density, FieldView temperature, FieldView flame, FieldView vel_x, FieldView vel_y,
                     FieldView vel_z, float flame_temperature);

    float max_temperature();
    float max_speed();
    CloudDiagnostics diagnostics();

    const GridInfo& grid() const { return grid_; }
    Device device() const { return dev_; }
    FieldView theta() { return {theta_.data(), grid_.dims}; }
    FieldView smoke() { return {smoke_.data(), grid_.dims}; }
    MacField& velocity() { return vel_; }
    const AtmosphereColumns& ambient() const { return ambient_; }
    std::size_t memory_bytes() const;

    // Step stages. Public only because nvcc does not allow device lambdas inside
    // private member functions.
    void cool_and_build_source(float dt);
    void update_density();
    void add_forces(MacField& field, float dt);
    void advect_scalars(const MacView& vel, float dt);
    ProjectionStats project(MacField& field);

private:
    Device dev_;
    GridInfo grid_;
    SolverParams params_;
    RadiationParams radiation_;
    AtmosphereColumns ambient_;

    Buffer<float> theta_, smoke_, scalar_tmp_, rho_, source_;
    Buffer<float> omega_x_, omega_y_, omega_z_;
    MacField vel_, vel_a_, vel_b_;
    AdvectionScratch scratch_;
    PressureSolver pressure_;
};

}  // namespace sim
