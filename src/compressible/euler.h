#pragma once
// Stage A: compressible gas dynamics for the first second of a surface burst.
//
// 3D Euler equations with gravity on a uniform grid, finite volumes, MUSCL
// reconstruction (MC limiter) and the HLLC Riemann solver, SSP-RK2 in time.
// The hydrostatic atmosphere is subtracted in the reconstruction and in the
// momentum flux, so the ambient air stays exactly at rest (well-balanced).
//
// Boundaries: the ground (z = 0) reflects, the sides and the top let waves out.
// Transported scalars: the fireball-gas marker ("smoke": air that has been
// hotter than ~2000 K, i.e. the luminous fireball and its NO2) and the dust
// bins, lifted from the ground by the blast wind.

#include <vector>

#include "core/grid.h"
#include "lowmach/solver.h"
#include "physics/atmosphere.h"

namespace sim {

struct BlastParams {
    float start_time = 0.0005f;        // s after detonation at which the simulation starts
    float yield_kt = 20.f;
    float hydro_fraction = 0.55f;      // share of the yield driving the blast (the rest leaves as radiation)
    float deposit_radius = 30.f;       // m, hot hemisphere the energy starts in
    float noise_amplitude = 0.08f;     // relative radius perturbation (seeds the fireball surface instabilities)
    float noise_frequency = 0.08f;     // 1/m
    float handoff_time = 1.0f;         // s, switch to the low-Mach solver
    float gamma = 1.4f;
    float cfl = 0.3f;
    float gravity = 9.80665f;
    bool well_balanced = true;
    // Radiative loss of the fireball, emitted from its surface: q = 3 sigma min(T, T_cap)^4 / L.
    float radiation_temperature_cap = 8000.f;  // K, hottest apparent surface temperature
    float radiation_length = 150.f;            // m, about the fireball radius
    float fireball_temperature = 2000.f;       // K, air hotter than this becomes "fireball gas"
    float fireball_dust_loading = 0.02f;       // kg dust / kg air in the initial hemisphere
    float fireball_dust_split[kDustBins] = {0.5f, 0.3f, 0.2f};
    float blast_erosion = 3e-4f;               // dust lifting coefficient under the blast wind
};

class CompressibleSolver {
public:
    static constexpr int kScalars = 1 + kDustBins;  // smoke + dust bins
    static constexpr int kVars = 5 + kScalars;      // rho, rho u, rho v, rho w, E, rho Y...

    CompressibleSolver(Device dev, const GridInfo& grid, const Atmosphere& atm, const BlastParams& params,
                       const DustParams& dust);

    // Ambient atmosphere at rest (no wind: it is irrelevant within one second).
    void reset_to_ambient();
    // Puts the explosion energy into a hemisphere at ground zero (plus soil dust in it).
    void deposit_energy(std::uint32_t seed);
    // Uniform state everywhere (tests).
    void set_uniform(float rho, float p);

    double suggest_dt();
    void step(double dt);

    // Same fields as LowMachSolver::fill_export.
    void fill_export(FieldView density, FieldView dust, FieldView temperature, FieldView flame, FieldView vel_x,
                     FieldView vel_y, FieldView vel_z, float flame_temperature);

    // State for the handoff to the low-Mach solver, on the host (cell centred).
    struct HandoffState {
        std::vector<float> temperature, smoke, u, v, w;
        std::vector<float> dust[kDustBins];
    };
    HandoffState handoff_state();

    float max_temperature();
    float max_speed();
    float max_pressure_ratio();  // max p / p_ambient, shows the shock strength
    double total_energy();       // J, sum of E over the domain (tests)
    float shock_radius();        // m, furthest ground-level cell with p > 1.05 p_ambient

    const GridInfo& grid() const { return grid_; }
    Device device() const { return dev_; }
    std::size_t memory_bytes() const;

    // Raw arrays for tests and kernels (rho, mx, my, mz, E, rhoY...).
    FieldView var(int v) { return {u_[std::size_t(v)].data(), grid_.dims}; }

    // Stages. Public only because nvcc does not allow device lambdas inside private member functions.
    void compute_rhs(int src_set);
    void apply_sources_after_step(float dt);

private:
    Device dev_;
    GridInfo grid_;
    BlastParams params_;
    DustParams dust_;
    AtmosphereColumns ambient_;
    Buffer<float> rho0_c_, p0_c_, rho0_f_, p0_f_;  // hydrostatic background at cell centres / z-faces

public:
    // Conserved variables (two sets for the RK stages), right-hand side and one direction of fluxes.
    std::vector<Buffer<float>> u_, u1_, rhs_, flux_;
};

}  // namespace sim
