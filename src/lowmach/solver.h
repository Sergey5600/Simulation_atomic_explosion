#pragma once
// Low-Mach-number, variable-density solver for the buoyant phase of the
// explosion (from about one second after detonation, once the shock has left
// the fireball and the pressure is close to ambient).
//
// State: potential temperature theta (K), smoke mass fraction (0..1), dust mass
// mixing ratios (kg dust per kg air, one per size bin) and the staggered
// velocity. Gas density follows from the ideal gas law at the ambient
// hydrostatic pressure p0(z): rho = p0 / (R T), T = theta * exner(z); dust adds
// its mass on top. Unlike the Boussinesq approximation this stays valid for gas
// 10-30 times lighter than air.
//
//   Du/Dt = -(1/rho) grad p' + g ((theta / theta_a) / (1 + dust) - 1) z  (+ confinement, turbulence)
//   div u = (D ln T / Dt)_radiation + w g / (gamma R T_a)                (heat loss, expansion with height)
//   Dtheta/Dt = radiative cooling only (adiabatic motion conserves theta)
//   dust: carried by the flow, falls at its settling speed, lifted from the
//         ground where the near-surface wind exceeds the erosion threshold

#include <memory>
#include <vector>

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
    float smagorinsky = 0.15f;         // subgrid turbulence constant C_s (0 = off)
    float turbulent_prandtl = 0.7f;    // momentum / heat eddy diffusivity ratio
    float pressure_tolerance = 1e-4f;  // relative residual
    int pressure_max_iterations = 200;
};

// Dust size bins: fine (stays aloft, colours the cap), medium, coarse (falls back in curtains).
constexpr int kDustBins = 3;

struct DustParams {
    bool enabled = true;
    float settling[kDustBins] = {0.02f, 0.4f, 2.5f};       // m/s, terminal fall speed
    float optical_weight[kDustBins] = {1.f, 0.35f, 0.1f};  // extinction per unit mass relative to fine dust
    float erosion = 0.03f;                     // dimensionless vertical flux coefficient
    float threshold_friction_velocity = 0.3f;  // m/s, loose desert sand
    float roughness_length = 0.001f;           // m
    float erosion_split[kDustBins] = {0.3f, 0.5f, 0.2f};
    bool mass_loading = true;                  // dust weighs the air down
    float render_scale = 100.f;                // exported dust = scale * optical mass concentration / ground air density
};

struct CloudDiagnostics {
    float top = 0.f;         // m, highest cell with smoke > 0.05
    float centroid_z = 0.f;  // m, smoke-weighted mean height
    float max_speed = 0.f;   // m/s
    float max_speed_z = 0.f; // m, height of the fastest cell
    double dust_tonnes = 0.0;
};

// Extent of the visible cloud (smoke above a threshold), for growing the domain.
struct CloudExtent {
    float top = 0.f;     // m above the domain bottom
    float radius = 0.f;  // m, largest horizontal distance from the domain centre
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
                  const RadiationParams& radiation, const DustParams& dust = DustParams{});

    // Ambient atmosphere at rest plus the wind profile; no smoke or dust.
    void reset_to_ambient(const WindProfile& wind);

    // Doubles the voxel size and the domain (same cell counts, centred horizontally, still
    // resting on the ground). The old domain is averaged into the middle; the new outer
    // region starts as ambient air with the wind. Needs x, y divisible by 4 and z by 2.
    void coarsen_domain();
    static bool can_coarsen(Int3 dims) { return dims.x % 4 == 0 && dims.y % 4 == 0 && dims.z % 2 == 0; }
    CloudExtent cloud_extent(float smoke_threshold);

    // Replaces the state with host data on this grid (cell centred), e.g. from the
    // compressible stage: temperature in K, mass fractions, cell velocities. The
    // velocity is moved to the faces and projected.
    void load_cell_state(const std::vector<float>& temperature, const std::vector<float>& smoke,
                         const std::vector<float>* dust_bins, const std::vector<float>& u,
                         const std::vector<float>& v, const std::vector<float>& w);

    // Largest stable time step for the current state (capped by params.max_dt).
    double suggest_dt();
    StepStats step(double dt);

    // Cell-centred export fields: density = smoke mass concentration relative to ground air,
    // dust = optical dust concentration (see DustParams::render_scale), temperature in K
    // (0 where ambient), flame = (T / flame_temperature)^4 above ambient.
    void fill_export(FieldView density, FieldView dust, FieldView temperature, FieldView flame, FieldView vel_x,
                     FieldView vel_y, FieldView vel_z, float flame_temperature);

    float max_temperature();
    float max_speed();
    CloudDiagnostics diagnostics();

    const GridInfo& grid() const { return grid_; }
    Device device() const { return dev_; }
    FieldView theta() { return scalar(kTheta); }
    FieldView smoke() { return scalar(kSmoke); }
    bool has_dust() const { return dust_.enabled; }
    FieldView dust(int bin) { return scalar(kDust0 + bin); }
    MacField& velocity() { return vel_; }
    const AtmosphereColumns& ambient() const { return ambient_; }
    const DustParams& dust_params() const { return dust_; }
    std::size_t memory_bytes() const;

    // Step stages. Public only because nvcc does not allow device lambdas inside
    // private member functions.
    void cool_and_build_source(float dt);
    void update_density();
    void add_forces(MacField& field, float dt);
    void advect_scalars(const MacView& vel, float dt);
    void apply_turbulence(MacField& field, float dt);
    void settle_and_lift_dust(const MacField& field, float dt);
    ProjectionStats project(MacField& field);

    // Pointers to the dust bins (null when dust is off) for kernels.
    struct DustPointers {
        const float* bin[kDustBins] = {};
        int count = 0;
    };
    DustPointers dust_pointers();

private:
    static constexpr int kTheta = 0, kSmoke = 1, kDust0 = 2;
    FieldView scalar(int i) { return {scalars_[std::size_t(i)].data(), grid_.dims}; }

    Device dev_;
    GridInfo grid_;
    SolverParams params_;
    RadiationParams radiation_;
    DustParams dust_;
    Atmosphere atmosphere_;
    WindProfile wind_;
    AtmosphereColumns ambient_;
    Buffer<float> wind_u_, wind_v_;  // ambient wind per level, for the current grid

    // Transported cell fields (theta, smoke, dust bins...), their next-step buffers and MacCormack scratch.
    std::vector<Buffer<float>> scalars_, scalars_next_, scalars_bwd_;
    Buffer<float> rho_, source_;
    Buffer<float> omega_x_, omega_y_, omega_z_;
    MacField vel_, vel_a_, vel_b_;
    AdvectionScratch scratch_;
    PressureSolver pressure_;
};

}  // namespace sim
