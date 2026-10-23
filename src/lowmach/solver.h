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
//   moisture (optional): vapour and cloud water, saturation adjustment with latent heat;
//         vapour lightens the air (virtual temperature), cloud water weighs it down

#include <memory>
#include <vector>

#include "core/export_field.h"
#include "core/grid.h"
#include "core/profiler.h"
#include "lowmach/advection.h"
#include "lowmach/mac.h"
#include "lowmach/pressure.h"
#include "physics/atmosphere.h"
#include "physics/humidity.h"
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
                  const RadiationParams& radiation, const DustParams& dust = DustParams{},
                  const HumidityProfile& humidity = HumidityProfile{});

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
                         const std::vector<float>& v, const std::vector<float>& w,
                         const std::vector<float>* vapor = nullptr, const std::vector<float>* condensate = nullptr);

    // Largest stable time step for the current state (capped by params.max_dt).
    double suggest_dt();
    StepStats step(double dt);

    // Cell-centred export fields: density = smoke mass concentration relative to ground air,
    // dust = optical dust concentration (see DustParams::render_scale), temperature in K
    // (0 where ambient), flame = (T / flame_temperature)^4 above ambient.
    void fill_export(FieldView density, FieldView dust, FieldView condensate, FieldView temperature, FieldView flame,
                     FieldView vel_x, FieldView vel_y, FieldView vel_z, float flame_temperature);
    // One export field into dst (see ExportField), or into the solver's own scratch and then
    // to the host, so exporting needs no extra device memory.
    void export_field(ExportField f, FieldView dst, float flame_temperature);
    void export_field_to_host(ExportField f, float* host, float flame_temperature);

    float max_temperature();
    float max_speed();
    CloudDiagnostics diagnostics();

    const GridInfo& grid() const { return grid_; }
    Device device() const { return dev_; }
    FieldView theta() { return scalar(kTheta); }
    FieldView smoke() { return scalar(kSmoke); }
    bool has_dust() const { return dust_.enabled; }
    FieldView dust(int bin) { return scalar(kDust0 + bin); }
    bool has_moisture() const { return humidity_.enabled; }
    FieldView vapor() { return scalar(vapor_index_); }
    FieldView condensate() { return scalar(vapor_index_ + 1); }
    double condensate_tonnes();
    MacField& velocity() { return vel_; }
    const AtmosphereColumns& ambient() const { return ambient_; }
    const DustParams& dust_params() const { return dust_; }
    std::size_t memory_bytes() const;
    StageProfiler& profiler() { return profiler_; }

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

    // Vapour and cloud water arrays (null without moisture) and the ambient vapour column.
    struct MoisturePointers {
        float* qv = nullptr;
        float* qc = nullptr;
        const float* qv_ambient = nullptr;
    };
    MoisturePointers moisture_pointers();

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
    HumidityProfile humidity_;
    int vapor_index_ = -1;      // vapour, then cloud water right after it
    Buffer<float> ambient_qv_;  // ambient specific humidity per level
    AtmosphereColumns ambient_;
    Buffer<float> wind_u_, wind_v_;  // ambient wind per level, for the current grid

    // Transported cell fields (theta, smoke, dust bins...) and two shared cell temporaries.
    // Memory is what limits the resolution, so the step reuses these and the advection
    // scratch instead of keeping per-field copies:
    //   tmp_a_, tmp_b_  MacCormack forward/result buffers, diffusion and settling output,
    //                   vorticity components (with scratch_.c), the eddy viscosity
    //   scratch_        traced positions, the confinement force (px, py, pz), face temporaries
    std::vector<Buffer<float>> scalars_;
    Buffer<float> tmp_a_, tmp_b_;
    Buffer<float> rho_, source_;
    MacField vel_, vel_a_;
    AdvectionScratch scratch_;
    PressureSolver pressure_;
    StageProfiler profiler_;
};

}  // namespace sim
