#pragma once
// Scene "fireball_rise": the buoyant phase of a surface burst, from a hot
// fireball at ambient pressure to the rising mushroom cloud (M1).

#include <string>

#include "lowmach/solver.h"
#include "scenes/scene.h"

namespace sim {

// Writes the initial fireball into the solver state (on top of the ambient atmosphere).
void init_fireball(LowMachSolver& solver, const FireballConfig& cfg, std::uint32_t seed);

// Adds the dust present at the start: soil inside the fireball and the skirt raised by the shock.
void init_dust(LowMachSolver& solver, const FireballConfig& fireball, const DustInitConfig& cfg, std::uint32_t seed);

// The buoyant phase: the low-Mach solver stepped to each frame time, growing its domain
// as the cloud grows. Shared by the fireball_rise and surface_burst scenes.
class RisePhase {
public:
    RisePhase(const SceneConfig& cfg, Device dev, const Atmosphere& atm);

    void advance_to(double t);
    void fill_export(ExportBuffers& out);
    std::string status();

    LowMachSolver& solver() { return solver_; }
    GridInfo grid() const { return solver_.grid(); }
    double time() const { return time_; }
    void set_time(double t) { time_ = t; }
    std::size_t memory_bytes() const { return solver_.memory_bytes(); }

private:
    void maybe_grow_domain();

    SceneConfig cfg_;
    LowMachSolver solver_;
    double time_ = 0.0;
    int expansions_ = 0, steps_since_check_ = 0;
    int steps_ = 0, iterations_ = 0, projections_ = 0;
    double residual_ = 0.0, last_dt_ = 0.0;
};

std::unique_ptr<Scene> make_fireball_rise_scene(const SceneConfig& cfg, Device dev);

}  // namespace sim
