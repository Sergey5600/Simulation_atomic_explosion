#pragma once
// Scene "fireball_rise": the buoyant phase of a surface burst, from a hot
// fireball at ambient pressure to the rising mushroom cloud (M1).

#include "scenes/scene.h"

namespace sim {

// Writes the initial fireball into the solver state (on top of the ambient atmosphere).
void init_fireball(LowMachSolver& solver, const FireballConfig& cfg, std::uint32_t seed);

std::unique_ptr<Scene> make_fireball_rise_scene(const SceneConfig& cfg, Device dev);

}  // namespace sim
