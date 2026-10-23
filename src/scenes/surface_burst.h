#pragma once
// Scene "surface_burst": the whole explosion from half a millisecond on. Stage A
// (compressible) covers the blast wave and the growing fireball, then the state is
// handed to the low-Mach solver for the rising mushroom cloud (stage B).

#include "scenes/scene.h"

namespace sim {

std::unique_ptr<Scene> make_surface_burst_scene(const SceneConfig& cfg, Device dev);

}  // namespace sim
