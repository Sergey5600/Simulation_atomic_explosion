#pragma once
// A scene produces the exported volume fields at requested simulation times.

#include <memory>
#include <string>

#include "core/grid.h"
#include "io/config.h"

namespace sim {

// Cell-centred fields every scene fills for export, on the compute device.
struct ExportBuffers {
    ScalarField density, dust, temperature, flame, vel_x, vel_y, vel_z;

    ExportBuffers(const GridInfo& g, Device dev)
        : density("density", g, dev),
          dust("dust", g, dev),
          temperature("temperature", g, dev),
          flame("flame", g, dev),
          vel_x("vel_x", g, dev),
          vel_y("vel_y", g, dev),
          vel_z("vel_z", g, dev) {}

    std::size_t memory_bytes() const { return 7 * density.buffer().bytes(); }
};

class Scene {
public:
    virtual ~Scene() = default;

    // Simulates forward to sim time t (seconds after detonation). Times only increase.
    virtual void advance_to(double t) = 0;

    // Current grid; may change between frames when the domain grows (same cell counts).
    virtual GridInfo grid() const = 0;
    virtual void fill_export(ExportBuffers& out) = 0;

    // One line for the per-frame log (steps, solver iterations, extremes).
    virtual std::string status() { return {}; }
    virtual std::size_t memory_bytes() const { return 0; }
};

std::unique_ptr<Scene> make_scene(const SceneConfig& cfg, Device dev);

}  // namespace sim
