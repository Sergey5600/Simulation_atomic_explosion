#pragma once
// A scene produces the exported volume fields at requested simulation times.

#include <memory>
#include <string>

#include "core/export_field.h"
#include "core/grid.h"
#include "io/config.h"

namespace sim {

class Scene {
public:
    virtual ~Scene() = default;

    // Simulates forward to sim time t (seconds after detonation). Times only increase.
    virtual void advance_to(double t) = 0;

    // Current grid; may change between frames when the domain grows (same cell counts).
    virtual GridInfo grid() const = 0;
    // Computes one export field for the current state and copies it to `host`
    // (grid().count() floats). Scenes use their own scratch memory for this.
    virtual void export_field(ExportField f, float* host) = 0;

    // One line for the per-frame log (steps, solver iterations, extremes).
    virtual std::string status() { return {}; }
    // Per-stage timing since the last call when profiling is on (SceneConfig::profile), else empty.
    virtual std::string profile_report() { return {}; }
    virtual std::size_t memory_bytes() const { return 0; }
};

std::unique_ptr<Scene> make_scene(const SceneConfig& cfg, Device dev);

}  // namespace sim
