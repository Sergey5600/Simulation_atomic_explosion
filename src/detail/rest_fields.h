#pragma once
// "Rest" fields for render-time detail (M6).
//
// A rest field stores, for every point of the flow, where that material was when
// the field was last reset; it is carried by the velocity like any tracer. A
// renderer can then evaluate procedural noise at the rest position: the noise
// moves and stretches with the smoke instead of staying fixed in space, giving
// sub-voxel detail that is coherent from frame to frame.
//
// Rest fields drift out of shape over time, so two of them are kept and reset
// alternately; the renderer cross-fades between them with rest_weight() (each one
// has zero weight at its own reset). Resets happen on a logarithmic clock because
// the flow slows down as the explosion grows: a field lives `growth_ratio` times
// longer than the previous one.
//
// The fields are smooth, so they live on a grid with half the resolution of the
// simulation and are advected semi-Lagrangian (exact for a linear field).

#include "core/grid.h"
#include "lowmach/mac.h"

namespace sim {

struct RestParams {
    bool enabled = true;
    float growth_ratio = 1.6f;   // each reset period is this many times the last
    float start_time = 0.01f;    // s, the logarithmic clock starts here
};

class RestFields {
public:
    RestFields(Device dev, const GridInfo& sim_grid, const RestParams& params);

    // Both fields set to the current positions.
    void reset(double time);
    // Advect by `vel` (sampled in simulation-grid units) from time - dt to time; resets as due.
    void advect(const MacView& vel, double dt, double time);
    // Follow the simulation grid when it doubles (see LowMachSolver::coarsen_domain).
    void coarsen(const GridInfo& new_sim_grid);

    // Weight of field A in the cross-fade at `time` (field B gets 1 - weight).
    float weight_a(double time) const;

    const GridInfo& grid() const { return grid_; }  // the half-resolution grid
    Device device() const { return dev_; }

    // a = 0 or 1, c = component (x, y, z): world positions in metres.
    Buffer<float>& component(int field, int c) { return buf_[std::size_t(3 * field + c)]; }

    // Kernels. Public only because nvcc does not allow device lambdas inside private member functions.
    void set_to_position(int field);

private:
    double clock(double time) const;  // logarithmic time: one unit per reset period

    Device dev_;
    RestParams params_;
    GridInfo sim_grid_;
    GridInfo grid_;
    Buffer<float> buf_[6];
    Buffer<float> tmp_;
    double last_clock_ = 0.0;
};

}  // namespace sim
