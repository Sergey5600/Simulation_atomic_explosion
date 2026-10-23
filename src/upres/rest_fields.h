#pragma once
// "Rest" fields for render-time detail (M6).
//
// A rest field stores, for every point of the flow, where that material was when
// the field was last reset; it is carried by the velocity like any tracer. The
// renderer evaluates procedural noise at the rest position, so the noise moves
// and stretches with the smoke instead of standing still in space: detail far
// below the voxel size that stays coherent from frame to frame (the texture
// coordinates of wavelet turbulence, Kim et al. 2008).
//
// A rest field drifts out of shape as the flow shears it, so two are kept and
// reset alternately; the renderer cross-fades them with weight_a() (each field
// has zero weight at its own reset). Resets follow a logarithmic clock because
// the explosion slows down as it grows: each period is `growth_ratio` times the
// previous one, which keeps the distortion per period about constant.
//
// Values are rest positions divided by the noise feature length L chosen at the
// reset (feature_voxels * simulation voxel), so noise of scale 1 in the renderer
// matches the grid of that moment. The fields are smooth, so they live on a grid
// with twice the voxel size of the simulation.

#include "core/grid.h"
#include "lowmach/mac.h"

namespace sim {

struct RestParams {
    bool enabled = true;
    float feature_voxels = 3.f;  // noise feature length L in simulation voxels at the reset
    float growth_ratio = 1.6f;   // each reset period is this many times the previous one
    float clock_origin = 1e-3f;  // s, the logarithmic clock counts periods from here
};

class RestFields {
public:
    RestFields(Device dev, const GridInfo& sim_grid, const RestParams& params);

    // Both fields set to the current positions.
    void reset(double time);
    // Carries both fields along `vel` (a sampler in simulation-grid units) from time - dt to
    // `time`, then resets the fields that are due.
    void advect(const MacView& vel, double dt, double time);
    // Follows the simulation grid when it doubles (LowMachSolver::coarsen_domain).
    void coarsen(const GridInfo& new_sim_grid);

    // Weight of field A in the cross-fade at `time`; field B gets 1 - weight.
    float weight_a(double time) const;

    const GridInfo& grid() const { return grid_; }  // the rest grid (twice the simulation voxel)
    // field 0 = A, 1 = B; c = x, y, z.
    const Buffer<float>& component(int field, int c) const { return buf_[3 * field + c]; }
    std::size_t memory_bytes() const { return 7 * tmp_.bytes(); }

    // Kernels. Public only because nvcc does not allow device lambdas inside private member functions.
    void set_to_position(int field);

private:
    double clock(double time) const;

    Device dev_;
    RestParams params_;
    GridInfo sim_grid_;
    GridInfo grid_;
    Buffer<float> buf_[6];
    Buffer<float> tmp_;
    float length_[2] = {1.f, 1.f};  // m, feature length of each field since its last reset
    long long period_[2] = {0, 0};  // clock periods at the last resets
};

}  // namespace sim
