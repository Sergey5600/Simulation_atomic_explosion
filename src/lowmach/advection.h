#pragma once
// Semi-Lagrangian MacCormack advection (Selle et al. 2008) with the
// limiter that falls back to plain semi-Lagrangian where it would create
// new extrema.

#include "core/grid.h"
#include "lowmach/mac.h"

namespace sim {

// Scratch arrays large enough for any staggered component of an n-cell grid.
struct AdvectionScratch {
    Buffer<float> a, b;
    AdvectionScratch(Int3 n, Device dev) : a(dev, max_face_count(n)), b(dev, max_face_count(n)) {}
};

// dst = src carried along `vel` for dt seconds. src and dst share dims and offset
// and must not alias.
void advect_maccormack(Device dev, const MacView& vel, float dt, float dx, const SampleView& src, FieldView dst,
                       AdvectionScratch& scratch);

// Advects the three components of `field` by `vel` into `out`.
void advect_velocity(Device dev, const MacView& vel, float dt, float dx, const MacField& field, MacField& out,
                     AdvectionScratch& scratch);

}  // namespace sim
