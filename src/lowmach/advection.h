#pragma once
// Semi-Lagrangian MacCormack advection (Selle et al. 2008) with the
// limiter that falls back to plain semi-Lagrangian where it would create
// new extrema.

#include "core/grid.h"
#include "lowmach/mac.h"

namespace sim {

// Scratch arrays large enough for any staggered component of an n-cell grid (and so for
// any cell field). Between advection calls they are free for other temporary use.
struct AdvectionScratch {
    Buffer<float> a, b, c;
    Buffer<float> px, py, pz;  // forward-traced positions, reused by the limiter pass
    AdvectionScratch(Int3 n, Device dev)
        : a(dev, max_face_count(n)),
          b(dev, max_face_count(n)),
          c(dev, max_face_count(n)),
          px(dev, max_face_count(n)),
          py(dev, max_face_count(n)),
          pz(dev, max_face_count(n)) {}
    std::size_t memory_bytes() const { return 6 * a.bytes(); }
};

// dst = src carried along `vel` for dt seconds. src and dst share dims and offset
// and must not alias. Uses scratch.a and the traced positions.
void advect_maccormack(Device dev, const MacView& vel, float dt, float dx, const SampleView& src, FieldView dst,
                       AdvectionScratch& scratch);

// Cell-centred fields advected in place, one after another. The characteristics are traced
// once (forward into scratch.px/py/pz, backward into scratch.a/b/c) and shared by all fields,
// so the only per-field memory is two shared temporaries: `fwd` and `out` (cell-sized).
// After the call each field buffer holds its advected values; `out` holds garbage.
void advect_cells_maccormack(Device dev, const MacView& vel, float dt, float dx, Int3 n, Buffer<float>* const* fields,
                             int count, Buffer<float>& fwd, Buffer<float>& out, AdvectionScratch& scratch);

// Advects the three components of `field` by `vel` into `out`.
void advect_velocity(Device dev, const MacView& vel, float dt, float dx, const MacField& field, MacField& out,
                     AdvectionScratch& scratch);

// Advects the three components of `field` by `vel` in place (through scratch.b, which must
// be swappable with the components: see MacField). `vel` must not view `field`.
void advect_velocity_in_place(Device dev, const MacView& vel, float dt, float dx, MacField& field,
                              AdvectionScratch& scratch);

}  // namespace sim
