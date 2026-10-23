#include "lowmach/advection.h"

#include "core/parallel.h"

namespace sim {

void advect_maccormack(Device dev, const MacView& vel, float dt, float dx, const SampleView& src, FieldView dst,
                       AdvectionScratch& scratch) {
    const Int3 n = src.dims;
    const Float3 off = src.offset;
    const float s = dt / dx;
    FieldView fwd{scratch.a.data(), n};
    FieldView bwd{scratch.b.data(), n};
    const SampleView fwd_s{scratch.a.data(), n, off};

    // Forward semi-Lagrangian step.
    parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 g = make_f3(i + off.x, j + off.y, k + off.z);
        fwd(i, j, k) = src.sample(vel.trace(g, s));
    });
    // Backward step from the forward result.
    parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 g = make_f3(i + off.x, j + off.y, k + off.z);
        bwd(i, j, k) = fwd_s.sample(vel.trace(g, -s));
    });
    // Error correction, limited to the range of the values the forward step interpolated.
    parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 g = make_f3(i + off.x, j + off.y, k + off.z);
        const float f = fwd(i, j, k);
        const float corrected = f + 0.5f * (src.at(i, j, k) - bwd(i, j, k));
        float lo, hi;
        src.minmax(vel.trace(g, s), lo, hi);
        dst(i, j, k) = (corrected < lo || corrected > hi) ? f : corrected;
    });
}

void advect_velocity(Device dev, const MacView& vel, float dt, float dx, const MacField& field, MacField& out,
                     AdvectionScratch& scratch) {
    const MacView src = field.sampler();
    advect_maccormack(dev, vel, dt, dx, src.u, out.u_view(), scratch);
    advect_maccormack(dev, vel, dt, dx, src.v, out.v_view(), scratch);
    advect_maccormack(dev, vel, dt, dx, src.w, out.w_view(), scratch);
    // The ground stays impermeable.
    FieldView w = out.w_view();
    parallel_for(dev, Int3{out.dims.x, out.dims.y, 1}, SIM_LAMBDA(int i, int j, int) { w(i, j, 0) = 0.f; });
}

}  // namespace sim
