#include "lowmach/advection.h"

#include <stdexcept>
#include <utility>

#include "core/parallel.h"

namespace sim {

void advect_maccormack(Device dev, const MacView& vel, float dt, float dx, const SampleView& src, FieldView dst,
                       AdvectionScratch& scratch) {
    const Int3 n = src.dims;
    const Float3 off = src.offset;
    const float s = dt / dx;
    FieldView fwd{scratch.a.data(), n};
    FieldView px{scratch.px.data(), n}, py{scratch.py.data(), n}, pz{scratch.pz.data(), n};
    const SampleView fwd_s{scratch.a.data(), n, off};

    // Forward semi-Lagrangian step (the traced point is kept for the limiter).
    parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 g = make_f3(i + off.x, j + off.y, k + off.z);
        const Float3 p = vel.trace(g, s);
        px(i, j, k) = p.x;
        py(i, j, k) = p.y;
        pz(i, j, k) = p.z;
        fwd(i, j, k) = src.sample(p);
    });
    // Backward step from the forward result and error correction, limited to the range of the
    // values the forward step interpolated. Each cell reads only its own backward value, so the
    // two are fused.
    parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 g = make_f3(i + off.x, j + off.y, k + off.z);
        const float bwd = fwd_s.sample(vel.trace(g, -s));
        const float f = fwd(i, j, k);
        const float corrected = f + 0.5f * (src.at(i, j, k) - bwd);
        float lo, hi;
        src.minmax(make_f3(px(i, j, k), py(i, j, k), pz(i, j, k)), lo, hi);
        dst(i, j, k) = (corrected < lo || corrected > hi) ? f : corrected;
    });
}

void advect_cells_maccormack(Device dev, const MacView& vel, float dt, float dx, Int3 n, Buffer<float>* const* fields,
                             int count, Buffer<float>& fwd_buf, Buffer<float>& out_buf, AdvectionScratch& scratch) {
    if (count <= 0) return;
    const float s = dt / dx;
    FieldView px{scratch.px.data(), n}, py{scratch.py.data(), n}, pz{scratch.pz.data(), n};
    FieldView qx{scratch.a.data(), n}, qy{scratch.b.data(), n}, qz{scratch.c.data(), n};

    parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 g = make_f3(i + 0.5f, j + 0.5f, k + 0.5f);
        const Float3 p = vel.trace(g, s);
        px(i, j, k) = p.x;
        py(i, j, k) = p.y;
        pz(i, j, k) = p.z;
        const Float3 q = vel.trace(g, -s);
        qx(i, j, k) = q.x;
        qy(i, j, k) = q.y;
        qz(i, j, k) = q.z;
    });

    for (int c = 0; c < count; ++c) {
        const SampleView src{fields[c]->data(), n, kCellOffset};
        const SampleView fwd_s{fwd_buf.data(), n, kCellOffset};
        FieldView fwd{fwd_buf.data(), n}, out{out_buf.data(), n};
        parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
            fwd(i, j, k) = src.sample(make_f3(px(i, j, k), py(i, j, k), pz(i, j, k)));
        });
        parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
            const float bwd = fwd_s.sample(make_f3(qx(i, j, k), qy(i, j, k), qz(i, j, k)));
            const float fw = fwd(i, j, k);
            const float corrected = fw + 0.5f * (src.at(i, j, k) - bwd);
            float lo, hi;
            src.minmax(make_f3(px(i, j, k), py(i, j, k), pz(i, j, k)), lo, hi);
            out(i, j, k) = (corrected < lo || corrected > hi) ? fw : corrected;
        });
        std::swap(*fields[c], out_buf);
    }
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

void advect_velocity_in_place(Device dev, const MacView& vel, float dt, float dx, MacField& field,
                              AdvectionScratch& scratch) {
    const Int3 n = field.dims;
    advect_maccormack(dev, vel, dt, dx, field.sampler().u, {scratch.b.data(), u_dims(n)}, scratch);
    std::swap(field.u, scratch.b);
    advect_maccormack(dev, vel, dt, dx, field.sampler().v, {scratch.b.data(), v_dims(n)}, scratch);
    std::swap(field.v, scratch.b);
    advect_maccormack(dev, vel, dt, dx, field.sampler().w, {scratch.b.data(), w_dims(n)}, scratch);
    std::swap(field.w, scratch.b);
    FieldView w = field.w_view();
    parallel_for(dev, Int3{n.x, n.y, 1}, SIM_LAMBDA(int i, int j, int) { w(i, j, 0) = 0.f; });
}

}  // namespace sim
