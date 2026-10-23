#include "lowmach/advection.h"

#include <stdexcept>

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

void advect_cells_maccormack(Device dev, const MacView& vel, float dt, float dx, Int3 n, const CellFieldSet& fields) {
    if (fields.count <= 0 || fields.count > kMaxAdvectFields) throw std::invalid_argument("advect_cells: bad field count");
    const float s = dt / dx;
    const CellFieldSet f = fields;

    parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 p = vel.trace(make_f3(i + 0.5f, j + 0.5f, k + 0.5f), s);
        const std::int64_t idx = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
        for (int c = 0; c < f.count; ++c) f.dst[c][idx] = SampleView{f.src[c], n, make_f3(0.5f, 0.5f, 0.5f)}.sample(p);
    });
    parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 p = vel.trace(make_f3(i + 0.5f, j + 0.5f, k + 0.5f), -s);
        const std::int64_t idx = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
        for (int c = 0; c < f.count; ++c) f.bwd[c][idx] = SampleView{f.dst[c], n, make_f3(0.5f, 0.5f, 0.5f)}.sample(p);
    });
    parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 p = vel.trace(make_f3(i + 0.5f, j + 0.5f, k + 0.5f), s);
        const std::int64_t idx = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
        for (int c = 0; c < f.count; ++c) {
            const float fw = f.dst[c][idx];
            const float corrected = fw + 0.5f * (f.src[c][idx] - f.bwd[c][idx]);
            float lo, hi;
            SampleView{f.src[c], n, make_f3(0.5f, 0.5f, 0.5f)}.minmax(p, lo, hi);
            f.dst[c][idx] = (corrected < lo || corrected > hi) ? fw : corrected;
        }
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
