#include "upres/rest_fields.h"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "core/parallel.h"

namespace sim {

namespace {

GridInfo rest_grid_for(const GridInfo& sim) {
    GridInfo g;
    g.dims = Int3{(sim.dims.x + 1) / 2, (sim.dims.y + 1) / 2, (sim.dims.z + 1) / 2};
    g.dx = 2.f * sim.dx;
    g.origin = sim.origin;
    return g;
}

}  // namespace

RestFields::RestFields(Device dev, const GridInfo& sim_grid, const RestParams& params)
    : dev_(dev), params_(params), sim_grid_(sim_grid), grid_(rest_grid_for(sim_grid)) {
    if (!(params_.growth_ratio > 1.f)) throw std::invalid_argument("detail.growth_ratio must be > 1");
    if (!(params_.feature_voxels > 0.f)) throw std::invalid_argument("detail.feature_voxels must be > 0");
    const std::size_t n = std::size_t(grid_.count());
    for (Buffer<float>& b : buf_) b.allocate(dev, n);
    tmp_.allocate(dev, n);
}

double RestFields::clock(double time) const {
    return std::log(std::max(time, 1e-9) / double(params_.clock_origin)) / std::log(double(params_.growth_ratio));
}

float RestFields::weight_a(double time) const {
    const double c = clock(time);
    const double phase = c - std::floor(c);  // field A resets at phase 0, field B at phase 0.5
    return float(1.0 - std::fabs(2.0 * phase - 1.0));
}

void RestFields::reset(double time) {
    const double c = clock(time);
    period_[0] = static_cast<long long>(std::floor(c));
    period_[1] = static_cast<long long>(std::floor(c + 0.5));
    set_to_position(0);
    set_to_position(1);
}

void RestFields::set_to_position(int field) {
    const float length = params_.feature_voxels * sim_grid_.dx;
    length_[field] = length;
    const float inv = 1.f / length;
    const GridInfo g = grid_;
    FieldView x{buf_[3 * field].data(), g.dims}, y{buf_[3 * field + 1].data(), g.dims},
        z{buf_[3 * field + 2].data(), g.dims};
    parallel_for(dev_, g.dims, SIM_LAMBDA(int i, int j, int k) {
        const Float3 p = g.cell_center(i, j, k);
        x(i, j, k) = p.x * inv;
        y(i, j, k) = p.y * inv;
        z(i, j, k) = p.z * inv;
    });
}

void RestFields::advect(const MacView& vel, double dt, double time) {
    const GridInfo g = grid_;
    const GridInfo s = sim_grid_;
    const float ratio = g.dx / s.dx;
    const Float3 off = (g.origin - s.origin) * (1.f / s.dx);  // rest-grid origin in simulation units
    const float sdt = float(dt) / s.dx;
    for (int f = 0; f < 2; ++f) {
        const float inv_len = 1.f / length_[f];
        for (int c = 0; c < 3; ++c) {
            Buffer<float>& buf = buf_[3 * f + c];
            const SampleView src{buf.data(), g.dims, kCellOffset};
            FieldView dst{tmp_.data(), g.dims};
            parallel_for(dev_, g.dims, SIM_LAMBDA(int i, int j, int k) {
                const Float3 gs = off + make_f3((i + 0.5f) * ratio, (j + 0.5f) * ratio, (k + 0.5f) * ratio);
                const Float3 p = vel.trace(gs, sdt);
                const Float3 pr = (p - off) * (1.f / ratio);
                // Air coming in from beyond the outermost cell centres starts at its own position
                // (it was never marked), instead of copying the edge value.
                const bool outside = pr.x < 0.5f || pr.y < 0.5f || pr.z < 0.5f || pr.x > g.dims.x - 0.5f ||
                                     pr.y > g.dims.y - 0.5f || pr.z > g.dims.z - 0.5f;
                if (outside) {
                    const Float3 w = g.origin + pr * g.dx;
                    dst(i, j, k) = (c == 0 ? w.x : (c == 1 ? w.y : w.z)) * inv_len;
                } else {
                    dst(i, j, k) = src.sample(pr);
                }
            });
            std::swap(buf, tmp_);
        }
    }
    const double clk = clock(time);
    const long long pa = static_cast<long long>(std::floor(clk));
    const long long pb = static_cast<long long>(std::floor(clk + 0.5));
    if (pa != period_[0]) {
        period_[0] = pa;
        set_to_position(0);
    }
    if (pb != period_[1]) {
        period_[1] = pb;
        set_to_position(1);
    }
}

void RestFields::coarsen(const GridInfo& new_sim_grid) {
    const GridInfo old = grid_;
    const GridInfo ng = rest_grid_for(new_sim_grid);
    if (ng.count() != old.count()) throw std::runtime_error("RestFields::coarsen: the cell counts must stay the same");
    for (int f = 0; f < 2; ++f) {
        const float inv_len = 1.f / length_[f];
        for (int c = 0; c < 3; ++c) {
            Buffer<float>& buf = buf_[3 * f + c];
            const SampleView src{buf.data(), old.dims, kCellOffset};
            FieldView dst{tmp_.data(), ng.dims};
            parallel_for(dev_, ng.dims, SIM_LAMBDA(int i, int j, int k) {
                const Float3 p = ng.cell_center(i, j, k);
                const Float3 q = (p - old.origin) * (1.f / old.dx);
                const bool inside = q.x >= 0.f && q.y >= 0.f && q.z >= 0.f && q.x <= float(old.dims.x) &&
                                    q.y <= float(old.dims.y) && q.z <= float(old.dims.z);
                dst(i, j, k) = inside ? src.sample(q) : (c == 0 ? p.x : (c == 1 ? p.y : p.z)) * inv_len;
            });
            std::swap(buf, tmp_);
        }
    }
    sim_grid_ = new_sim_grid;
    grid_ = ng;
}

}  // namespace sim
