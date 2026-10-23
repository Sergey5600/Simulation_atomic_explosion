#include "lowmach/pressure.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/parallel.h"

namespace sim {

namespace detail {

constexpr int kPreSmooth = 2;
constexpr int kPostSmooth = 2;
constexpr int kCoarsestSweeps = 40;
// Restriction = sum of the 8 children * weight (1/8 = average, consistent with the
// re-discretised coarse operator for smooth errors).
constexpr float kRestrictWeight = 1.f / 8.f;

// Coefficient views of one level, passed by value into kernels.
struct Stencil {
    FieldView bx, by, bz;
    Int3 n;
    float inv_dx2;

    SIM_INLINE float diag(int i, int j, int k) const {
        return (bx(i, j, k) + bx(i + 1, j, k) + by(i, j, k) + by(i, j + 1, k) + bz(i, j, k) + bz(i, j, k + 1)) *
               inv_dx2;
    }
    // Sum over neighbours of coefficient * x; cells outside the domain are 0 (Dirichlet)
    // or sit behind a zero coefficient (wall).
    SIM_INLINE float neighbours(FieldView x, int i, int j, int k) const {
        float s = 0.f;
        if (i > 0) s += bx(i, j, k) * x(i - 1, j, k);
        if (i < n.x - 1) s += bx(i + 1, j, k) * x(i + 1, j, k);
        if (j > 0) s += by(i, j, k) * x(i, j - 1, k);
        if (j < n.y - 1) s += by(i, j + 1, k) * x(i, j + 1, k);
        if (k > 0) s += bz(i, j, k) * x(i, j, k - 1);
        if (k < n.z - 1) s += bz(i, j, k + 1) * x(i, j, k + 1);
        return s * inv_dx2;
    }
};

inline Stencil stencil_of(Int3 n, float dx, Buffer<float>& bx, Buffer<float>& by, Buffer<float>& bz) {
    return Stencil{{bx.data(), u_dims(n)}, {by.data(), v_dims(n)}, {bz.data(), w_dims(n)}, n, 1.f / (dx * dx)};
}

}  // namespace detail

using detail::Stencil;
using detail::stencil_of;

PressureSolver::PressureSolver(Device dev, Int3 dims, float dx) : dev_(dev), dims_(dims), dx_(dx) {
    const std::size_t n = std::size_t(dims.volume());
    phi_.allocate(dev, n);
    rhs_.allocate(dev, n);
    r_.allocate(dev, n);
    z_.allocate(dev, n);
    p_.allocate(dev, n);
    ap_.allocate(dev, n);

    Int3 d = dims;
    float h = dx;
    for (;;) {
        Level lv;
        lv.dims = d;
        lv.dx = h;
        lv.bx.allocate(dev, std::size_t(u_dims(d).volume()));
        lv.by.allocate(dev, std::size_t(v_dims(d).volume()));
        lv.bz.allocate(dev, std::size_t(w_dims(d).volume()));
        lv.r.allocate(dev, std::size_t(d.volume()));
        if (!levels_.empty()) {
            lv.x.allocate(dev, std::size_t(d.volume()));
            lv.b.allocate(dev, std::size_t(d.volume()));
        }
        levels_.push_back(std::move(lv));
        const bool even = d.x % 2 == 0 && d.y % 2 == 0 && d.z % 2 == 0;
        if (!even || std::min({d.x, d.y, d.z}) < 8 || levels_.size() >= 12) break;
        d = Int3{d.x / 2, d.y / 2, d.z / 2};
        h *= 2.f;
    }
}

std::size_t PressureSolver::memory_bytes() const {
    std::size_t b = phi_.bytes() + rhs_.bytes() + r_.bytes() + z_.bytes() + p_.bytes() + ap_.bytes();
    for (const Level& lv : levels_) b += lv.bx.bytes() + lv.by.bytes() + lv.bz.bytes() + lv.x.bytes() + lv.b.bytes() + lv.r.bytes();
    return b;
}

void PressureSolver::set_density(FieldView rho) {
    const Int3 n = dims_;
    Level& l0 = levels_[0];
    FieldView bx{l0.bx.data(), u_dims(n)}, by{l0.by.data(), v_dims(n)}, bz{l0.bz.data(), w_dims(n)};

    parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) {
        if (i == 0) bx(i, j, k) = 2.f / rho(0, j, k);
        else if (i == n.x) bx(i, j, k) = 2.f / rho(n.x - 1, j, k);
        else bx(i, j, k) = 2.f / (rho(i - 1, j, k) + rho(i, j, k));
    });
    parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) {
        if (j == 0) by(i, j, k) = 2.f / rho(i, 0, k);
        else if (j == n.y) by(i, j, k) = 2.f / rho(i, n.y - 1, k);
        else by(i, j, k) = 2.f / (rho(i, j - 1, k) + rho(i, j, k));
    });
    parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) {
        if (k == 0) bz(i, j, k) = 0.f;  // ground: solid wall
        else if (k == n.z) bz(i, j, k) = 2.f / rho(i, j, n.z - 1);
        else bz(i, j, k) = 2.f / (rho(i, j, k - 1) + rho(i, j, k));
    });

    // Coarse coefficients: average of the 4 fine faces covering each coarse face.
    for (std::size_t l = 1; l < levels_.size(); ++l) {
        const Level& f = levels_[l - 1];
        Level& c = levels_[l];
        const Int3 nf = f.dims, nc = c.dims;
        FieldView fx{const_cast<float*>(f.bx.data()), u_dims(nf)}, fy{const_cast<float*>(f.by.data()), v_dims(nf)},
            fz{const_cast<float*>(f.bz.data()), w_dims(nf)};
        FieldView cx{c.bx.data(), u_dims(nc)}, cy{c.by.data(), v_dims(nc)}, cz{c.bz.data(), w_dims(nc)};
        parallel_for(dev_, u_dims(nc), SIM_LAMBDA(int i, int j, int k) {
            cx(i, j, k) = 0.25f * (fx(2 * i, 2 * j, 2 * k) + fx(2 * i, 2 * j + 1, 2 * k) + fx(2 * i, 2 * j, 2 * k + 1) +
                                   fx(2 * i, 2 * j + 1, 2 * k + 1));
        });
        parallel_for(dev_, v_dims(nc), SIM_LAMBDA(int i, int j, int k) {
            cy(i, j, k) = 0.25f * (fy(2 * i, 2 * j, 2 * k) + fy(2 * i + 1, 2 * j, 2 * k) + fy(2 * i, 2 * j, 2 * k + 1) +
                                   fy(2 * i + 1, 2 * j, 2 * k + 1));
        });
        parallel_for(dev_, w_dims(nc), SIM_LAMBDA(int i, int j, int k) {
            cz(i, j, k) = 0.25f * (fz(2 * i, 2 * j, 2 * k) + fz(2 * i + 1, 2 * j, 2 * k) + fz(2 * i, 2 * j + 1, 2 * k) +
                                   fz(2 * i + 1, 2 * j + 1, 2 * k));
        });
    }
}

void PressureSolver::apply(std::size_t l, FieldView x, FieldView out) {
    Level& lv = levels_[l];
    const Stencil st = stencil_of(lv.dims, lv.dx, lv.bx, lv.by, lv.bz);
    parallel_for(dev_, lv.dims, SIM_LAMBDA(int i, int j, int k) {
        out(i, j, k) = st.diag(i, j, k) * x(i, j, k) - st.neighbours(x, i, j, k);
    });
}

void PressureSolver::residual(std::size_t l, FieldView x, FieldView b, FieldView r) {
    Level& lv = levels_[l];
    const Stencil st = stencil_of(lv.dims, lv.dx, lv.bx, lv.by, lv.bz);
    parallel_for(dev_, lv.dims, SIM_LAMBDA(int i, int j, int k) {
        r(i, j, k) = b(i, j, k) - (st.diag(i, j, k) * x(i, j, k) - st.neighbours(x, i, j, k));
    });
}

void PressureSolver::smooth(std::size_t l, FieldView x, FieldView b, int color) {
    Level& lv = levels_[l];
    const Stencil st = stencil_of(lv.dims, lv.dx, lv.bx, lv.by, lv.bz);
    parallel_for(dev_, lv.dims, SIM_LAMBDA(int i, int j, int k) {
        if (((i + j + k) & 1) != color) return;
        const float d = st.diag(i, j, k);
        if (d > 0.f) x(i, j, k) = (b(i, j, k) + st.neighbours(x, i, j, k)) / d;
    });
}

void PressureSolver::vcycle(std::size_t l, FieldView x, FieldView b) {
    Level& lv = levels_[l];
    const Int3 n = lv.dims;
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) { x(i, j, k) = 0.f; });

    if (l + 1 == levels_.size()) {
        for (int s = 0; s < detail::kCoarsestSweeps; ++s) {
            smooth(l, x, b, 0);
            smooth(l, x, b, 1);
        }
        for (int s = 0; s < detail::kCoarsestSweeps; ++s) {
            smooth(l, x, b, 1);
            smooth(l, x, b, 0);
        }
        return;
    }

    for (int s = 0; s < detail::kPreSmooth; ++s) {
        smooth(l, x, b, 0);
        smooth(l, x, b, 1);
    }
    FieldView r{lv.r.data(), n};
    residual(l, x, b, r);

    Level& cl = levels_[l + 1];
    FieldView cb{cl.b.data(), cl.dims}, cxv{cl.x.data(), cl.dims};
    const float weight = detail::kRestrictWeight;
    parallel_for(dev_, cl.dims, SIM_LAMBDA(int i, int j, int k) {
        float s = 0.f;
        for (int c = 0; c < 8; ++c) s += r(2 * i + (c & 1), 2 * j + ((c >> 1) & 1), 2 * k + ((c >> 2) & 1));
        cb(i, j, k) = weight * s;
    });
    vcycle(l + 1, cxv, cb);
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) { x(i, j, k) += cxv(i / 2, j / 2, k / 2); });

    for (int s = 0; s < detail::kPostSmooth; ++s) {
        smooth(l, x, b, 1);
        smooth(l, x, b, 0);
    }
}

double PressureSolver::dot(FieldView a, FieldView b) {
    return parallel_sum(dev_, dims_, SIM_LAMBDA(int i, int j, int k) { return double(a(i, j, k)) * double(b(i, j, k)); });
}

ProjectionStats PressureSolver::project(MacField& vel, FieldView source, float tolerance, int max_iterations) {
    const Int3 n = dims_;
    const float inv_dx = 1.f / dx_;
    FieldView u = vel.u_view(), v = vel.v_view(), w = vel.w_view();
    FieldView phi{phi_.data(), n}, rhs{rhs_.data(), n}, r{r_.data(), n}, z{z_.data(), n}, p{p_.data(), n},
        ap{ap_.data(), n};

    // rhs = S - div(u*)  (A = -div(beta grad), so A phi = rhs gives div(u) = S)
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const float div = (u(i + 1, j, k) - u(i, j, k) + v(i, j + 1, k) - v(i, j, k) + w(i, j, k + 1) - w(i, j, k)) * inv_dx;
        rhs(i, j, k) = source(i, j, k) - div;
    });

    ProjectionStats stats;
    const double cells = double(n.volume());
    const double bnorm = std::sqrt(dot(rhs, rhs));
    stats.initial_residual = bnorm / std::sqrt(cells);
    // Below this RMS divergence error (1/s) the flow is divergence-free for any practical purpose.
    const double target = std::max(double(tolerance) * bnorm, 1e-6 * std::sqrt(cells));
    if (bnorm <= target) {
        stats.final_residual = stats.initial_residual;
        return stats;
    }

    // Warm start from the previous solution.
    residual(0, phi, rhs, r);
    double rnorm = std::sqrt(dot(r, r));
    if (rnorm > target) {
        vcycle(0, z, r);
        parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) { p(i, j, k) = z(i, j, k); });
        double rz = dot(r, z);
        for (int it = 0; it < max_iterations; ++it) {
            apply(0, p, ap);
            const double pap = dot(p, ap);
            if (!(pap > 0.0)) break;
            const float alpha = float(rz / pap);
            parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
                phi(i, j, k) += alpha * p(i, j, k);
                r(i, j, k) -= alpha * ap(i, j, k);
            });
            stats.iterations = it + 1;
            rnorm = std::sqrt(dot(r, r));
            if (rnorm <= target) break;
            vcycle(0, z, r);
            const double rz_new = dot(r, z);
            const float beta = float(rz_new / rz);
            rz = rz_new;
            parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) { p(i, j, k) = z(i, j, k) + beta * p(i, j, k); });
        }
    }
    stats.final_residual = rnorm / std::sqrt(cells);

    // u -= beta grad(phi), with phi = 0 outside the open boundaries.
    Level& l0 = levels_[0];
    FieldView bx{l0.bx.data(), u_dims(n)}, by{l0.by.data(), v_dims(n)}, bz{l0.bz.data(), w_dims(n)};
    parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) {
        const float hi = i < n.x ? phi(i, j, k) : 0.f;
        const float lo = i > 0 ? phi(i - 1, j, k) : 0.f;
        u(i, j, k) -= bx(i, j, k) * (hi - lo) * inv_dx;
    });
    parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) {
        const float hi = j < n.y ? phi(i, j, k) : 0.f;
        const float lo = j > 0 ? phi(i, j - 1, k) : 0.f;
        v(i, j, k) -= by(i, j, k) * (hi - lo) * inv_dx;
    });
    parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) {
        const float hi = k < n.z ? phi(i, j, k) : 0.f;
        const float lo = k > 0 ? phi(i, j, k - 1) : 0.f;
        w(i, j, k) -= bz(i, j, k) * (hi - lo) * inv_dx;
    });
    return stats;
}

}  // namespace sim
