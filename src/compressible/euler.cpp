#include "compressible/euler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/noise.h"
#include "core/parallel.h"
#include "lowmach/mac.h"
#include "physics/constants.h"

namespace sim {

namespace detail {

constexpr int kEV = CompressibleSolver::kVars;     // conserved variables
constexpr int kES = CompressibleSolver::kScalars;  // transported scalars

struct ConsPointers {
    float* c[kEV];
};

// Primitive state in a frame rotated so that `un` is normal to the face.
struct Prim {
    float r, un, t1, t2, p;
    float y[kES];
};

struct EulerView {
    ConsPointers u;
    Int3 n;
    float gamma;
    const float* rho0c;  // hydrostatic background per level (zeros if not well-balanced)
    const float* p0c;

    // Perturbation primitives of cell (i, j, k) for direction d, with the boundary conditions:
    // reflecting ground (mirror, w -> -w), zero gradient elsewhere.
    SIM_INLINE Prim load(int i, int j, int k, int d) const {
        bool mirror = false;
        if (k < 0) {
            k = -1 - k;
            mirror = true;
        }
        k = k >= n.z ? n.z - 1 : k;
        i = i < 0 ? 0 : (i >= n.x ? n.x - 1 : i);
        j = j < 0 ? 0 : (j >= n.y ? n.y - 1 : j);
        const std::int64_t idx = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
        const float r = u.c[0][idx];
        const float inv = 1.f / r;
        const float vx = u.c[1][idx] * inv, vy = u.c[2][idx] * inv;
        float vz = u.c[3][idx] * inv;
        if (mirror) vz = -vz;
        const float p = (gamma - 1.f) * (u.c[4][idx] - 0.5f * r * (vx * vx + vy * vy + vz * vz));
        Prim q;
        q.r = r - rho0c[k];
        q.p = p - p0c[k];
        if (d == 0) {
            q.un = vx, q.t1 = vy, q.t2 = vz;
        } else if (d == 1) {
            q.un = vy, q.t1 = vx, q.t2 = vz;
        } else {
            q.un = vz, q.t1 = vx, q.t2 = vy;
        }
        for (int s = 0; s < kES; ++s) q.y[s] = u.c[5 + s][idx] * inv;
        return q;
    }
};

SIM_INLINE float mc_limiter(float a, float b) {
    if (a * b <= 0.f) return 0.f;
    const float s = a > 0.f ? 1.f : -1.f;
    return s * fminf(fminf(2.f * fabsf(a), 2.f * fabsf(b)), 0.5f * fabsf(a + b));
}

// MUSCL face values between cells b (left) and c (right), with a and e their outer neighbours.
SIM_INLINE void muscl(float a, float b, float c, float e, float& left, float& right) {
    left = b + 0.5f * mc_limiter(b - a, c - b);
    right = c - 0.5f * mc_limiter(c - b, e - c);
}

// Physical flux and conserved state of a full (not perturbation) primitive state, rotated frame.
SIM_INLINE void euler_flux(const Prim& q, float gamma, float* f, float* u) {
    const float e = q.p / (gamma - 1.f) + 0.5f * q.r * (q.un * q.un + q.t1 * q.t1 + q.t2 * q.t2);
    u[0] = q.r;
    u[1] = q.r * q.un;
    u[2] = q.r * q.t1;
    u[3] = q.r * q.t2;
    u[4] = e;
    f[0] = q.r * q.un;
    f[1] = q.r * q.un * q.un + q.p;
    f[2] = q.r * q.un * q.t1;
    f[3] = q.r * q.un * q.t2;
    f[4] = q.un * (e + q.p);
    for (int s = 0; s < kES; ++s) {
        u[5 + s] = q.r * q.y[s];
        f[5 + s] = q.r * q.un * q.y[s];
    }
}

// HLLC approximate Riemann solver (Toro), rotated frame.
SIM_INLINE void hllc(const Prim& L, const Prim& R, float gamma, float* F) {
    const float cl = sqrtf(gamma * L.p / L.r), cr = sqrtf(gamma * R.p / R.r);
    const float sl = fminf(L.un - cl, R.un - cr);
    const float sr = fmaxf(L.un + cl, R.un + cr);
    float fl[kEV], ul[kEV], fr[kEV], ur[kEV];
    euler_flux(L, gamma, fl, ul);
    euler_flux(R, gamma, fr, ur);
    if (sl >= 0.f) {
        for (int v = 0; v < kEV; ++v) F[v] = fl[v];
        return;
    }
    if (sr <= 0.f) {
        for (int v = 0; v < kEV; ++v) F[v] = fr[v];
        return;
    }
    const float denom = L.r * (sl - L.un) - R.r * (sr - R.un);
    const float ss = (R.p - L.p + L.r * L.un * (sl - L.un) - R.r * R.un * (sr - R.un)) / denom;
    const bool left = ss >= 0.f;
    const Prim& q = left ? L : R;
    const float sk = left ? sl : sr;
    const float* fk = left ? fl : fr;
    const float* uk = left ? ul : ur;
    const float factor = q.r * (sk - q.un) / (sk - ss);
    float us[kEV];
    us[0] = factor;
    us[1] = factor * ss;
    us[2] = factor * q.t1;
    us[3] = factor * q.t2;
    us[4] = factor * (uk[4] / q.r + (ss - q.un) * (ss + q.p / (q.r * (sk - q.un))));
    for (int s = 0; s < kES; ++s) us[5 + s] = factor * q.y[s];
    for (int v = 0; v < kEV; ++v) F[v] = fk[v] + sk * (us[v] - uk[v]);
}

SIM_INLINE Prim add_background(Prim q, float r0, float p0) {
    q.r += r0;
    q.p += p0;
    return q;
}

// Fluxes through the faces normal to direction d (face (i, j, k) lies on the low side of cell (i, j, k)).
void compute_fluxes(Device dev, const EulerView ev, ConsPointers flux, const float* rho0f, const float* p0f, int d) {
    const Int3 n = ev.n;
    const Int3 fd{n.x + (d == 0), n.y + (d == 1), n.z + (d == 2)};
    const float gamma = ev.gamma;
    parallel_for(dev, fd, SIM_LAMBDA(int i, int j, int k) {
        const int di = d == 0, dj = d == 1, dk = d == 2;
        const Prim a = ev.load(i - 2 * di, j - 2 * dj, k - 2 * dk, d);
        const Prim b = ev.load(i - di, j - dj, k - dk, d);
        const Prim c = ev.load(i, j, k, d);
        const Prim e = ev.load(i + di, j + dj, k + dk, d);
        Prim l, r;
        muscl(a.r, b.r, c.r, e.r, l.r, r.r);
        muscl(a.un, b.un, c.un, e.un, l.un, r.un);
        muscl(a.t1, b.t1, c.t1, e.t1, l.t1, r.t1);
        muscl(a.t2, b.t2, c.t2, e.t2, l.t2, r.t2);
        muscl(a.p, b.p, c.p, e.p, l.p, r.p);
        for (int s = 0; s < kES; ++s) muscl(a.y[s], b.y[s], c.y[s], e.y[s], l.y[s], r.y[s]);

        // Hydrostatic background at the face (exact, so a resting atmosphere has no flux).
        float r0, p0;
        const int kc = k < n.z ? k : n.z - 1;
        if (d == 2) {
            r0 = rho0f[k];
            p0 = p0f[k];
        } else {
            r0 = ev.rho0c[kc];
            p0 = ev.p0c[kc];
        }
        l = add_background(l, r0, p0);
        r = add_background(r, r0, p0);
        if (!(l.r > 0.f && l.p > 0.f && r.r > 0.f && r.p > 0.f)) {
            // First order fallback with the cells' own backgrounds.
            const int kb = (k - dk) < 0 ? 0 : (k - dk >= n.z ? n.z - 1 : k - dk);
            l = add_background(b, ev.rho0c[kb], ev.p0c[kb]);
            r = add_background(c, ev.rho0c[kc], ev.p0c[kc]);
        }
        float F[kEV];
        hllc(l, r, gamma, F);
        F[1] -= p0;  // the hydrostatic pressure gradient is balanced by the gravity source instead

        const std::int64_t idx = i + std::int64_t(fd.x) * (j + std::int64_t(fd.y) * k);
        flux.c[0][idx] = F[0];
        flux.c[4][idx] = F[4];
        // Rotated momentum (normal, t1, t2) back to x, y, z.
        const int mn = 1 + d;
        const int mt1 = d == 0 ? 2 : 1;
        const int mt2 = d == 2 ? 2 : 3;
        flux.c[mn][idx] = F[1];
        flux.c[mt1][idx] = F[2];
        flux.c[mt2][idx] = F[3];
        for (int s = 0; s < kES; ++s) flux.c[5 + s][idx] = F[5 + s];
    });
}

}  // namespace detail

CompressibleSolver::CompressibleSolver(Device dev, const GridInfo& grid, const Atmosphere& atm,
                                       const BlastParams& params, const DustParams& dust,
                                       const HumidityProfile& humidity)
    : dev_(dev), grid_(grid), params_(params), dust_(dust), humidity_(humidity), ambient_(atm, grid, dev) {
    {
        std::vector<float> q(std::size_t(grid.dims.z), 0.f);
        if (humidity.enabled) {
            for (int k = 0; k < grid.dims.z; ++k) {
                const float z = grid.origin.z + (k + 0.5f) * grid.dx;
                const AtmosphereState a = atm.at(z);
                q[std::size_t(k)] = std::clamp(humidity.relative_humidity_at(z), 0.f, 1.f) *
                                    saturation_humidity(float(a.temperature), float(a.pressure));
            }
        }
        ambient_qv_.allocate(dev, q.size());
        ambient_qv_.upload(q.data(), q.size());
    }
    const std::size_t cells = std::size_t(grid.count());
    const std::size_t faces = max_face_count(grid.dims);
    for (int v = 0; v < kVars; ++v) {
        u_.emplace_back(dev, cells);
        u1_.emplace_back(dev, cells);
        flux_.emplace_back(dev, faces);
    }
    // Hydrostatic background (zeros when the scheme is not well-balanced).
    const std::size_t nz = std::size_t(grid.dims.z);
    std::vector<float> r0c(nz, 0.f), p0c(nz, 0.f), r0f(nz + 1, 0.f), p0f(nz + 1, 0.f);
    if (params.well_balanced) {
        for (std::size_t k = 0; k <= nz; ++k) {
            const AtmosphereState f = atm.at(grid.origin.z + double(k) * grid.dx);
            r0f[k] = float(f.density);
            p0f[k] = float(f.pressure);
            if (k < nz) {
                const AtmosphereState c = atm.at(grid.origin.z + (double(k) + 0.5) * grid.dx);
                r0c[k] = float(c.density);
                p0c[k] = float(c.pressure);
            }
        }
    }
    rho0_c_.allocate(dev, nz);
    p0_c_.allocate(dev, nz);
    rho0_f_.allocate(dev, nz + 1);
    p0_f_.allocate(dev, nz + 1);
    rho0_c_.upload(r0c.data(), nz);
    p0_c_.upload(p0c.data(), nz);
    rho0_f_.upload(r0f.data(), nz + 1);
    p0_f_.upload(p0f.data(), nz + 1);
}

std::size_t CompressibleSolver::memory_bytes() const {
    return std::size_t(kVars) * (u_[0].bytes() + u1_[0].bytes() + flux_[0].bytes());
}

void CompressibleSolver::reset_to_ambient() {
    const Int3 n = grid_.dims;
    FieldView r = var(0), mx = var(1), my = var(2), mz = var(3), e = var(4);
    const float* rho_a = ambient_.density.data();
    const float* p_a = ambient_.pressure.data();
    const float gm1 = params_.gamma - 1.f;
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        r(i, j, k) = rho_a[k];
        mx(i, j, k) = my(i, j, k) = mz(i, j, k) = 0.f;
        e(i, j, k) = p_a[k] / gm1;
    });
    for (int s = 0; s < kScalars; ++s) u_[std::size_t(5 + s)].zero();
    FieldView qv = var(kVapor);
    const float* qa = ambient_qv_.data();
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) { qv(i, j, k) = rho_a[k] * qa[k]; });
}

void CompressibleSolver::set_uniform(float rho, float p) {
    FieldView r = var(0), mx = var(1), my = var(2), mz = var(3), e = var(4);
    const float gm1 = params_.gamma - 1.f;
    parallel_for(dev_, grid_.dims, SIM_LAMBDA(int i, int j, int k) {
        r(i, j, k) = rho;
        mx(i, j, k) = my(i, j, k) = mz(i, j, k) = 0.f;
        e(i, j, k) = p / gm1;
    });
    for (int s = 0; s < kScalars; ++s) u_[std::size_t(5 + s)].zero();
}

void CompressibleSolver::deposit_energy(std::uint32_t seed) {
    const GridInfo g = grid_;
    const float r_dep = params_.deposit_radius, amp = params_.noise_amplitude, freq = params_.noise_frequency;
    // Total weighted volume, so exactly the hydrodynamic share of the yield is deposited.
    const double wsum = parallel_sum(dev_, g.dims, SIM_LAMBDA(int i, int j, int k) {
        const Float3 p = g.cell_center(i, j, k);
        const float d = length(p) / r_dep + amp * noise::fbm(p * freq, 4, 2.0f, 0.5f, seed);
        return double(smoothstepf(1.f, 0.8f, d));
    });
    if (wsum <= 0.0) throw std::runtime_error("blast: deposit_radius is smaller than a voxel");
    const double energy = double(params_.yield_kt) * 4.184e12 * params_.hydro_fraction;
    const float e_density = float(energy / (wsum * double(g.dx) * g.dx * g.dx));
    FieldView r = var(0), e = var(4), smoke = var(5);
    FieldView d0 = var(6), d1 = var(7), d2 = var(8);
    const float load = params_.fireball_dust_loading;
    const float s0 = params_.fireball_dust_split[0] * load, s1 = params_.fireball_dust_split[1] * load,
                s2 = params_.fireball_dust_split[2] * load;
    parallel_for(dev_, g.dims, SIM_LAMBDA(int i, int j, int k) {
        const Float3 p = g.cell_center(i, j, k);
        const float d = length(p) / r_dep + amp * noise::fbm(p * freq, 4, 2.0f, 0.5f, seed);
        const float w = smoothstepf(1.f, 0.8f, d);
        if (w <= 0.f) return;
        const float rho = r(i, j, k);
        e(i, j, k) += w * e_density;
        smoke(i, j, k) = rho * w;
        d0(i, j, k) += rho * w * s0;
        d1(i, j, k) += rho * w * s1;
        d2(i, j, k) += rho * w * s2;
    });
}

void CompressibleSolver::rk_stage(int src_set, int dst_set, float alpha, float beta, float c) {
    const Int3 n = grid_.dims;
    std::vector<Buffer<float>>& src = src_set == 0 ? u_ : u1_;
    detail::EulerView ev;
    for (int v = 0; v < kVars; ++v) ev.u.c[v] = src[std::size_t(v)].data();
    ev.n = n;
    ev.gamma = params_.gamma;
    ev.rho0c = rho0_c_.data();
    ev.p0c = p0_c_.data();
    detail::ConsPointers flux, dst, u0, u1;
    for (int v = 0; v < kVars; ++v) {
        flux.c[v] = flux_[std::size_t(v)].data();
        dst.c[v] = (dst_set == 0 ? u_ : u1_)[std::size_t(v)].data();
        u0.c[v] = u_[std::size_t(v)].data();
        u1.c[v] = u1_[std::size_t(v)].data();
    }
    const float c_dx = c / grid_.dx;
    for (int d = 0; d < 3; ++d) {
        // The flux kernel reads only src; dst may be src's partner set but never src itself
        // while fluxes are still being computed from it.
        detail::compute_fluxes(dev_, ev, flux, rho0_f_.data(), p0_f_.data(), d);
        const Int3 fd{n.x + (d == 0), n.y + (d == 1), n.z + (d == 2)};
        const bool first = d == 0;
        parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
            const std::int64_t cc = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
            const std::int64_t lo = i + std::int64_t(fd.x) * (j + std::int64_t(fd.y) * k);
            const std::int64_t hi = (i + (d == 0)) + std::int64_t(fd.x) * ((j + (d == 1)) + std::int64_t(fd.y) * (k + (d == 2)));
            for (int v = 0; v < detail::kEV; ++v) {
                const float base = first ? alpha * u0.c[v][cc] + beta * u1.c[v][cc] : dst.c[v][cc];
                dst.c[v][cc] = base - (flux.c[v][hi] - flux.c[v][lo]) * c_dx;
            }
        });
    }
    // Gravity: buoyancy of the density perturbation and the work done on rising gas.
    const float g = params_.gravity;
    const float* rho0c = rho0_c_.data();
    const bool balanced = params_.well_balanced;
    const detail::ConsPointers cur = ev.u;
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const std::int64_t cc = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
        const float rho = cur.c[0][cc];
        dst.c[3][cc] -= c * (rho - (balanced ? rho0c[k] : 0.f)) * g;
        dst.c[4][cc] -= c * cur.c[3][cc] * g;
    });
}

namespace detail {

// Keeps density and pressure positive and mass fractions in [0, 0.5].
void apply_floors(Device dev, ConsPointers u, Int3 n, float gamma) {
    parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
        const std::int64_t c = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
        float rho = fmaxf(u.c[0][c], 1e-4f);
        u.c[0][c] = rho;
        const float ke = 0.5f * (u.c[1][c] * u.c[1][c] + u.c[2][c] * u.c[2][c] + u.c[3][c] * u.c[3][c]) / rho;
        const float p_min = 1.f;
        if ((gamma - 1.f) * (u.c[4][c] - ke) < p_min) u.c[4][c] = ke + p_min / (gamma - 1.f);
        for (int s = 0; s < kES; ++s) u.c[5 + s][c] = fminf(fmaxf(u.c[5 + s][c], 0.f), (s == 0 ? 1.f : 0.5f) * rho);
    });
}

}  // namespace detail

void CompressibleSolver::step(double dt_d) {
    const float dt = float(dt_d);
    const Int3 n = grid_.dims;
    detail::ConsPointers u, u1;
    for (int v = 0; v < kVars; ++v) {
        u.c[v] = u_[std::size_t(v)].data();
        u1.c[v] = u1_[std::size_t(v)].data();
    }
    // SSP-RK2 (Heun): u1 = u + dt L(u), then u = (u + u1 + dt L(u1)) / 2.
    profiler_.start();
    rk_stage(0, 1, 1.f, 0.f, dt);
    detail::apply_floors(dev_, u1, n, params_.gamma);
    profiler_.lap("rk1");
    rk_stage(1, 0, 0.5f, 0.5f, 0.5f * dt);
    detail::apply_floors(dev_, u, n, params_.gamma);
    profiler_.lap("rk2");
    apply_sources_after_step(dt);
    profiler_.lap("sources");
}

void CompressibleSolver::apply_sources_after_step(float dt) {
    const GridInfo g = grid_;
    const Int3 n = g.dims;
    detail::ConsPointers u;
    for (int v = 0; v < kVars; ++v) u.c[v] = u_[std::size_t(v)].data();
    const float gamma = params_.gamma, r_gas = phys::kGasConstantAir;
    const float* t_amb = ambient_.temperature.data();
    const float tcap = params_.radiation_temperature_cap;
    const float rad = 3.f * phys::kStefanBoltzmann / params_.radiation_length;
    const float tf = params_.fireball_temperature;
    // Dust lifting by the blast wind (same law as in the low-Mach solver).
    const float ustar_per_u = 0.4f / std::log(std::max(0.5f * g.dx / dust_.roughness_length, 1.01f));
    const float ut = dust_.threshold_friction_velocity;
    const float lift = params_.blast_erosion * dt / g.dx;
    const float sp0 = dust_.erosion_split[0], sp1 = dust_.erosion_split[1], sp2 = dust_.erosion_split[2];
    const bool moist = humidity_.enabled;
    const int iv = kVapor;

    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const std::int64_t c = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
        const float rho = u.c[0][c];
        const float vx = u.c[1][c] / rho, vy = u.c[2][c] / rho, vz = u.c[3][c] / rho;
        const float ke = 0.5f * rho * (vx * vx + vy * vy + vz * vz);
        const float eint = u.c[4][c] - ke;
        const float t = (gamma - 1.f) * eint / (rho * r_gas);
        const float ta = t_amb[k];
        // Radiation from the fireball surface; never cools below ambient or by more than half per step.
        if (t > ta) {
            const float ts = fminf(t, tcap);
            const float e_min = rho * r_gas * fmaxf(ta, 0.5f * t) / (gamma - 1.f);
            const float e_new = fmaxf(eint - rad * ts * ts * ts * ts * dt, e_min);
            u.c[4][c] = ke + e_new;
        }
        // Condensation / evaporation in the closed cell (heat capacity at constant volume).
        if (moist) {
            const float e_now = u.c[4][c] - ke;
            const float tc = (gamma - 1.f) * e_now / (rho * r_gas);
            float qv = u.c[iv][c] / rho, qc = u.c[iv + 1][c] / rho;
            const float cv = r_gas / (gamma - 1.f);
            const float dt_latent = saturation_adjust(tc, rho * r_gas * tc, qv, qc, cv);
            u.c[4][c] += rho * cv * dt_latent;
            u.c[iv][c] = rho * qv;
            u.c[iv + 1][c] = rho * qc;
        }
        // Air that has been hotter than the fireball threshold carries the fireball look (and its NO2).
        const float mark = smoothstepf(0.8f * tf, 1.2f * tf, t);
        u.c[5][c] = fmaxf(u.c[5][c], rho * mark);
        if (k == 0 && lift > 0.f) {
            const float us = ustar_per_u * sqrtf(vx * vx + vy * vy);
            if (us > ut) {
                const float dy = lift * us * (us * us - ut * ut);
                u.c[6][c] = fminf(u.c[6][c] + rho * sp0 * dy, 0.5f * rho);
                u.c[7][c] = fminf(u.c[7][c] + rho * sp1 * dy, 0.5f * rho);
                u.c[8][c] = fminf(u.c[8][c] + rho * sp2 * dy, 0.5f * rho);
            }
        }
    });
}

double CompressibleSolver::suggest_dt() {
    FieldView r = var(0), mx = var(1), my = var(2), mz = var(3), e = var(4);
    const float gamma = params_.gamma;
    const float s = parallel_max(dev_, grid_.dims, SIM_LAMBDA(int i, int j, int k) {
        const float rho = r(i, j, k);
        const float vx = mx(i, j, k) / rho, vy = my(i, j, k) / rho, vz = mz(i, j, k) / rho;
        const float p = (gamma - 1.f) * (e(i, j, k) - 0.5f * rho * (vx * vx + vy * vy + vz * vz));
        const float c = sqrtf(gamma * fmaxf(p, 1.f) / rho);
        return fmaxf(fmaxf(fabsf(vx), fabsf(vy)), fabsf(vz)) + c;
    });
    return params_.cfl * grid_.dx / std::max(s, 1e-3f);
}

namespace detail {

struct CellState {
    float rho, vx, vy, vz, p, t;
};

SIM_INLINE CellState cell_state(FieldView r, FieldView mx, FieldView my, FieldView mz, FieldView e, float gamma,
                                int i, int j, int k) {
    CellState s;
    s.rho = r(i, j, k);
    s.vx = mx(i, j, k) / s.rho;
    s.vy = my(i, j, k) / s.rho;
    s.vz = mz(i, j, k) / s.rho;
    s.p = (gamma - 1.f) * (e(i, j, k) - 0.5f * s.rho * (s.vx * s.vx + s.vy * s.vy + s.vz * s.vz));
    s.t = s.p / (s.rho * phys::kGasConstantAir);
    return s;
}

// Cell velocity from momentum and density, indices clamped to the grid (for central differences).
struct MomentumVelocity {
    FieldView r, mx, my, mz;
    Int3 n;
    SIM_INLINE Float3 at(int i, int j, int k) const {
        i = i < 0 ? 0 : (i >= n.x ? n.x - 1 : i);
        j = j < 0 ? 0 : (j >= n.y ? n.y - 1 : j);
        k = k < 0 ? 0 : (k >= n.z ? n.z - 1 : k);
        const float inv = 1.f / r(i, j, k);
        return make_f3(mx(i, j, k) * inv, my(i, j, k) * inv, mz(i, j, k) * inv);
    }
};

}  // namespace detail

void CompressibleSolver::fill_export(FieldView density, FieldView dust_out, FieldView cond_out, FieldView temperature,
                                     FieldView flame, FieldView vel_x, FieldView vel_y, FieldView vel_z,
                                     float flame_temperature) {
    export_field(ExportField::Density, density, flame_temperature);
    export_field(ExportField::Dust, dust_out, flame_temperature);
    export_field(ExportField::Condensate, cond_out, flame_temperature);
    export_field(ExportField::Temperature, temperature, flame_temperature);
    export_field(ExportField::Flame, flame, flame_temperature);
    export_field(ExportField::VelX, vel_x, flame_temperature);
    export_field(ExportField::VelY, vel_y, flame_temperature);
    export_field(ExportField::VelZ, vel_z, flame_temperature);
}

void CompressibleSolver::export_field_to_host(ExportField f, float* host, float flame_temperature) {
    export_field(f, FieldView{u1_[0].data(), grid_.dims}, flame_temperature);
    u1_[0].download(host, std::size_t(grid_.count()));
}

void CompressibleSolver::export_field(ExportField field, FieldView out, float flame_temperature) {
    FieldView r = var(0), mx = var(1), my = var(2), mz = var(3), e = var(4), smoke = var(5);
    FieldView d0 = var(6), d1 = var(7), d2 = var(8), qc = var(kVapor + 1);
    const Int3 n = grid_.dims;
    const float gamma = params_.gamma;
    const float* t_amb = ambient_.temperature.data();
    const float rho_ground = ambient_.ground_density;
    const float cond_scale = humidity_.enabled ? humidity_.render_scale : 0.f;
    const float inv_tf = 1.f / flame_temperature;
    const float inv_2dx = 0.5f / grid_.dx;
    const float w0 = dust_.optical_weight[0] * dust_.render_scale, w1 = dust_.optical_weight[1] * dust_.render_scale,
                w2 = dust_.optical_weight[2] * dust_.render_scale;
    const int which = int(field);
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const detail::CellState s = detail::cell_state(r, mx, my, mz, e, gamma, i, j, k);
        const float ta = t_amb[k];
        // Half-float storage tops out at 65504.
        const float t = fminf(s.t, 60000.f);
        const float optical_dust = (w0 * d0(i, j, k) + w1 * d1(i, j, k) + w2 * d2(i, j, k)) / rho_ground;
        const float sm = smoke(i, j, k);
        const float cloud = cond_scale * qc(i, j, k) / rho_ground;
        const bool hot = t > ta + 1.f;
        const bool active = hot || sm > 1e-4f * s.rho || optical_dust > 1e-4f || cloud > 1e-4f;
        float v = 0.f;
        switch (ExportField(which)) {
            case ExportField::Density: v = sm / rho_ground; break;
            case ExportField::Dust: v = optical_dust; break;
            case ExportField::Condensate: v = cloud; break;
            case ExportField::Temperature: v = active ? t : 0.f; break;
            case ExportField::Flame: {
                const float x = t * inv_tf, xa = ta * inv_tf;
                v = hot ? fminf(fmaxf(x * x * x * x - xa * xa * xa * xa, 0.f), 60000.f) : 0.f;
                break;
            }
            case ExportField::VelX: v = active ? s.vx : 0.f; break;
            case ExportField::VelY: v = active ? s.vy : 0.f; break;
            case ExportField::VelZ: v = active ? s.vz : 0.f; break;
            case ExportField::Vorticity:
                if (active) {
                    const detail::MomentumVelocity vel{r, mx, my, mz, n};
                    const Float3 xp = vel.at(i + 1, j, k), xm = vel.at(i - 1, j, k);
                    const Float3 yp = vel.at(i, j + 1, k), ym = vel.at(i, j - 1, k);
                    const Float3 zp = vel.at(i, j, k + 1), zm = vel.at(i, j, k - 1);
                    const Float3 om{((yp.z - ym.z) - (zp.y - zm.y)) * inv_2dx, ((zp.x - zm.x) - (xp.z - xm.z)) * inv_2dx,
                                    ((xp.y - xm.y) - (yp.x - ym.x)) * inv_2dx};
                    v = length(om);
                }
                break;
            default: break;
        }
        out(i, j, k) = v;
    });
}

CompressibleSolver::HandoffState CompressibleSolver::handoff_state() {
    // Use the second RK set (free between steps) as scratch: temperature, smoke, u, v, w, dust bins.
    FieldView r = var(0), mx = var(1), my = var(2), mz = var(3), e = var(4), smoke = var(5);
    FieldView d0 = var(6), d1 = var(7), d2 = var(8);
    const Int3 n = grid_.dims;
    FieldView ot{u1_[0].data(), n}, os{u1_[1].data(), n}, ou{u1_[2].data(), n}, ov{u1_[3].data(), n},
        ow{u1_[4].data(), n}, o0{u1_[5].data(), n}, o1{u1_[6].data(), n}, o2{u1_[7].data(), n},
        oqv{u1_[8].data(), n}, oqc{u1_[9].data(), n};
    FieldView qv = var(kVapor), qc = var(kVapor + 1);
    const float gamma = params_.gamma;
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const detail::CellState s = detail::cell_state(r, mx, my, mz, e, gamma, i, j, k);
        ot(i, j, k) = s.t;
        os(i, j, k) = smoke(i, j, k) / s.rho;
        ou(i, j, k) = s.vx;
        ov(i, j, k) = s.vy;
        ow(i, j, k) = s.vz;
        o0(i, j, k) = d0(i, j, k) / s.rho;
        o1(i, j, k) = d1(i, j, k) / s.rho;
        o2(i, j, k) = d2(i, j, k) / s.rho;
        oqv(i, j, k) = qv(i, j, k) / s.rho;
        oqc(i, j, k) = qc(i, j, k) / s.rho;
    });
    HandoffState h;
    h.temperature = u1_[0].to_host();
    h.smoke = u1_[1].to_host();
    h.u = u1_[2].to_host();
    h.v = u1_[3].to_host();
    h.w = u1_[4].to_host();
    for (int b = 0; b < kDustBins; ++b) h.dust[b] = u1_[std::size_t(5 + b)].to_host();
    h.vapor = u1_[8].to_host();
    h.condensate = u1_[9].to_host();
    return h;
}

float CompressibleSolver::max_temperature() {
    FieldView r = var(0), mx = var(1), my = var(2), mz = var(3), e = var(4);
    const float gamma = params_.gamma;
    return parallel_max(dev_, grid_.dims, SIM_LAMBDA(int i, int j, int k) {
        return detail::cell_state(r, mx, my, mz, e, gamma, i, j, k).t;
    });
}

float CompressibleSolver::max_speed() {
    FieldView r = var(0), mx = var(1), my = var(2), mz = var(3);
    return parallel_max(dev_, grid_.dims, SIM_LAMBDA(int i, int j, int k) {
        const float m = sqrtf(mx(i, j, k) * mx(i, j, k) + my(i, j, k) * my(i, j, k) + mz(i, j, k) * mz(i, j, k));
        return m / r(i, j, k);
    });
}

float CompressibleSolver::max_pressure_ratio() {
    FieldView r = var(0), mx = var(1), my = var(2), mz = var(3), e = var(4);
    const float gamma = params_.gamma;
    const float* p_amb = ambient_.pressure.data();
    return parallel_max(dev_, grid_.dims, SIM_LAMBDA(int i, int j, int k) {
        return detail::cell_state(r, mx, my, mz, e, gamma, i, j, k).p / p_amb[k];
    });
}

double CompressibleSolver::total_energy() {
    FieldView e = var(4);
    const double cell = double(grid_.dx) * grid_.dx * grid_.dx;
    return cell * parallel_sum(dev_, grid_.dims, SIM_LAMBDA(int i, int j, int k) { return double(e(i, j, k)); });
}

float CompressibleSolver::shock_radius() {
    FieldView r = var(0), mx = var(1), my = var(2), mz = var(3), e = var(4);
    const float gamma = params_.gamma;
    const float* p_amb = ambient_.pressure.data();
    const GridInfo g = grid_;
    return parallel_max(dev_, Int3{g.dims.x, g.dims.y, 1}, SIM_LAMBDA(int i, int j, int) {
        const detail::CellState s = detail::cell_state(r, mx, my, mz, e, gamma, i, j, 0);
        if (s.p < 1.05f * p_amb[0]) return 0.f;
        const Float3 c = g.cell_center(i, j, 0);
        return sqrtf(c.x * c.x + c.y * c.y);
    });
}

}  // namespace sim
