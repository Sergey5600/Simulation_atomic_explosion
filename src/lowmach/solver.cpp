#include "lowmach/solver.h"

#include <algorithm>
#include <cmath>

#include "core/parallel.h"
#include "physics/constants.h"

namespace sim {

namespace detail {

// Cell-centred velocity from the faces around cell (i, j, k), indices clamped to the grid.
struct CellVelocity {
    FieldView u, v, w;
    Int3 n;

    SIM_INLINE Float3 at(int i, int j, int k) const {
        i = i < 0 ? 0 : (i >= n.x ? n.x - 1 : i);
        j = j < 0 ? 0 : (j >= n.y ? n.y - 1 : j);
        k = k < 0 ? 0 : (k >= n.z ? n.z - 1 : k);
        return {0.5f * (u(i, j, k) + u(i + 1, j, k)), 0.5f * (v(i, j, k) + v(i, j + 1, k)),
                0.5f * (w(i, j, k) + w(i, j, k + 1))};
    }
};

// Vorticity confinement force (Fedkiw et al. 2001) at a cell, from precomputed vorticity.
struct Confinement {
    FieldView ox, oy, oz;
    Int3 n;
    float eps_dx;   // epsilon * dx
    float inv_2dx;
    float inv_dx;

    SIM_INLINE float mag(int i, int j, int k) const {
        i = i < 0 ? 0 : (i >= n.x ? n.x - 1 : i);
        j = j < 0 ? 0 : (j >= n.y ? n.y - 1 : j);
        k = k < 0 ? 0 : (k >= n.z ? n.z - 1 : k);
        const float a = ox(i, j, k), b = oy(i, j, k), c = oz(i, j, k);
        return sqrtf(a * a + b * b + c * c);
    }

    SIM_INLINE Float3 force(int i, int j, int k) const {
        i = i < 0 ? 0 : (i >= n.x ? n.x - 1 : i);
        j = j < 0 ? 0 : (j >= n.y ? n.y - 1 : j);
        k = k < 0 ? 0 : (k >= n.z ? n.z - 1 : k);
        const Float3 om{ox(i, j, k), oy(i, j, k), oz(i, j, k)};
        const float m = length(om);
        const Float3 grad{(mag(i + 1, j, k) - mag(i - 1, j, k)) * inv_2dx, (mag(i, j + 1, k) - mag(i, j - 1, k)) * inv_2dx,
                          (mag(i, j, k + 1) - mag(i, j, k - 1)) * inv_2dx};
        // The floor keeps smooth shear flow (e.g. the wind profile) free of spurious forces.
        const float floor = 0.1f * m * inv_dx;
        const float norm = sqrtf(dot(grad, grad) + floor * floor);
        if (norm <= 0.f) return {0.f, 0.f, 0.f};
        const Float3 N = grad / norm;
        return Float3{N.y * om.z - N.z * om.y, N.z * om.x - N.x * om.z, N.x * om.y - N.y * om.x} * eps_dx;
    }
};

}  // namespace detail

LowMachSolver::LowMachSolver(Device dev, const GridInfo& grid, const Atmosphere& atm, const SolverParams& params,
                             const RadiationParams& radiation)
    : dev_(dev),
      grid_(grid),
      params_(params),
      radiation_(radiation),
      ambient_(atm, grid, dev),
      theta_(dev, std::size_t(grid.count())),
      smoke_(dev, std::size_t(grid.count())),
      scalar_tmp_(dev, std::size_t(grid.count())),
      rho_(dev, std::size_t(grid.count())),
      source_(dev, std::size_t(grid.count())),
      omega_x_(dev, std::size_t(grid.count())),
      omega_y_(dev, std::size_t(grid.count())),
      omega_z_(dev, std::size_t(grid.count())),
      vel_(grid.dims, dev),
      vel_a_(grid.dims, dev),
      vel_b_(grid.dims, dev),
      scratch_(grid.dims, dev),
      pressure_(dev, grid.dims, grid.dx) {}

std::size_t LowMachSolver::memory_bytes() const {
    const std::size_t cells = 8 * theta_.bytes();
    const std::size_t faces = 3 * (vel_.u.bytes() + vel_.v.bytes() + vel_.w.bytes()) + scratch_.a.bytes() + scratch_.b.bytes();
    return cells + faces + pressure_.memory_bytes();
}

void LowMachSolver::reset_to_ambient(const WindProfile& wind) {
    const Int3 n = grid_.dims;
    const std::size_t nz = std::size_t(n.z);
    // Wind per level, evaluated on the host.
    std::vector<float> wu(nz), wv(nz);
    const float heading = wind.heading_deg * 3.14159265f / 180.f;
    for (std::size_t k = 0; k < nz; ++k) {
        const float s = wind.speed_at(grid_.origin.z + (float(k) + 0.5f) * grid_.dx);
        wu[k] = s * std::cos(heading);
        wv[k] = s * std::sin(heading);
    }
    Buffer<float> wind_u(dev_, nz), wind_v(dev_, nz);
    wind_u.upload(wu.data(), nz);
    wind_v.upload(wv.data(), nz);
    const float* pu = wind_u.data();
    const float* pv = wind_v.data();
    const float* th_a = ambient_.theta.data();

    FieldView th = theta(), sm = smoke();
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        th(i, j, k) = th_a[k];
        sm(i, j, k) = 0.f;
    });
    FieldView u = vel_.u_view(), v = vel_.v_view(), w = vel_.w_view();
    parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) { u(i, j, k) = pu[k]; });
    parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) { v(i, j, k) = pv[k]; });
    parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) { w(i, j, k) = 0.f; });
    device_synchronize(dev_);  // the wind buffers go out of scope
}

void LowMachSolver::cool_and_build_source(float dt) {
    const Int3 n = grid_.dims;
    FieldView th = theta(), src{source_.data(), n};
    FieldView w = vel_.w_view();
    const float* exner = ambient_.exner.data();
    const float* t_amb = ambient_.temperature.data();
    const float* p_amb = ambient_.pressure.data();

    const float m = 4.f + radiation_.exponent;
    const float t_ref = radiation_.reference_temperature;
    // lambda * p0 for dx/dt = -lambda x^(m+1), x = T / T_ref, lambda = 4 sigma k0 R T_ref^4 / (p0 c_p).
    const float lambda_p = 4.f * phys::kStefanBoltzmann * radiation_.absorption * phys::kGasConstantAir *
                           (t_ref * t_ref * t_ref * t_ref) / phys::kHeatCapacityAir;
    const float expansion = phys::kGravity / (phys::kGamma * phys::kGasConstantAir);

    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const float ta = t_amb[k];
        const float t = th(i, j, k) * exner[k];
        float s = 0.f;
        if (t > ta && lambda_p > 0.f) {
            const float x = t / t_ref;
            const float lambda = lambda_p / p_amb[k];
            // Exact solution of the power-law cooling over dt (unconditionally stable).
            float t_new = t * powf(1.f + m * lambda * powf(x, m) * dt, -1.f / m);
            t_new = fmaxf(t_new, ta);
            th(i, j, k) = t_new / exner[k];
            s = logf(t_new / t) / dt;
        }
        const float wc = 0.5f * (w(i, j, k) + w(i, j, k + 1));
        src(i, j, k) = s + wc * expansion / ta;
    });
}

void LowMachSolver::update_density() {
    const Int3 n = grid_.dims;
    FieldView th = theta(), rho{rho_.data(), n};
    const float* exner = ambient_.exner.data();
    const float* p_amb = ambient_.pressure.data();
    const float r_gas = phys::kGasConstantAir;
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        rho(i, j, k) = p_amb[k] / (r_gas * th(i, j, k) * exner[k]);
    });
    pressure_.set_density(rho);
}

void LowMachSolver::add_forces(MacField& field, float dt) {
    const Int3 n = grid_.dims;
    FieldView th = theta();
    FieldView u = field.u_view(), v = field.v_view(), w = field.w_view();
    const float* th_a = ambient_.theta.data();
    const float g = phys::kGravity;

    // Buoyancy relative to the ambient column, averaged from the two cells around each z-face.
    parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) {
        if (k == 0) return;  // ground
        const int kb = k - 1, kt = k < n.z ? k : n.z - 1;
        const float bb = (th(i, j, kb) - th_a[kb]) / th_a[kb];
        const float bt = (th(i, j, kt) - th_a[kt]) / th_a[kt];
        w(i, j, k) += dt * g * 0.5f * (bb + bt);
    });

    if (params_.vorticity_confinement <= 0.f) return;
    const float inv_2dx = 0.5f / grid_.dx;
    const detail::CellVelocity cv{u, v, w, n};
    FieldView ox{omega_x_.data(), n}, oy{omega_y_.data(), n}, oz{omega_z_.data(), n};
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 xp = cv.at(i + 1, j, k), xm = cv.at(i - 1, j, k);
        const Float3 yp = cv.at(i, j + 1, k), ym = cv.at(i, j - 1, k);
        const Float3 zp = cv.at(i, j, k + 1), zm = cv.at(i, j, k - 1);
        ox(i, j, k) = ((yp.z - ym.z) - (zp.y - zm.y)) * inv_2dx;
        oy(i, j, k) = ((zp.x - zm.x) - (xp.z - xm.z)) * inv_2dx;
        oz(i, j, k) = ((xp.y - xm.y) - (yp.x - ym.x)) * inv_2dx;
    });
    const detail::Confinement conf{ox, oy, oz, n, params_.vorticity_confinement * grid_.dx, inv_2dx, 1.f / grid_.dx};
    parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) {
        u(i, j, k) += dt * 0.5f * (conf.force(i - 1, j, k).x + conf.force(i, j, k).x);
    });
    parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) {
        v(i, j, k) += dt * 0.5f * (conf.force(i, j - 1, k).y + conf.force(i, j, k).y);
    });
    parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) {
        if (k == 0) return;
        w(i, j, k) += dt * 0.5f * (conf.force(i, j, k - 1).z + conf.force(i, j, k).z);
    });
}

void LowMachSolver::advect_scalars(const MacView& vel, float dt) {
    const Int3 n = grid_.dims;
    advect_maccormack(dev_, vel, dt, grid_.dx, SampleView{theta_.data(), n, kCellOffset}, {scalar_tmp_.data(), n},
                      scratch_);
    std::swap(theta_, scalar_tmp_);
    advect_maccormack(dev_, vel, dt, grid_.dx, SampleView{smoke_.data(), n, kCellOffset}, {scalar_tmp_.data(), n},
                      scratch_);
    std::swap(smoke_, scalar_tmp_);
}

namespace detail {

SIM_INLINE float clamped(FieldView f, int i, int j, int k) {
    i = i < 0 ? 0 : (i >= f.dims.x ? f.dims.x - 1 : i);
    j = j < 0 ? 0 : (j >= f.dims.y ? f.dims.y - 1 : j);
    k = k < 0 ? 0 : (k >= f.dims.z ? f.dims.z - 1 : k);
    return f(i, j, k);
}

// dst = src + c * div(nu grad src) for a cell field, zero flux through every boundary.
void diffuse_cells(Device dev, FieldView src, FieldView dst, FieldView nu, float c) {
    const Int3 n = src.dims;
    parallel_for(dev, n, SIM_LAMBDA(int i, int j, int k) {
        const float v = src(i, j, k), nc = nu(i, j, k);
        float flux = 0.f;
        if (i > 0) flux += 0.5f * (nc + nu(i - 1, j, k)) * (src(i - 1, j, k) - v);
        if (i < n.x - 1) flux += 0.5f * (nc + nu(i + 1, j, k)) * (src(i + 1, j, k) - v);
        if (j > 0) flux += 0.5f * (nc + nu(i, j - 1, k)) * (src(i, j - 1, k) - v);
        if (j < n.y - 1) flux += 0.5f * (nc + nu(i, j + 1, k)) * (src(i, j + 1, k) - v);
        if (k > 0) flux += 0.5f * (nc + nu(i, j, k - 1)) * (src(i, j, k - 1) - v);
        if (k < n.z - 1) flux += 0.5f * (nc + nu(i, j, k + 1)) * (src(i, j, k + 1) - v);
        dst(i, j, k) = v + c * flux;
    });
}

// dst = src + c * nu_face * laplacian(src) for one staggered velocity component
// (axis 0, 1, 2 = u, v, w); nu is cell centred, n the cell counts.
void diffuse_faces(Device dev, FieldView src, FieldView dst, FieldView nu, Int3 n, int axis, float c) {
    parallel_for(dev, src.dims, SIM_LAMBDA(int i, int j, int k) {
        if (axis == 2 && k == 0) {  // the ground stays impermeable
            dst(i, j, k) = 0.f;
            return;
        }
        const int ia = axis == 0 ? (i > 0 ? i - 1 : 0) : i;
        const int ja = axis == 1 ? (j > 0 ? j - 1 : 0) : j;
        const int ka = axis == 2 ? (k > 0 ? k - 1 : 0) : k;
        const int ib = axis == 0 ? (i < n.x ? i : n.x - 1) : i;
        const int jb = axis == 1 ? (j < n.y ? j : n.y - 1) : j;
        const int kb = axis == 2 ? (k < n.z ? k : n.z - 1) : k;
        const float nf = 0.5f * (nu(ia, ja, ka) + nu(ib, jb, kb));
        const float v = src(i, j, k);
        const float lap = clamped(src, i - 1, j, k) + clamped(src, i + 1, j, k) + clamped(src, i, j - 1, k) +
                          clamped(src, i, j + 1, k) + clamped(src, i, j, k - 1) + clamped(src, i, j, k + 1) - 6.f * v;
        dst(i, j, k) = v + c * nf * lap;
    });
}

}  // namespace detail

// Smagorinsky eddy viscosity nu = (C_s dx)^2 |S|, applied as explicit diffusion to the
// velocity, theta and smoke. It stands in for the turbulent mixing of hot cloud gas with
// entrained air that the grid cannot resolve.
void LowMachSolver::apply_turbulence(MacField& field, float dt) {
    if (params_.smagorinsky <= 0.f) return;
    const Int3 n = grid_.dims;
    const float dx = grid_.dx;
    const float inv_2dx = 0.5f / dx;
    const float cs_dx2 = (params_.smagorinsky * dx) * (params_.smagorinsky * dx);
    const float nu_max = 0.9f * dx * dx / (6.f * dt);  // explicit stability limit
    const detail::CellVelocity cv{field.u_view(), field.v_view(), field.w_view(), n};
    FieldView nu{omega_x_.data(), n};  // the vorticity scratch is free outside add_forces

    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 dxp = cv.at(i + 1, j, k), dxm = cv.at(i - 1, j, k);
        const Float3 dyp = cv.at(i, j + 1, k), dym = cv.at(i, j - 1, k);
        const Float3 dzp = cv.at(i, j, k + 1), dzm = cv.at(i, j, k - 1);
        const float uxx = (dxp.x - dxm.x) * inv_2dx, uyy = (dyp.y - dym.y) * inv_2dx, uzz = (dzp.z - dzm.z) * inv_2dx;
        const float sxy = 0.5f * ((dyp.x - dym.x) + (dxp.y - dxm.y)) * inv_2dx;
        const float sxz = 0.5f * ((dzp.x - dzm.x) + (dxp.z - dxm.z)) * inv_2dx;
        const float syz = 0.5f * ((dzp.y - dzm.y) + (dyp.z - dym.z)) * inv_2dx;
        const float s2 = uxx * uxx + uyy * uyy + uzz * uzz + 2.f * (sxy * sxy + sxz * sxz + syz * syz);
        nu(i, j, k) = fminf(cs_dx2 * sqrtf(2.f * s2), nu_max);
    });

    const float c_scalar = dt / (dx * dx * params_.turbulent_prandtl);
    detail::diffuse_cells(dev_, theta(), {scalar_tmp_.data(), n}, nu, c_scalar);
    std::swap(theta_, scalar_tmp_);
    detail::diffuse_cells(dev_, smoke(), {scalar_tmp_.data(), n}, nu, c_scalar);
    std::swap(smoke_, scalar_tmp_);

    const float c_mom = dt / (dx * dx);
    detail::diffuse_faces(dev_, field.u_view(), vel_b_.u_view(), nu, n, 0, c_mom);
    detail::diffuse_faces(dev_, field.v_view(), vel_b_.v_view(), nu, n, 1, c_mom);
    detail::diffuse_faces(dev_, field.w_view(), vel_b_.w_view(), nu, n, 2, c_mom);
    field.swap(vel_b_);
}

ProjectionStats LowMachSolver::project(MacField& field) {
    return pressure_.project(field, FieldView{source_.data(), grid_.dims}, params_.pressure_tolerance,
                             params_.pressure_max_iterations);
}

StepStats LowMachSolver::step(double dt_d) {
    const float dt = float(dt_d);
    const Int3 n = grid_.dims;
    StepStats stats;
    stats.dt = dt_d;
    auto record = [&](const ProjectionStats& p) {
        ++stats.projections;
        stats.pcg_iterations += p.iterations;
        stats.max_residual = std::max(stats.max_residual, p.final_residual);
    };

    cool_and_build_source(dt);
    update_density();

    if (params_.reflection) {
        const float h = 0.5f * dt;
        // Half step: advect, force, project.
        advect_velocity(dev_, vel_.sampler(), h, grid_.dx, vel_, vel_a_, scratch_);
        add_forces(vel_a_, h);
        FieldView ua = vel_a_.u_view(), va = vel_a_.v_view(), wa = vel_a_.w_view();
        FieldView ub = vel_b_.u_view(), vb = vel_b_.v_view(), wb = vel_b_.w_view();
        parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) { ub(i, j, k) = ua(i, j, k); });
        parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) { vb(i, j, k) = va(i, j, k); });
        parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) { wb(i, j, k) = wa(i, j, k); });
        record(project(vel_a_));
        // Reflect the pressure correction: u_hat = 2 u_proj - u_unproj.
        parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) { ub(i, j, k) = 2.f * ua(i, j, k) - ub(i, j, k); });
        parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) { vb(i, j, k) = 2.f * va(i, j, k) - vb(i, j, k); });
        parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) { wb(i, j, k) = 2.f * wa(i, j, k) - wb(i, j, k); });
        // Second half: advect the reflected field with the projected midpoint velocity.
        const MacView mid = vel_a_.sampler();
        advect_velocity(dev_, mid, h, grid_.dx, vel_b_, vel_, scratch_);
        advect_scalars(mid, dt);
        apply_turbulence(vel_, dt);
        update_density();
        add_forces(vel_, h);
        record(project(vel_));
    } else {
        advect_scalars(vel_.sampler(), dt);
        advect_velocity(dev_, vel_.sampler(), dt, grid_.dx, vel_, vel_a_, scratch_);
        apply_turbulence(vel_a_, dt);
        update_density();
        add_forces(vel_a_, dt);
        record(project(vel_a_));
        vel_.swap(vel_a_);
    }
    return stats;
}

float LowMachSolver::max_speed() {
    const Int3 n = grid_.dims;
    const detail::CellVelocity cv{vel_.u_view(), vel_.v_view(), vel_.w_view(), n};
    return parallel_max(dev_, n, SIM_LAMBDA(int i, int j, int k) { return length(cv.at(i, j, k)); });
}

float LowMachSolver::max_temperature() {
    FieldView th = theta();
    const float* exner = ambient_.exner.data();
    return parallel_max(dev_, grid_.dims, SIM_LAMBDA(int i, int j, int k) { return th(i, j, k) * exner[k]; });
}

CloudDiagnostics LowMachSolver::diagnostics() {
    const Int3 n = grid_.dims;
    const GridInfo g = grid_;
    FieldView sm = smoke();
    const detail::CellVelocity cv{vel_.u_view(), vel_.v_view(), vel_.w_view(), n};
    CloudDiagnostics d;
    d.top = parallel_max(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        return sm(i, j, k) > 0.05f ? g.origin.z + (k + 1) * g.dx : 0.f;
    });
    const double mass = parallel_sum(dev_, n, SIM_LAMBDA(int i, int j, int k) { return double(sm(i, j, k)); });
    const double moment = parallel_sum(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        return double(sm(i, j, k)) * (g.origin.z + (k + 0.5) * g.dx);
    });
    d.centroid_z = mass > 0.0 ? float(moment / mass) : 0.f;
    d.max_speed = max_speed();
    const float threshold = 0.999f * d.max_speed;
    d.max_speed_z = parallel_max(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        return length(cv.at(i, j, k)) >= threshold ? g.origin.z + (k + 0.5f) * g.dx : 0.f;
    });
    return d;
}

double LowMachSolver::suggest_dt() {
    FieldView th = theta();
    const float* th_a = ambient_.theta.data();
    const float max_b = phys::kGravity * parallel_max(dev_, grid_.dims, SIM_LAMBDA(int i, int j, int k) {
                            return fabsf(th(i, j, k) - th_a[k]) / th_a[k];
                        });
    const double umax = max_speed();
    double dt = params_.max_dt;
    if (umax > 1e-6) dt = std::min(dt, params_.cfl * grid_.dx / umax);
    // A parcel accelerating from rest moves at most cfl cells in one step.
    if (max_b > 1e-6) dt = std::min(dt, std::sqrt(2.0 * params_.cfl * grid_.dx / max_b));
    return dt;
}

void LowMachSolver::fill_export(FieldView density, FieldView temperature, FieldView flame, FieldView vel_x,
                                FieldView vel_y, FieldView vel_z, float flame_temperature) {
    const Int3 n = grid_.dims;
    FieldView th = theta(), sm = smoke();
    const float* exner = ambient_.exner.data();
    const float* t_amb = ambient_.temperature.data();
    const float* p_amb = ambient_.pressure.data();
    const float rho_ground = ambient_.ground_density;
    const float r_gas = phys::kGasConstantAir;
    const float inv_tf = 1.f / flame_temperature;
    const detail::CellVelocity cv{vel_.u_view(), vel_.v_view(), vel_.w_view(), n};

    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const float t = th(i, j, k) * exner[k];
        const float ta = t_amb[k];
        const float s = sm(i, j, k);
        const float rho = p_amb[k] / (r_gas * t);
        const bool hot = t > ta + 1.f;
        const bool active = hot || s > 1e-4f;
        density(i, j, k) = s * rho / rho_ground;
        temperature(i, j, k) = active ? t : 0.f;
        const float x = t * inv_tf, xa = ta * inv_tf;
        flame(i, j, k) = hot ? fmaxf(x * x * x * x - xa * xa * xa * xa, 0.f) : 0.f;
        const Float3 vc = cv.at(i, j, k);
        vel_x(i, j, k) = active ? vc.x : 0.f;
        vel_y(i, j, k) = active ? vc.y : 0.f;
        vel_z(i, j, k) = active ? vc.z : 0.f;
    });
}

}  // namespace sim
