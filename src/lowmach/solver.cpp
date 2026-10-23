#include "lowmach/solver.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/parallel.h"
#include "physics/constants.h"
#include "physics/humidity.h"

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

// Sum of the dust mixing ratios at a linear cell index.
struct DustSum {
    const float* bin[kDustBins];
    int count;
    SIM_INLINE float at(std::int64_t idx) const {
        float s = 0.f;
        for (int b = 0; b < count; ++b) s += bin[b][idx];
        return s;
    }
};

}  // namespace detail

namespace {

// Ambient specific humidity per cell level from the relative humidity profile.
std::vector<float> ambient_humidity(const HumidityProfile& h, const Atmosphere& atm, const GridInfo& g) {
    std::vector<float> q(std::size_t(g.dims.z), 0.f);
    if (!h.enabled) return q;
    for (int k = 0; k < g.dims.z; ++k) {
        const float z = g.origin.z + (k + 0.5f) * g.dx;
        const AtmosphereState a = atm.at(z);
        const float rh = std::clamp(h.relative_humidity_at(z), 0.f, 1.f);
        q[std::size_t(k)] = rh * saturation_humidity(float(a.temperature), float(a.pressure));
    }
    return q;
}

}  // namespace

LowMachSolver::LowMachSolver(Device dev, const GridInfo& grid, const Atmosphere& atm, const SolverParams& params,
                             const RadiationParams& radiation, const DustParams& dust, const HumidityProfile& humidity)
    : dev_(dev),
      grid_(grid),
      params_(params),
      radiation_(radiation),
      dust_(dust),
      atmosphere_(atm),
      ambient_(atm, grid, dev),
      tmp_a_(dev, std::size_t(grid.count())),
      tmp_b_(dev, std::size_t(grid.count())),
      rho_(dev, std::size_t(grid.count())),
      source_(dev, std::size_t(grid.count())),
      vel_(grid.dims, dev),
      vel_a_(grid.dims, dev),
      scratch_(grid.dims, dev),
      pressure_(dev, grid.dims, grid.dx) {
    humidity_ = humidity;
    int count = kDust0 + (dust_.enabled ? kDustBins : 0);
    if (humidity_.enabled) {
        vapor_index_ = count;
        count += 2;
        const std::vector<float> q = ambient_humidity(humidity_, atmosphere_, grid_);
        ambient_qv_.allocate(dev, q.size());
        ambient_qv_.upload(q.data(), q.size());
    }
    for (int i = 0; i < count; ++i) scalars_.emplace_back(dev, std::size_t(grid.count()));
}

LowMachSolver::MoisturePointers LowMachSolver::moisture_pointers() {
    MoisturePointers m;
    if (!humidity_.enabled) return m;
    m.qv = scalars_[std::size_t(vapor_index_)].data();
    m.qc = scalars_[std::size_t(vapor_index_ + 1)].data();
    m.qv_ambient = ambient_qv_.data();
    return m;
}

LowMachSolver::DustPointers LowMachSolver::dust_pointers() {
    DustPointers d;
    if (!dust_.enabled) return d;
    d.count = kDustBins;
    for (int b = 0; b < kDustBins; ++b) d.bin[b] = scalars_[std::size_t(kDust0 + b)].data();
    return d;
}

std::size_t LowMachSolver::memory_bytes() const {
    const std::size_t cells = (scalars_.size() + 4) * rho_.bytes();
    const std::size_t faces = vel_.memory_bytes() + vel_a_.memory_bytes() + scratch_.memory_bytes();
    return cells + faces + pressure_.memory_bytes();
}

namespace {

// Ambient wind components per cell level of a grid, uploaded to the device.
void upload_wind(const WindProfile& wind, const GridInfo& g, Device dev, Buffer<float>& wu_d, Buffer<float>& wv_d) {
    const std::size_t nz = std::size_t(g.dims.z);
    std::vector<float> wu(nz), wv(nz);
    const float heading = wind.heading_deg * 3.14159265f / 180.f;
    for (std::size_t k = 0; k < nz; ++k) {
        const float s = wind.speed_at(g.origin.z + (float(k) + 0.5f) * g.dx);
        wu[k] = s * std::cos(heading);
        wv[k] = s * std::sin(heading);
    }
    wu_d.allocate(dev, nz);
    wv_d.allocate(dev, nz);
    wu_d.upload(wu.data(), nz);
    wv_d.upload(wv.data(), nz);
}

}  // namespace

void LowMachSolver::reset_to_ambient(const WindProfile& wind) {
    const Int3 n = grid_.dims;
    wind_ = wind;
    upload_wind(wind_, grid_, dev_, wind_u_, wind_v_);
    const float* pu = wind_u_.data();
    const float* pv = wind_v_.data();
    const float* th_a = ambient_.theta.data();

    FieldView th = theta(), sm = smoke();
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        th(i, j, k) = th_a[k];
        sm(i, j, k) = 0.f;
    });
    for (std::size_t b = kDust0; b < scalars_.size(); ++b) scalars_[b].zero();
    if (humidity_.enabled) {
        FieldView qv = vapor();
        const float* qa = ambient_qv_.data();
        parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) { qv(i, j, k) = qa[k]; });
    }
    FieldView u = vel_.u_view(), v = vel_.v_view(), w = vel_.w_view();
    parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) { u(i, j, k) = pu[k]; });
    parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) { v(i, j, k) = pv[k]; });
    parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) { w(i, j, k) = 0.f; });
}

void LowMachSolver::coarsen_domain() {
    const Int3 n = grid_.dims;
    if (!can_coarsen(n)) throw std::runtime_error("coarsen_domain: grid x, y must be divisible by 4 and z by 2");
    GridInfo ng = grid_;
    ng.dx = 2.f * grid_.dx;
    const float cx = grid_.origin.x + 0.5f * n.x * grid_.dx, cy = grid_.origin.y + 0.5f * n.y * grid_.dx;
    ng.origin = Float3{cx - 0.5f * n.x * ng.dx, cy - 0.5f * n.y * ng.dx, grid_.origin.z};
    // The old domain becomes new cells [ox, ox + n.x/2) x [oy, oy + n.y/2) x [0, n.z/2).
    const int ox = n.x / 4, oy = n.y / 4;
    const int hx = n.x / 2, hy = n.y / 2, hz = n.z / 2;

    AtmosphereColumns new_ambient(atmosphere_, ng, dev_);
    Buffer<float> new_wu, new_wv;
    upload_wind(wind_, ng, dev_, new_wu, new_wv);
    Buffer<float> new_qv;
    if (humidity_.enabled) {
        const std::vector<float> q = ambient_humidity(humidity_, atmosphere_, ng);
        new_qv.allocate(dev_, q.size());
        new_qv.upload(q.data(), q.size());
    }

    // Cell fields: average of the 8 old cells, ambient outside.
    for (std::size_t s = 0; s < scalars_.size(); ++s) {
        const FieldView src{scalars_[s].data(), n};
        FieldView dst{tmp_a_.data(), n};
        const float* amb = s == std::size_t(kTheta)                                  ? new_ambient.theta.data()
                           : (humidity_.enabled && int(s) == vapor_index_) ? new_qv.data()
                                                                            : nullptr;
        parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
            const int a = i - ox, b = j - oy, c = k;
            if (a >= 0 && a < hx && b >= 0 && b < hy && c < hz) {
                float sum = 0.f;
                for (int q = 0; q < 8; ++q) sum += src(2 * a + (q & 1), 2 * b + ((q >> 1) & 1), 2 * c + ((q >> 2) & 1));
                dst(i, j, k) = 0.125f * sum;
            } else {
                dst(i, j, k) = amb ? amb[k] : 0.f;
            }
        });
        std::swap(scalars_[s], tmp_a_);
    }

    // Faces: average of the 4 old faces covering each new face, ambient wind outside.
    const FieldView ou = vel_.u_view(), ov = vel_.v_view(), ow = vel_.w_view();
    FieldView nu = vel_a_.u_view(), nv = vel_a_.v_view(), nw = vel_a_.w_view();
    const float* pu = new_wu.data();
    const float* pv = new_wv.data();
    parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) {
        const int a = i - ox, b = j - oy, c = k;
        if (a >= 0 && a <= hx && b >= 0 && b < hy && c < hz) {
            nu(i, j, k) = 0.25f * (ou(2 * a, 2 * b, 2 * c) + ou(2 * a, 2 * b + 1, 2 * c) + ou(2 * a, 2 * b, 2 * c + 1) +
                                   ou(2 * a, 2 * b + 1, 2 * c + 1));
        } else {
            nu(i, j, k) = pu[k];
        }
    });
    parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) {
        const int a = i - ox, b = j - oy, c = k;
        if (a >= 0 && a < hx && b >= 0 && b <= hy && c < hz) {
            nv(i, j, k) = 0.25f * (ov(2 * a, 2 * b, 2 * c) + ov(2 * a + 1, 2 * b, 2 * c) + ov(2 * a, 2 * b, 2 * c + 1) +
                                   ov(2 * a + 1, 2 * b, 2 * c + 1));
        } else {
            nv(i, j, k) = pv[k];
        }
    });
    parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) {
        const int a = i - ox, b = j - oy, c = k;
        if (a >= 0 && a < hx && b >= 0 && b < hy && c <= hz) {
            nw(i, j, k) = 0.25f * (ow(2 * a, 2 * b, 2 * c) + ow(2 * a + 1, 2 * b, 2 * c) + ow(2 * a, 2 * b + 1, 2 * c) +
                                   ow(2 * a + 1, 2 * b + 1, 2 * c));
        } else {
            nw(i, j, k) = 0.f;
        }
    });
    vel_.swap(vel_a_);

    grid_ = ng;
    ambient_ = std::move(new_ambient);
    wind_u_ = std::move(new_wu);
    wind_v_ = std::move(new_wv);
    if (humidity_.enabled) ambient_qv_ = std::move(new_qv);
    pressure_ = PressureSolver(dev_, n, ng.dx);

    // The averaged velocity is not exactly divergence-free on the new grid.
    source_.zero();
    update_density();
    project(vel_);
}

void LowMachSolver::load_cell_state(const std::vector<float>& temperature, const std::vector<float>& smoke_in,
                                    const std::vector<float>* dust_bins, const std::vector<float>& u_in,
                                    const std::vector<float>& v_in, const std::vector<float>& w_in,
                                    const std::vector<float>* vapor_in, const std::vector<float>* cond_in) {
    const Int3 n = grid_.dims;
    const std::size_t cells = std::size_t(grid_.count());
    if (temperature.size() != cells || smoke_in.size() != cells || u_in.size() != cells || v_in.size() != cells ||
        w_in.size() != cells) {
        throw std::invalid_argument("load_cell_state: arrays do not match the grid");
    }
    // Temperature -> potential temperature, on the device.
    tmp_a_.upload(temperature.data(), cells);
    const FieldView tin{tmp_a_.data(), n};
    FieldView th = theta();
    const float* exner = ambient_.exner.data();
    const float* t_amb = ambient_.temperature.data();
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        th(i, j, k) = fmaxf(tin(i, j, k), 0.5f * t_amb[k]) / exner[k];
    });
    scalars_[kSmoke].upload(smoke_in.data(), cells);
    if (dust_.enabled && dust_bins) {
        for (int b = 0; b < kDustBins; ++b) scalars_[std::size_t(kDust0 + b)].upload(dust_bins[b].data(), cells);
    }
    if (humidity_.enabled && vapor_in && cond_in) {
        scalars_[std::size_t(vapor_index_)].upload(vapor_in->data(), cells);
        scalars_[std::size_t(vapor_index_ + 1)].upload(cond_in->data(), cells);
    }

    // Cell velocities -> faces (mean of the two neighbours; outer faces take the boundary cell).
    Buffer<float>& cu = tmp_a_;
    Buffer<float>& cv = tmp_b_;
    Buffer<float>& cw = scratch_.a;
    cu.upload(u_in.data(), cells);
    cv.upload(v_in.data(), cells);
    cw.upload(w_in.data(), cells);
    const FieldView fu{cu.data(), n}, fv{cv.data(), n}, fw{cw.data(), n};
    FieldView u = vel_.u_view(), v = vel_.v_view(), w = vel_.w_view();
    parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) {
        const int a = i > 0 ? i - 1 : 0, b = i < n.x ? i : n.x - 1;
        u(i, j, k) = 0.5f * (fu(a, j, k) + fu(b, j, k));
    });
    parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) {
        const int a = j > 0 ? j - 1 : 0, b = j < n.y ? j : n.y - 1;
        v(i, j, k) = 0.5f * (fv(i, a, k) + fv(i, b, k));
    });
    parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) {
        if (k == 0) {
            w(i, j, k) = 0.f;
            return;
        }
        const int b = k < n.z ? k : n.z - 1;
        w(i, j, k) = 0.5f * (fw(i, j, k - 1) + fw(i, j, b));
    });
    source_.zero();
    update_density();
    project(vel_);
}

CloudExtent LowMachSolver::cloud_extent(float smoke_threshold) {
    const GridInfo g = grid_;
    FieldView sm = smoke();
    const float cx = g.origin.x + 0.5f * g.dims.x * g.dx, cy = g.origin.y + 0.5f * g.dims.y * g.dx;
    CloudExtent e;
    e.top = parallel_max(dev_, g.dims, SIM_LAMBDA(int i, int j, int k) {
        return sm(i, j, k) > smoke_threshold ? (k + 1) * g.dx : 0.f;
    });
    e.radius = parallel_max(dev_, g.dims, SIM_LAMBDA(int i, int j, int k) {
        if (sm(i, j, k) <= smoke_threshold) return 0.f;
        const Float3 p = g.cell_center(i, j, k);
        return sqrtf((p.x - cx) * (p.x - cx) + (p.y - cy) * (p.y - cy)) + 0.5f * g.dx;
    });
    return e;
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
    const MoisturePointers mp = moisture_pointers();
    const float cp = phys::kHeatCapacityAir;

    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const float ta = t_amb[k];
        const float t = th(i, j, k) * exner[k];
        float t_new = t;
        if (t > ta && lambda_p > 0.f) {
            const float x = t / t_ref;
            const float lambda = lambda_p / p_amb[k];
            // Exact solution of the power-law cooling over dt (unconditionally stable).
            t_new = fmaxf(t * powf(1.f + m * lambda * powf(x, m) * dt, -1.f / m), ta);
        }
        if (mp.qv) {
            // Condensation / evaporation at constant pressure, with its latent heat.
            const std::int64_t idx = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
            float qv = mp.qv[idx], qc = mp.qc[idx];
            t_new += saturation_adjust(t_new, p_amb[k], qv, qc, cp);
            mp.qv[idx] = qv;
            mp.qc[idx] = qc;
        }
        float s = 0.f;
        if (t_new != t) {
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
    const DustPointers dp = dust_pointers();
    const detail::DustSum dust{{dp.bin[0], dp.bin[1], dp.bin[2]}, dust_.mass_loading ? dp.count : 0};
    const MoisturePointers mp = moisture_pointers();
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const std::int64_t idx = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
        // Vapour lightens the gas (virtual temperature), cloud water and dust add mass.
        const float qv = mp.qv ? mp.qv[idx] : 0.f, qc = mp.qc ? mp.qc[idx] : 0.f;
        rho(i, j, k) = p_amb[k] / (r_gas * th(i, j, k) * exner[k] * (1.f + 0.608f * qv)) * (1.f + dust.at(idx) + qc);
    });
    pressure_.set_density(rho);
}

void LowMachSolver::add_forces(MacField& field, float dt) {
    const Int3 n = grid_.dims;
    FieldView th = theta();
    FieldView u = field.u_view(), v = field.v_view(), w = field.w_view();
    const float* th_a = ambient_.theta.data();
    const float g = phys::kGravity;
    const DustPointers dp = dust_pointers();
    const detail::DustSum dust{{dp.bin[0], dp.bin[1], dp.bin[2]}, dust_.mass_loading ? dp.count : 0};
    const MoisturePointers mp = moisture_pointers();

    // Buoyancy of the gas-dust-water mixture relative to the ambient column, averaged from the
    // two cells around each z-face: rho_a / rho_mix - 1 = (theta_v / theta_va) / (1 + dust + qc) - 1.
    parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) {
        if (k == 0) return;  // ground
        const int kb = k - 1, kt = k < n.z ? k : n.z - 1;
        const std::int64_t ib = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * kb);
        const std::int64_t it = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * kt);
        float vb = 1.f, vt = 1.f, lb = 1.f + dust.at(ib), lt = 1.f + dust.at(it);
        if (mp.qv) {
            vb = (1.f + 0.608f * mp.qv[ib]) / (1.f + 0.608f * mp.qv_ambient[kb]);
            vt = (1.f + 0.608f * mp.qv[it]) / (1.f + 0.608f * mp.qv_ambient[kt]);
            lb += mp.qc[ib];
            lt += mp.qc[it];
        }
        const float bb = th(i, j, kb) / th_a[kb] * vb / lb - 1.f;
        const float bt = th(i, j, kt) / th_a[kt] * vt / lt - 1.f;
        w(i, j, k) += dt * g * 0.5f * (bb + bt);
    });

    if (params_.vorticity_confinement <= 0.f) return;
    const float inv_2dx = 0.5f / grid_.dx;
    const detail::CellVelocity cv{u, v, w, n};
    // Temporaries are free here: vorticity in tmp_a_, tmp_b_, scratch_.c; force in scratch_.px/py/pz.
    FieldView ox{tmp_a_.data(), n}, oy{tmp_b_.data(), n}, oz{scratch_.c.data(), n};
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 xp = cv.at(i + 1, j, k), xm = cv.at(i - 1, j, k);
        const Float3 yp = cv.at(i, j + 1, k), ym = cv.at(i, j - 1, k);
        const Float3 zp = cv.at(i, j, k + 1), zm = cv.at(i, j, k - 1);
        ox(i, j, k) = ((yp.z - ym.z) - (zp.y - zm.y)) * inv_2dx;
        oy(i, j, k) = ((zp.x - zm.x) - (xp.z - xm.z)) * inv_2dx;
        oz(i, j, k) = ((xp.y - xm.y) - (yp.x - ym.x)) * inv_2dx;
    });
    const detail::Confinement conf{ox, oy, oz, n, params_.vorticity_confinement * grid_.dx, inv_2dx, 1.f / grid_.dx};
    // Force per cell once, then averaged onto the faces.
    FieldView fx{scratch_.px.data(), n}, fy{scratch_.py.data(), n}, fz{scratch_.pz.data(), n};
    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const Float3 f = conf.force(i, j, k);
        fx(i, j, k) = f.x;
        fy(i, j, k) = f.y;
        fz(i, j, k) = f.z;
    });
    parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) {
        const int a = i > 0 ? i - 1 : 0, b = i < n.x ? i : n.x - 1;
        u(i, j, k) += dt * 0.5f * (fx(a, j, k) + fx(b, j, k));
    });
    parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) {
        const int a = j > 0 ? j - 1 : 0, b = j < n.y ? j : n.y - 1;
        v(i, j, k) += dt * 0.5f * (fy(i, a, k) + fy(i, b, k));
    });
    parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) {
        if (k == 0) return;
        const int b = k < n.z ? k : n.z - 1;
        w(i, j, k) += dt * 0.5f * (fz(i, j, k - 1) + fz(i, j, b));
    });
}

void LowMachSolver::advect_scalars(const MacView& vel, float dt) {
    std::vector<Buffer<float>*> fields;
    for (Buffer<float>& s : scalars_) fields.push_back(&s);
    advect_cells_maccormack(dev_, vel, dt, grid_.dx, grid_.dims, fields.data(), int(fields.size()), tmp_a_, tmp_b_,
                            scratch_);
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
    FieldView nu{tmp_b_.data(), n};

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
    for (std::size_t s = 0; s < scalars_.size(); ++s) {
        detail::diffuse_cells(dev_, {scalars_[s].data(), n}, {tmp_a_.data(), n}, nu, c_scalar);
        std::swap(scalars_[s], tmp_a_);
    }

    // Faces through scratch_.b, swapped in component by component.
    const float c_mom = dt / (dx * dx);
    Buffer<float>& tmp = scratch_.b;
    detail::diffuse_faces(dev_, field.u_view(), {tmp.data(), u_dims(n)}, nu, n, 0, c_mom);
    std::swap(field.u, tmp);
    detail::diffuse_faces(dev_, field.v_view(), {tmp.data(), v_dims(n)}, nu, n, 1, c_mom);
    std::swap(field.v, tmp);
    detail::diffuse_faces(dev_, field.w_view(), {tmp.data(), w_dims(n)}, nu, n, 2, c_mom);
    std::swap(field.w, tmp);
}

// Settling (first-order upwind, falling dust leaves through the ground) and lifting of
// loose sand by strong near-surface wind into the lowest cells.
void LowMachSolver::settle_and_lift_dust(const MacField& field, float dt) {
    if (!dust_.enabled) return;
    const Int3 n = grid_.dims;
    const float dx = grid_.dx;
    const float kappa = 0.4f;  // von Karman
    // Friction velocity from the wind in the lowest cell (log law): u* = kappa U / ln(z / z0).
    const float ustar_per_u = kappa / std::log(std::max(0.5f * dx / dust_.roughness_length, 1.01f));
    const float ut = dust_.threshold_friction_velocity;
    const float lift_rate = dust_.erosion * dt / dx;  // dY = C u* (u*^2 - u*t^2) dt / dx
    const FieldView u = const_cast<MacField&>(field).u_view(), v = const_cast<MacField&>(field).v_view();

    for (int b = 0; b < kDustBins; ++b) {
        FieldView src = dust(b), dst{tmp_a_.data(), n};
        const float c = dust_.settling[b] * dt / dx;  // Courant number of the fall
        const float split = dust_.erosion_split[b];
        parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
            const float above = k + 1 < n.z ? src(i, j, k + 1) : 0.f;  // clean air enters from the top
            float y = src(i, j, k) + fminf(c, 1.f) * (above - src(i, j, k));
            if (k == 0 && lift_rate > 0.f) {
                const float uc = 0.5f * (u(i, j, 0) + u(i + 1, j, 0));
                const float vc = 0.5f * (v(i, j, 0) + v(i, j + 1, 0));
                const float us = ustar_per_u * sqrtf(uc * uc + vc * vc);
                if (us > ut) y += split * lift_rate * us * (us * us - ut * ut);
            }
            dst(i, j, k) = fminf(y, 0.5f);
        });
        std::swap(scalars_[std::size_t(kDust0 + b)], tmp_a_);
    }
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

    StageProfiler& prof = profiler_;
    prof.start();
    cool_and_build_source(dt);
    update_density();
    prof.lap("cool");

    if (params_.reflection) {
        const float h = 0.5f * dt;
        // Half step: advect, force, project. vel_ is free once advected and keeps the
        // unprojected copy for the reflection.
        advect_velocity(dev_, vel_.sampler(), h, grid_.dx, vel_, vel_a_, scratch_);
        prof.lap("advect_u");
        add_forces(vel_a_, h);
        prof.lap("forces");
        FieldView ua = vel_a_.u_view(), va = vel_a_.v_view(), wa = vel_a_.w_view();
        FieldView ub = vel_.u_view(), vb = vel_.v_view(), wb = vel_.w_view();
        parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) { ub(i, j, k) = ua(i, j, k); });
        parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) { vb(i, j, k) = va(i, j, k); });
        parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) { wb(i, j, k) = wa(i, j, k); });
        prof.lap("other");
        record(project(vel_a_));
        prof.lap("project");
        // Reflect the pressure correction: u_hat = 2 u_proj - u_unproj.
        parallel_for(dev_, u_dims(n), SIM_LAMBDA(int i, int j, int k) { ub(i, j, k) = 2.f * ua(i, j, k) - ub(i, j, k); });
        parallel_for(dev_, v_dims(n), SIM_LAMBDA(int i, int j, int k) { vb(i, j, k) = 2.f * va(i, j, k) - vb(i, j, k); });
        parallel_for(dev_, w_dims(n), SIM_LAMBDA(int i, int j, int k) { wb(i, j, k) = 2.f * wa(i, j, k) - wb(i, j, k); });
        prof.lap("other");
        // Second half: advect the reflected field with the projected midpoint velocity.
        const MacView mid = vel_a_.sampler();
        advect_velocity_in_place(dev_, mid, h, grid_.dx, vel_, scratch_);
        prof.lap("advect_u");
        advect_scalars(mid, dt);
        prof.lap("advect_scalars");
        apply_turbulence(vel_, dt);
        prof.lap("turbulence");
        settle_and_lift_dust(vel_, dt);
        update_density();
        prof.lap("dust");
        add_forces(vel_, h);
        prof.lap("forces");
        record(project(vel_));
        prof.lap("project");
    } else {
        advect_scalars(vel_.sampler(), dt);
        prof.lap("advect_scalars");
        advect_velocity(dev_, vel_.sampler(), dt, grid_.dx, vel_, vel_a_, scratch_);
        prof.lap("advect_u");
        apply_turbulence(vel_a_, dt);
        prof.lap("turbulence");
        settle_and_lift_dust(vel_a_, dt);
        update_density();
        prof.lap("dust");
        add_forces(vel_a_, dt);
        prof.lap("forces");
        record(project(vel_a_));
        prof.lap("project");
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
    if (dust_.enabled) {
        const DustPointers dp = dust_pointers();
        const detail::DustSum dust{{dp.bin[0], dp.bin[1], dp.bin[2]}, dp.count};
        FieldView th = theta();
        const float* exner = ambient_.exner.data();
        const float* p_amb = ambient_.pressure.data();
        const float r_gas = phys::kGasConstantAir;
        const double cell_volume = double(g.dx) * g.dx * g.dx;
        d.dust_tonnes = 1e-3 * cell_volume * parallel_sum(dev_, n, SIM_LAMBDA(int i, int j, int k) {
            const std::int64_t idx = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
            return double(dust.at(idx)) * p_amb[k] / (r_gas * th(i, j, k) * exner[k]);
        });
    }
    return d;
}

double LowMachSolver::condensate_tonnes() {
    if (!humidity_.enabled) return 0.0;
    const Int3 n = grid_.dims;
    FieldView th = theta(), qc = condensate();
    const float* exner = ambient_.exner.data();
    const float* p_amb = ambient_.pressure.data();
    const float r_gas = phys::kGasConstantAir;
    const double cell_volume = double(grid_.dx) * grid_.dx * grid_.dx;
    return 1e-3 * cell_volume * parallel_sum(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        return double(qc(i, j, k)) * p_amb[k] / (r_gas * th(i, j, k) * exner[k]);
    });
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

void LowMachSolver::fill_export(FieldView density, FieldView dust_out, FieldView cond_out, FieldView temperature,
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

void LowMachSolver::export_field_to_host(ExportField f, float* host, float flame_temperature) {
    export_field(f, FieldView{tmp_a_.data(), grid_.dims}, flame_temperature);
    tmp_a_.download(host, std::size_t(grid_.count()));
}

void LowMachSolver::export_field(ExportField field, FieldView out, float flame_temperature) {
    const Int3 n = grid_.dims;
    FieldView th = theta(), sm = smoke();
    const float* exner = ambient_.exner.data();
    const float* t_amb = ambient_.temperature.data();
    const float* p_amb = ambient_.pressure.data();
    const float rho_ground = ambient_.ground_density;
    const float r_gas = phys::kGasConstantAir;
    const float inv_tf = 1.f / flame_temperature;
    const float inv_2dx = 0.5f / grid_.dx;
    const detail::CellVelocity cv{vel_.u_view(), vel_.v_view(), vel_.w_view(), n};
    const DustPointers dp = dust_pointers();
    float wgt[kDustBins];
    for (int b = 0; b < kDustBins; ++b) wgt[b] = dust_.optical_weight[b] * dust_.render_scale;
    const float w0 = wgt[0], w1 = wgt[1], w2 = wgt[2];
    const float* d0 = dp.bin[0];
    const float* d1 = dp.bin[1];
    const float* d2 = dp.bin[2];
    const bool has_dust = dp.count > 0;
    const MoisturePointers mp = moisture_pointers();
    const float cond_scale = humidity_.render_scale;
    const int which = int(field);

    parallel_for(dev_, n, SIM_LAMBDA(int i, int j, int k) {
        const std::int64_t idx = i + std::int64_t(n.x) * (j + std::int64_t(n.y) * k);
        const float t = th(i, j, k) * exner[k];
        const float ta = t_amb[k];
        const float s = sm(i, j, k);
        const float rho = p_amb[k] / (r_gas * t);
        const float optical_dust = has_dust ? (w0 * d0[idx] + w1 * d1[idx] + w2 * d2[idx]) * rho / rho_ground : 0.f;
        const float cloud = mp.qc ? cond_scale * mp.qc[idx] * rho / rho_ground : 0.f;
        const bool hot = t > ta + 1.f;
        const bool active = hot || s > 1e-4f || optical_dust > 1e-4f || cloud > 1e-4f;
        float v = 0.f;
        switch (ExportField(which)) {
            case ExportField::Density: v = s * rho / rho_ground; break;
            case ExportField::Dust: v = optical_dust; break;
            case ExportField::Condensate: v = cloud; break;
            case ExportField::Temperature: v = active ? t : 0.f; break;
            case ExportField::Flame: {
                const float x = t * inv_tf, xa = ta * inv_tf;
                v = hot ? fmaxf(x * x * x * x - xa * xa * xa * xa, 0.f) : 0.f;
                break;
            }
            case ExportField::VelX: v = active ? cv.at(i, j, k).x : 0.f; break;
            case ExportField::VelY: v = active ? cv.at(i, j, k).y : 0.f; break;
            case ExportField::VelZ: v = active ? cv.at(i, j, k).z : 0.f; break;
            case ExportField::Vorticity:
                if (active) {
                    const Float3 xp = cv.at(i + 1, j, k), xm = cv.at(i - 1, j, k);
                    const Float3 yp = cv.at(i, j + 1, k), ym = cv.at(i, j - 1, k);
                    const Float3 zp = cv.at(i, j, k + 1), zm = cv.at(i, j, k - 1);
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

}  // namespace sim
