#include "scenes/test_fireball.h"

#include <cmath>
#include <stdexcept>

#include "core/noise.h"
#include "core/parallel.h"

namespace sim {

void generate_test_fireball(const TestFireballConfig& cfg, std::uint32_t seed, double time, TestFireballFields& out) {
    if (!out.density || !out.temperature || !out.flame || !out.vel_x || !out.vel_y || !out.vel_z) {
        throw std::invalid_argument("generate_test_fireball: all fields are required");
    }
    const GridInfo g = out.density->grid();
    const Device dev = out.density->device();

    // Radius grows like a blast wave (t^0.4) until formation, then slowly; the
    // ball starts as a ground-hugging hemisphere and lifts off as it forms.
    const float t = float(time);
    const float tf = cfg.formation_time;
    const float growth = t < tf ? std::pow(t / tf, 0.4f) : 1.f + 0.08f * (t - tf);
    const float radius = cfg.max_radius * growth;
    const float dradius = t < tf ? cfg.max_radius * 0.4f * std::pow(std::fmax(t, 1e-4f) / tf, -0.6f) / tf
                                 : cfg.max_radius * 0.08f;
    const float lift = std::fmax(0.f, t - 0.5f * tf);
    const float center_z = 0.6f * radius * std::fmin(1.f, t / tf) + cfg.rise_speed * lift * lift / (lift + tf);
    const float rise = cfg.rise_speed * lift * (lift + 2.f * tf) / ((lift + tf) * (lift + tf));
    const float core_temp = cfg.peak_temperature * std::exp(-t / cfg.cooling_time);
    const float soot = 1.f - std::exp(-t / cfg.cooling_time);
    const float freq = cfg.noise_frequency;
    const float amp = cfg.noise_amplitude;
    const float drift = 0.15f * t;
    const float peak = cfg.peak_temperature;

    FieldView dens = out.density->view();
    FieldView temp = out.temperature->view();
    FieldView flame = out.flame->view();
    FieldView vx = out.vel_x->view(), vy = out.vel_y->view(), vz = out.vel_z->view();

    parallel_for(dev, g.dims, SIM_LAMBDA(int i, int j, int k) {
        const Float3 p = g.cell_center(i, j, k);
        const Float3 rel = make_f3(p.x, p.y, p.z - center_z);
        const float r = length(rel);
        // |noise| <= 1, so every output is zero beyond this radius: skip the noise.
        if (r > radius * (1.3f + amp)) {
            dens(i, j, k) = temp(i, j, k) = flame(i, j, k) = 0.f;
            vx(i, j, k) = vy(i, j, k) = vz(i, j, k) = 0.f;
            return;
        }
        // Boundary displaced by fractal noise in the ball's frame (it rises with the flow).
        const Float3 q = make_f3(p.x * freq, p.y * freq, (p.z - center_z) * freq - drift);
        const float n = noise::fbm(q, 5, 2.03f, 0.5f, seed);
        const float d = r / fmaxf(radius, 1e-3f) + amp * n;

        const float inside = smoothstepf(1.0f, 0.75f, d);
        const float shell = smoothstepf(1.05f, 0.85f, d) * (0.4f + 0.6f * smoothstepf(0.2f, 0.9f, d));
        const float detail = 0.5f + 0.5f * noise::fbm(q * 3.1f, 3, 2.1f, 0.55f, seed + 77u);

        const float T = 300.f + core_temp * inside * (0.7f + 0.3f * detail);
        temp(i, j, k) = inside > 0.f ? T : 0.f;
        flame(i, j, k) = inside * powf(fmaxf(T - 300.f, 0.f) / peak, 2.f);
        dens(i, j, k) = shell * (0.15f + 0.85f * soot) * detail;

        // Radial expansion plus rise inside the ball, zero outside.
        const float w = smoothstepf(1.3f, 0.9f, d);
        const float vr = r > 1e-3f ? dradius * fminf(r / fmaxf(radius, 1e-3f), 1.f) / r : 0.f;
        vx(i, j, k) = w * rel.x * vr;
        vy(i, j, k) = w * rel.y * vr;
        vz(i, j, k) = w * (rel.z * vr + rise);
    });
}

}  // namespace sim
