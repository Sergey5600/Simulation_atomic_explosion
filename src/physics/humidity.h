#pragma once
// Water vapour and its condensation (saturation adjustment), shared by both solvers.

#include <utility>
#include <vector>

#include "core/math.h"

namespace sim {

namespace phys {
constexpr float kLatentHeatVaporization = 2.5e6f;  // J/kg
constexpr float kGasConstantVapor = 461.5f;        // J/(kg K)
}  // namespace phys

// Relative humidity of the ambient air as a piecewise-linear function of height.
struct HumidityProfile {
    bool enabled = false;
    std::vector<std::pair<float, float>> points;  // (height above ground m, relative humidity 0..1)
    float render_scale = 1000.f;                   // exported condensate = scale * mass concentration / ground air density

    float relative_humidity_at(float z) const {
        if (points.empty()) return 0.f;
        if (z <= points.front().first) return points.front().second;
        for (std::size_t i = 1; i < points.size(); ++i) {
            if (z <= points[i].first) {
                const auto [z0, h0] = points[i - 1];
                const auto [z1, h1] = points[i];
                const float t = z1 > z0 ? (z - z0) / (z1 - z0) : 1.f;
                return h0 + (h1 - h0) * t;
            }
        }
        return points.back().second;
    }
};

// Saturation vapour pressure over water (Bolton 1980), Pa. Capped so it stays finite in the fireball.
SIM_INLINE float saturation_vapor_pressure(float t) {
    const float tc = fminf(t - 273.15f, 150.f);
    return 611.2f * expf(17.67f * tc / (tc + 243.5f));
}

// Saturation specific humidity (kg vapour / kg moist air) at temperature t and pressure p.
SIM_INLINE float saturation_humidity(float t, float p) {
    const float es = fminf(saturation_vapor_pressure(t), 0.5f * p);
    return 0.622f * es / (p - 0.378f * es);
}

// One step of saturation adjustment: moves water between vapour and condensate so the
// air ends at (or below) saturation, and returns the temperature change from latent
// heat. cp is the heat capacity at the process (c_p at constant pressure, c_v in a closed cell).
SIM_INLINE float saturation_adjust(float t, float p, float& qv, float& qc, float cp) {
    const float lv = phys::kLatentHeatVaporization;
    const float qs = saturation_humidity(t, p);
    // Linearised so the latent heating does not overshoot: dq = (qv - qs) / (1 + L^2 qs / (cp Rv T^2)).
    const float gamma = 1.f + lv * lv * qs / (cp * phys::kGasConstantVapor * t * t);
    float dq = (qv - qs) / gamma;  // > 0 condenses, < 0 evaporates
    if (dq < 0.f) dq = fmaxf(dq, -qc);
    qv -= dq;
    qc += dq;
    return lv * dq / cp;
}

}  // namespace sim
