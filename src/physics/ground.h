#pragma once
// Desert ground for dust lifting, shared by both solvers.
//
// Real ground is patchy: loose sand next to crusted playa, gravel and brush. A
// uniform ground makes the dust skirt a perfect ring and the stem a smooth
// column fed by regular streamers, so the lifting rate is scaled by an
// "erodibility" map (fractal noise over a few hundred metres down to tens of
// metres, zero on hard patches) and driven by the friction velocity averaged
// over neighbouring cells instead of a single cell.

#include <cstdint>

#include "core/math.h"
#include "core/noise.h"

namespace sim {

struct GroundParams {
    float variation = 0.8f;     // 0 = uniform ground; ~1 leaves bare patches between loose sand
    float scale = 250.f;        // m, size of the largest patches
    int smoothing = 1;          // friction velocity averaged over (2 n + 1)^2 ground cells
    std::uint32_t seed = 7;
};

// Relative erodibility at ground position (x, y): mean about 1, never negative.
SIM_INLINE float ground_erodibility(float x, float y, float variation, float inv_scale, std::uint32_t seed) {
    if (variation <= 0.f) return 1.f;
    const float n = noise::fbm(make_f3(x * inv_scale, y * inv_scale, 0.37f), 4, 2.2f, 0.55f, seed);
    return fmaxf(0.f, 1.f + 3.f * variation * n);  // fbm has a spread of about +-0.25
}

}  // namespace sim
