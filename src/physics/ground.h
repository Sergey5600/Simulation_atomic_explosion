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
    // No lifting near the sides of the domain: the open boundaries let air in too easily, which
    // would raise walls of dust along the edges. Fraction of the half-width over which it fades.
    float edge_fade = 0.15f;
    std::uint32_t seed = 7;
};

// 0 at the sides of the domain [x0, x1] x [y0, y1], rising smoothly to 1 over `width` metres.
SIM_INLINE float ground_edge_factor(float x, float y, float x0, float x1, float y0, float y1, float width) {
    if (width <= 0.f) return 1.f;
    const float d = fminf(fminf(x - x0, x1 - x), fminf(y - y0, y1 - y));
    return smoothstepf(0.f, width, d);
}

// Relative erodibility at ground position (x, y): mean about 1, never negative.
SIM_INLINE float ground_erodibility(float x, float y, float variation, float inv_scale, std::uint32_t seed) {
    if (variation <= 0.f) return 1.f;
    const float n = noise::fbm(make_f3(x * inv_scale, y * inv_scale, 0.37f), 4, 2.2f, 0.55f, seed);
    return fmaxf(0.f, 1.f + 3.f * variation * n);  // fbm has a spread of about +-0.25
}

}  // namespace sim
