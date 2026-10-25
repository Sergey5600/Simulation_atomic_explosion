#pragma once
// Ambient wind: speed as a piecewise-linear function of height, constant heading.

#include <utility>
#include <vector>

namespace sim {

struct WindProfile {
    float heading_deg = 0.f;                       // direction the wind blows towards, degrees from +X towards +Y
    // Gusts of the desert boundary layer, added once as a random velocity field when the
    // buoyant phase starts. Besides realism they break the four-fold symmetry of a burst
    // centred on a Cartesian grid, which otherwise survives into the cloud's shape.
    float turbulence = 1.5f;          // m/s, typical gust speed (0 = calm)
    float turbulence_scale = 150.f;   // m, size of the largest gusts
    std::vector<std::pair<float, float>> points;   // (height above ground m, speed m/s), ascending height

    float speed_at(float z) const {
        if (points.empty()) return 0.f;
        if (z <= points.front().first) return points.front().second;
        for (std::size_t i = 1; i < points.size(); ++i) {
            if (z <= points[i].first) {
                const auto [z0, s0] = points[i - 1];
                const auto [z1, s1] = points[i];
                const float t = z1 > z0 ? (z - z0) / (z1 - z0) : 1.f;
                return s0 + (s1 - s0) * t;
            }
        }
        return points.back().second;
    }
};

}  // namespace sim
