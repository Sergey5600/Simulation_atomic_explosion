#include "preview/preview.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace sim {

namespace {

struct Rgb {
    float r = 0.f, g = 0.f, b = 0.f;
};

// Approximate normalised colour of a black body (Tanner Helland's fit), 1000 K .. 40000 K.
Rgb blackbody_rgb(float kelvin) {
    const float t = std::clamp(kelvin, 1000.f, 40000.f) / 100.f;
    Rgb c;
    c.r = t <= 66.f ? 1.f : std::clamp(1.2929362f * std::pow(t - 60.f, -0.1332047592f), 0.f, 1.f);
    c.g = t <= 66.f ? std::clamp(0.3900816f * std::log(t) - 0.6318414f, 0.f, 1.f)
                    : std::clamp(1.1298909f * std::pow(t - 60.f, -0.0755148492f), 0.f, 1.f);
    c.b = t >= 66.f ? 1.f : (t <= 19.f ? 0.f : std::clamp(0.5432068f * std::log(t - 10.f) - 1.1962541f, 0.f, 1.f));
    return c;
}

std::uint8_t to_srgb8(float linear) {
    const float v = std::clamp(linear, 0.f, 1.f);
    const float s = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
    return std::uint8_t(std::lround(s * 255.f));
}

}  // namespace

void render_preview_png(const std::string& path, const PreviewInput& in, const PreviewOptions& opt) {
    const GridInfo& g = in.grid;

    // Frame the visible part of the volume (columns i0..i1, rows 0..k1), always including the ground.
    int i0 = 0, i1 = g.dims.x - 1, k1 = g.dims.z - 1;
    if (opt.auto_frame) {
        int lo = g.dims.x, hi = -1, top = -1;
        for (int k = 0; k < g.dims.z; ++k)
            for (int j = 0; j < g.dims.y; ++j)
                for (int i = 0; i < g.dims.x; ++i) {
                    const std::int64_t idx = g.index(i, j, k);
                    const bool visible = (in.density && in.density[idx] > 1e-3f) || (in.flame && in.flame[idx] > 1e-4f) ||
                                         (in.dust && in.dust[idx] > 1e-3f) || (in.condensate && in.condensate[idx] > 1e-3f);
                    if (!visible) continue;
                    lo = std::min(lo, i);
                    hi = std::max(hi, i);
                    top = std::max(top, k);
                }
        if (hi >= 0) {
            const int size = std::max({hi - lo + 1, top + 1, 24});
            const int margin = size / 4;
            const int half = (size + 2 * margin) / 2;
            const int cx = (lo + hi) / 2;
            i0 = std::max(0, cx - half);
            i1 = std::min(g.dims.x - 1, cx + half);
            k1 = std::min(g.dims.z - 1, top + margin + std::max(0, (i1 - i0) - (top + margin)) / 2);
        }
    }
    const int w = i1 - i0 + 1, h = k1 + 1;
    std::vector<float> img(std::size_t(w) * h * 3, 0.f);

    const Rgb smoke{0.55f, 0.50f, 0.46f};
    const Rgb sand{0.80f, 0.68f, 0.52f};
    const Rgb water{0.95f, 0.96f, 0.98f};
    const Rgb ambient{0.75f, 0.85f, 1.0f};

#pragma omp parallel for schedule(dynamic, 4)
    for (int k = 0; k < h; ++k) {
        for (int x = 0; x < w; ++x) {
            const int i = i0 + x;
            float tr = 1.f;
            Rgb L;
            for (int j = 0; j < g.dims.y && tr > 1e-3f; ++j) {
                const std::int64_t idx = g.index(i, j, k);
                const float ds = in.density ? std::max(in.density[idx], 0.f) : 0.f;
                const float dd = in.dust ? std::max(in.dust[idx], 0.f) : 0.f;
                const float dc = in.condensate ? std::max(in.condensate[idx], 0.f) : 0.f;
                const float f = in.flame ? std::max(in.flame[idx], 0.f) : 0.f;
                const float d = ds + dd + dc;
                if (d <= 0.f && f <= 0.f) continue;
                const float alpha = 1.f - std::exp(-d * opt.density_scale * g.dx);
                const float inv = d > 0.f ? 1.f / d : 0.f;
                const Rgb albedo{(ds * smoke.r + dd * sand.r + dc * water.r) * inv,
                                 (ds * smoke.g + dd * sand.g + dc * water.g) * inv,
                                 (ds * smoke.b + dd * sand.b + dc * water.b) * inv};
                Rgb e;
                if (f > 0.f) {
                    const Rgb bb = blackbody_rgb(in.temperature ? in.temperature[idx] : 3000.f);
                    const float s = f * opt.emission_scale * g.dx;
                    e = Rgb{bb.r * s, bb.g * s, bb.b * s};
                }
                // Scattered sky light (crudely constant) plus emission.
                L.r += tr * (alpha * albedo.r * ambient.r * 0.6f + e.r);
                L.g += tr * (alpha * albedo.g * ambient.g * 0.6f + e.g);
                L.b += tr * (alpha * albedo.b * ambient.b * 0.6f + e.b);
                tr *= 1.f - alpha;
            }
            // Sky gradient behind the volume.
            const float sky_t = float(k) / float(std::max(h - 1, 1));
            const Rgb sky{0.55f + 0.1f * sky_t, 0.70f + 0.08f * sky_t, 0.85f + 0.05f * sky_t};
            const std::size_t o = (std::size_t(h - 1 - k) * w + x) * 3;  // image rows go top-down
            img[o + 0] = L.r + tr * sky.r;
            img[o + 1] = L.g + tr * sky.g;
            img[o + 2] = L.b + tr * sky.b;
        }
    }

    // Integer upscale so small debug grids are still readable.
    const int scale = std::max(1, (opt.min_size + std::max(w, h) - 1) / std::max(w, h));
    const int W = w * scale, H = h * scale;
    std::vector<std::uint8_t> out(std::size_t(W) * H * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::size_t src = (std::size_t(y / scale) * w + x / scale) * 3;
            const std::size_t dst = (std::size_t(y) * W + x) * 3;
            for (int c = 0; c < 3; ++c) {
                // Simple exponential tone map.
                out[dst + c] = to_srgb8(1.f - std::exp(-img[src + c] * opt.exposure));
            }
        }
    }
    if (!stbi_write_png(path.c_str(), W, H, 3, out.data(), W * 3)) {
        throw std::runtime_error("failed to write " + path);
    }
}

}  // namespace sim
