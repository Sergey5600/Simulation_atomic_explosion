#pragma once
// Quick-look PNG renders of the volume, so iterations do not need Blender.

#include <string>

#include "core/grid.h"

namespace sim {

struct PreviewInput {
    GridInfo grid;
    const float* density = nullptr;      // optional
    const float* temperature = nullptr;  // optional, Kelvin
    const float* flame = nullptr;        // optional, emission strength
    const float* dust = nullptr;         // optional, optical dust concentration
    const float* condensate = nullptr;   // optional, cloud water (white)
};

struct PreviewOptions {
    float density_scale = 0.03f;   // extinction per metre per unit density
    float emission_scale = 0.01f;  // radiance per metre per unit flame
    float exposure = 1.f;
    int min_size = 512;            // the image is upscaled to at least this many pixels
    bool auto_frame = true;        // crop to the visible volume instead of the whole domain
};

// Orthographic side view looking along +Y (image x = world X, image up = world Z),
// cropped to the visible volume and the ground below it.
void render_preview_png(const std::string& path, const PreviewInput& in, const PreviewOptions& opt);

struct SliceInput {
    GridInfo grid;
    const float* density = nullptr;     // optional
    const float* dust = nullptr;        // optional
    const float* condensate = nullptr;  // optional
    const float* vorticity = nullptr;   // optional, 1/s
};

// Vertical slice through the middle of the domain (plane y = centre, world X right, Z up),
// two panels side by side for before/after comparisons:
//   left:  smoke, sand and cloud water on black, fixed exposure;
//   right: |vorticity| on a fixed log scale from 0.01 to 100 1/s (black, purple, orange, white).
// The scales never adapt to the data, so two runs can be compared image to image.
void render_slice_png(const std::string& path, const SliceInput& in, int min_size = 512);

}  // namespace sim
