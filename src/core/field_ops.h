#pragma once
// Basic field kernels, callable from host-only code.

#include "core/grid.h"

namespace sim {

void fill(ScalarField& f, float value);

// f(i, j, k) = linear index; used to verify backends and memory layout.
void fill_linear_index(ScalarField& f);

// dst = a * x + dst
void axpy(ScalarField& dst, float a, ScalarField& x);

double sum(ScalarField& f);
float max_abs(ScalarField& f);

}  // namespace sim
