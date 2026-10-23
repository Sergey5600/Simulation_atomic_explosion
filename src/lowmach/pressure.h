#pragma once
// Variable-coefficient pressure projection with a multigrid-preconditioned
// conjugate gradient solver (MGPCG, McAdams et al. 2010).
//
// Solves  div( (1/rho) grad(phi) ) = div(u*) - S  and sets  u = u* - (1/rho) grad(phi),
// so that div(u) = S, the low-Mach volume source. Boundaries: the ground (z = 0)
// is a solid wall, the sides and the top are open (phi = 0).

#include <vector>

#include "core/grid.h"
#include "lowmach/mac.h"

namespace sim {

struct ProjectionStats {
    int iterations = 0;
    double initial_residual = 0.0;  // ||div(u*) - S||_2 / sqrt(cells), 1/s
    double final_residual = 0.0;
};

class PressureSolver {
public:
    PressureSolver(Device dev, Int3 dims, float dx);

    // Face coefficients 1/rho from the cell density (kg/m^3).
    void set_density(FieldView rho);

    ProjectionStats project(MacField& vel, FieldView source, float tolerance, int max_iterations);

    int levels() const { return int(levels_.size()); }
    std::size_t memory_bytes() const;

    // Multigrid building blocks. Public only because nvcc does not allow device
    // lambdas inside private member functions.
    void vcycle(std::size_t l, FieldView x, FieldView b);
    void apply(std::size_t l, FieldView x, FieldView out);
    void residual(std::size_t l, FieldView x, FieldView b, FieldView r);
    void smooth(std::size_t l, FieldView x, FieldView b, int color);
    double dot(FieldView a, FieldView b);

private:
    struct Level {
        Int3 dims;
        float dx = 1.f;
        Buffer<float> bx, by, bz;  // face coefficients (Dirichlet faces doubled, wall faces zero)
        Buffer<float> x, b, r;     // unallocated on level 0: the CG vectors are used instead
    };
    // Residual buffer of a level; on level 0 it borrows ap_, which is idle during a V-cycle.
    Buffer<float>& level_residual(std::size_t l) { return l == 0 ? ap_ : levels_[l].r; }

    Device dev_;
    Int3 dims_;
    float dx_;
    std::vector<Level> levels_;
    Buffer<float> phi_, r_, z_, p_, ap_;  // the right-hand side is built directly in r_
};

}  // namespace sim
