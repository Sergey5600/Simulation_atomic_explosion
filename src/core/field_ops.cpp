#include "core/field_ops.h"

#include <stdexcept>

#include "core/parallel.h"

namespace sim {

void fill(ScalarField& f, float value) {
    FieldView v = f.view();
    parallel_for(f.device(), f.grid().dims, SIM_LAMBDA(int i, int j, int k) { v(i, j, k) = value; });
}

void fill_linear_index(ScalarField& f) {
    FieldView v = f.view();
    const GridInfo g = f.grid();
    parallel_for(f.device(), g.dims, SIM_LAMBDA(int i, int j, int k) { v(i, j, k) = float(g.index(i, j, k)); });
}

void axpy(ScalarField& dst, float a, ScalarField& x) {
    if (!(dst.grid().dims == x.grid().dims) || dst.device() != x.device()) {
        throw std::invalid_argument("axpy: fields must share dims and device");
    }
    FieldView d = dst.view();
    FieldView s = x.view();
    parallel_for(dst.device(), dst.grid().dims, SIM_LAMBDA(int i, int j, int k) { d(i, j, k) += a * s(i, j, k); });
}

}  // namespace sim
