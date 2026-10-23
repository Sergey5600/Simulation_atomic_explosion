#include <doctest/doctest.h>

#include <vector>

#include "core/backend.h"
#include "core/field_ops.h"
#include "core/grid.h"

using namespace sim;

namespace {

GridInfo small_grid() {
    GridInfo g;
    g.dims = Int3{13, 7, 5};  // odd sizes catch launch-bounds mistakes
    g.dx = 2.f;
    return g;
}

void check_backend(Device dev) {
    const GridInfo g = small_grid();
    ScalarField f("f", g, dev);
    fill_linear_index(f);
    const std::vector<float> host = f.buffer().to_host();
    REQUIRE(host.size() == std::size_t(g.count()));
    for (std::size_t i = 0; i < host.size(); ++i) REQUIRE(host[i] == float(i));

    ScalarField a("a", g, dev);
    fill(a, 2.f);
    axpy(a, 0.5f, f);  // a = 2 + 0.5 * index
    const std::vector<float> ha = a.buffer().to_host();
    for (std::size_t i = 0; i < ha.size(); ++i) REQUIRE(ha[i] == doctest::Approx(2.f + 0.5f * float(i)));
}

}  // namespace

TEST_CASE("CPU backend: parallel_for covers every cell in layout order") { check_backend(Device::CPU); }

TEST_CASE("CUDA backend matches CPU") {
    if (!cuda_available()) {
        MESSAGE("no CUDA device: skipped");
        return;
    }
    check_backend(Device::CUDA);
}

TEST_CASE("Buffer upload/download round trip and move") {
    const Device dev = cuda_available() ? Device::CUDA : Device::CPU;
    std::vector<float> src(1000);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = float(i) * 0.25f;
    Buffer<float> b(dev, src.size());
    b.upload(src.data(), src.size());
    Buffer<float> moved = std::move(b);
    CHECK(b.data() == nullptr);
    CHECK(moved.to_host() == src);
}

TEST_CASE("GridInfo geometry") {
    GridInfo g;
    g.dims = Int3{4, 4, 4};
    g.dx = 10.f;
    g.origin = Float3{-20.f, -20.f, 0.f};
    const Float3 c = g.cell_center(0, 3, 1);
    CHECK(c.x == doctest::Approx(-15.f));
    CHECK(c.y == doctest::Approx(15.f));
    CHECK(c.z == doctest::Approx(15.f));
    CHECK(g.index(1, 2, 3) == 1 + 4 * (2 + 4 * 3));
}
