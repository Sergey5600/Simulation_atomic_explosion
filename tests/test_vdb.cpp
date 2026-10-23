#include <doctest/doctest.h>

#include <openvdb/openvdb.h>

#include <filesystem>
#include <vector>

#include "io/vdb_writer.h"

using namespace sim;

TEST_CASE("VDB export preserves values, layout and world transform") {
    GridInfo g;
    g.dims = Int3{8, 6, 4};
    g.dx = 2.f;
    g.origin = Float3{-8.f, -6.f, 0.f};
    const std::size_t n = std::size_t(g.count());

    std::vector<float> dens(n, 0.f), vx(n, 0.f), vy(n, 0.f), vz(n, 0.f);
    for (int k = 0; k < g.dims.z; ++k)
        for (int j = 0; j < g.dims.y; ++j)
            for (int i = 0; i < g.dims.x; ++i) {
                const auto idx = std::size_t(g.index(i, j, k));
                if ((i + j + k) % 3 == 0) dens[idx] = 1.f + float(i) + 10.f * float(j) + 100.f * float(k);
                vx[idx] = float(i);
                vy[idx] = float(j);
                vz[idx] = float(k);
            }

    VdbFrame frame;
    frame.grid = g;
    frame.scalars.push_back({"density", dens.data()});
    frame.vectors.push_back({"velocity", vx.data(), vy.data(), vz.data(), 2.f});
    frame.metadata["sim_time"] = 0.5f;

    const auto path = (std::filesystem::temp_directory_path() / "nukesim_test_roundtrip.vdb").string();
    VdbWriteOptions opt;
    opt.half_float = false;  // exact comparison
    opt.world_scale = 0.5f;
    const VdbWriteStats stats = write_vdb(path, frame, opt);
    CHECK(stats.file_bytes > 0);

    openvdb::io::File file(path);
    file.open();
    auto dgrid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid("density"));
    auto vgrid = openvdb::gridPtrCast<openvdb::Vec3SGrid>(file.readGrid("velocity"));
    file.close();
    REQUIRE(dgrid);
    REQUIRE(vgrid);

    auto acc = dgrid->getConstAccessor();
    auto vacc = vgrid->getConstAccessor();
    std::size_t nonzero = 0;
    for (int k = 0; k < g.dims.z; ++k)
        for (int j = 0; j < g.dims.y; ++j)
            for (int i = 0; i < g.dims.x; ++i) {
                const float expected = dens[std::size_t(g.index(i, j, k))];
                REQUIRE(acc.getValue(openvdb::Coord(i, j, k)) == expected);
                if (expected != 0.f) {
                    ++nonzero;
                    // Velocity is stored only where density is active, scaled on export.
                    const openvdb::Vec3s v = vacc.getValue(openvdb::Coord(i, j, k));
                    REQUIRE(v.x() == doctest::Approx(2.f * float(i)));
                    REQUIRE(v.y() == doctest::Approx(2.f * float(j)));
                    REQUIRE(v.z() == doctest::Approx(2.f * float(k)));
                }
            }
    CHECK(stats.active_voxels == nonzero);
    CHECK(std::size_t(vgrid->activeVoxelCount()) == nonzero);

    // Voxel (i, j, k) centre maps to (origin + (ijk + 0.5) * dx) * world_scale.
    const openvdb::Vec3d w = dgrid->indexToWorld(openvdb::Coord(1, 2, 3));
    CHECK(w.x() == doctest::Approx((-8.0 + 1.5 * 2.0) * 0.5));
    CHECK(w.y() == doctest::Approx((-6.0 + 2.5 * 2.0) * 0.5));
    CHECK(w.z() == doctest::Approx((0.0 + 3.5 * 2.0) * 0.5));

    std::filesystem::remove(path);
}
