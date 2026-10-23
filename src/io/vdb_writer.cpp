#include "io/vdb_writer.h"

#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tree/LeafManager.h>

#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>

namespace sim {

namespace {

std::once_flag g_vdb_init;

openvdb::math::Transform::Ptr make_transform(const GridInfo& g, float world_scale) {
    // VDB index coordinates sit at voxel centres; ours are at origin + (i + 0.5) * dx.
    auto xform = openvdb::math::Transform::createLinearTransform(double(g.dx) * world_scale);
    xform->postTranslate(openvdb::Vec3d(g.origin.x + 0.5 * g.dx, g.origin.y + 0.5 * g.dx, g.origin.z + 0.5 * g.dx) *
                         double(world_scale));
    return xform;
}

openvdb::CoordBBox dense_bbox(const GridInfo& g) {
    return openvdb::CoordBBox(openvdb::Coord(0, 0, 0), openvdb::Coord(g.dims.x - 1, g.dims.y - 1, g.dims.z - 1));
}

}  // namespace

VdbWriteStats write_vdb(const std::string& path, const VdbFrame& frame, const VdbWriteOptions& opt) {
    std::call_once(g_vdb_init, [] { openvdb::initialize(); });

    const GridInfo& g = frame.grid;
    const auto xform = make_transform(g, opt.world_scale);
    const openvdb::CoordBBox bbox = dense_bbox(g);

    openvdb::GridPtrVec grids;
    openvdb::MaskGrid::Ptr active = openvdb::MaskGrid::create(false);

    for (const VdbScalarInput& in : frame.scalars) {
        if (!in.data) throw std::invalid_argument("write_vdb: null data for grid " + in.name);
        // Dense wraps the caller's memory without copying (it is only read);
        // LayoutXYZ = x fastest, like GridInfo.
        openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense(bbox, const_cast<float*>(in.data));
        openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.f);
        openvdb::tools::copyFromDense(dense, *grid, opt.tolerance);
        grid->setName(in.name);
        grid->setTransform(xform->copy());
        grid->setGridClass(openvdb::GRID_FOG_VOLUME);
        grid->setSaveFloatAsHalf(opt.half_float);
        active->tree().topologyUnion(grid->tree());
        grids.push_back(grid);
    }

    VdbWriteStats stats;
    active->tree().voxelizeActiveTiles();
    stats.active_voxels = std::size_t(active->tree().activeVoxelCount());

    for (const VdbVectorInput& in : frame.vectors) {
        if (!in.x || !in.y || !in.z) throw std::invalid_argument("write_vdb: null data for grid " + in.name);
        const GridInfo& vg = in.grid ? *in.grid : g;
        openvdb::Vec3SGrid::Ptr grid = openvdb::Vec3SGrid::create(openvdb::Vec3s(0.f));
        if (in.grid) {
            // Voxels of the own grid that cover active voxels of the frame grid, plus a margin so
            // interpolation near the edge of the volume does not fall back to the zero background.
            auto acc = grid->getAccessor();
            const float inv = 1.f / vg.dx;
            for (auto it = active->cbeginValueOn(); it; ++it) {
                const openvdb::Coord c = it.getCoord();
                const Float3 p = g.cell_center(c.x(), c.y(), c.z());
                const int i = int(std::floor((p.x - vg.origin.x) * inv));
                const int j = int(std::floor((p.y - vg.origin.y) * inv));
                const int k = int(std::floor((p.z - vg.origin.z) * inv));
                if (i >= 0 && j >= 0 && k >= 0 && i < vg.dims.x && j < vg.dims.y && k < vg.dims.z) {
                    acc.setValueOn(openvdb::Coord(i, j, k));
                }
            }
            openvdb::tools::dilateActiveValues(grid->tree(), 1, openvdb::tools::NN_FACE_EDGE_VERTEX);
            grid->tree().clip(dense_bbox(vg));  // dilation may step outside the grid
        } else {
            grid->tree().topologyUnion(active->tree());
        }
        grid->tree().voxelizeActiveTiles();
        openvdb::tree::LeafManager<openvdb::Vec3STree> leaves(grid->tree());
        const float s = in.scale;
        leaves.foreach([&](openvdb::Vec3STree::LeafNodeType& leaf, std::size_t) {
            for (auto it = leaf.beginValueOn(); it; ++it) {
                const openvdb::Coord c = it.getCoord();
                const std::int64_t idx = vg.index(c.x(), c.y(), c.z());
                it.setValue(openvdb::Vec3s(in.x[idx] * s, in.y[idx] * s, in.z[idx] * s));
            }
        });
        grid->setName(in.name);
        grid->setTransform(in.grid ? make_transform(vg, opt.world_scale) : xform->copy());
        grid->setVectorType(in.grid ? openvdb::VEC_INVARIANT : openvdb::VEC_CONTRAVARIANT_RELATIVE);
        grid->setSaveFloatAsHalf(opt.half_float && !in.full_precision);
        grids.push_back(grid);
    }

    openvdb::MetaMap file_meta;
    for (const auto& [key, value] : frame.metadata) file_meta.insertMeta(key, openvdb::FloatMetadata(value));

    // Write to a temporary file first so an interrupted run never leaves a truncated .vdb behind.
    const std::filesystem::path final_path(path);
    const std::filesystem::path tmp_path = final_path.string() + ".tmp";
    {
        openvdb::io::File file(tmp_path.string());
        std::uint32_t compression = openvdb::io::COMPRESS_ACTIVE_MASK;
        compression |= openvdb::io::Archive::hasBloscCompression() ? openvdb::io::COMPRESS_BLOSC
                                                                    : openvdb::io::COMPRESS_ZIP;
        file.setCompression(compression);
        file.write(grids, file_meta);
        file.close();
    }
    std::filesystem::rename(tmp_path, final_path);
    stats.file_bytes = std::size_t(std::filesystem::file_size(final_path));
    return stats;
}

}  // namespace sim
