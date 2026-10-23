#include "io/vdb_compare.h"

#include <openvdb/openvdb.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>

namespace sim {

namespace {

std::once_flag g_init;

openvdb::GridPtrVecPtr read_all(const std::string& path) {
    openvdb::io::File file(path);
    file.open();
    openvdb::GridPtrVecPtr grids = file.getGrids();
    file.close();
    return grids;
}

template <class GridT, class Diff>
GridDifference compare_typed(const openvdb::GridBase::Ptr& ga, const openvdb::GridBase::Ptr& gb, Diff diff) {
    GridDifference d;
    auto a = openvdb::gridPtrCast<GridT>(ga);
    auto b = openvdb::gridPtrCast<GridT>(gb);
    openvdb::MaskGrid mask(false);
    mask.tree().topologyUnion(a->tree());
    mask.tree().topologyUnion(b->tree());
    mask.tree().voxelizeActiveTiles();
    auto acc_a = a->getConstAccessor();
    auto acc_b = b->getConstAccessor();
    for (auto it = mask.cbeginValueOn(); it; ++it) {
        const openvdb::Coord c = it.getCoord();
        double m = 0.0, va = 0.0, vb = 0.0;
        diff(acc_a.getValue(c), acc_b.getValue(c), m, va, vb);
        ++d.voxels;
        if (m > 0.0) ++d.differing;
        d.max_abs = std::max(d.max_abs, m);
        d.max_value = std::max({d.max_value, va, vb});
    }
    return d;
}

}  // namespace

std::vector<GridDifference> compare_vdb(const std::string& path_a, const std::string& path_b) {
    std::call_once(g_init, [] { openvdb::initialize(); });
    const auto grids_a = read_all(path_a);
    const auto grids_b = read_all(path_b);
    std::map<std::string, openvdb::GridBase::Ptr> by_name_b;
    for (const auto& g : *grids_b) by_name_b[g->getName()] = g;

    std::vector<GridDifference> out;
    for (const auto& ga : *grids_a) {
        GridDifference d;
        d.name = ga->getName();
        const auto it = by_name_b.find(d.name);
        if (it == by_name_b.end() || it->second->type() != ga->type()) {
            d.missing = true;
            out.push_back(d);
            continue;
        }
        const auto gb = it->second;
        by_name_b.erase(it);
        if (ga->isType<openvdb::FloatGrid>()) {
            d = compare_typed<openvdb::FloatGrid>(ga, gb, [](float x, float y, double& m, double& vx, double& vy) {
                m = std::fabs(double(x) - double(y));
                vx = std::fabs(double(x));
                vy = std::fabs(double(y));
            });
        } else if (ga->isType<openvdb::Vec3SGrid>()) {
            d = compare_typed<openvdb::Vec3SGrid>(
                ga, gb, [](const openvdb::Vec3s& x, const openvdb::Vec3s& y, double& m, double& vx, double& vy) {
                    m = vx = vy = 0.0;
                    for (int c = 0; c < 3; ++c) {
                        m = std::max(m, std::fabs(double(x[c]) - double(y[c])));
                        vx = std::max(vx, std::fabs(double(x[c])));
                        vy = std::max(vy, std::fabs(double(y[c])));
                    }
                });
        } else {
            d.missing = true;  // unsupported type: reported, not compared
        }
        d.name = ga->getName();
        out.push_back(d);
    }
    for (const auto& [name, g] : by_name_b) {
        GridDifference d;
        d.name = name;
        d.missing = true;
        out.push_back(d);
    }
    return out;
}

}  // namespace sim
