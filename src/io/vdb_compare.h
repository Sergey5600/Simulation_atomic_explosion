#pragma once
// Grid-by-grid comparison of two .vdb files, for checking that a change is
// bit-identical or measuring how far it moved the result.

#include <string>
#include <vector>

namespace sim {

struct GridDifference {
    std::string name;
    bool missing = false;          // present in only one file
    std::size_t voxels = 0;        // active voxels in the union
    std::size_t differing = 0;     // voxels whose values differ
    double max_abs = 0.0;          // largest absolute difference (vector grids: per component)
    double max_value = 0.0;        // largest absolute value in either file, for scale
};

std::vector<GridDifference> compare_vdb(const std::string& a, const std::string& b);

}  // namespace sim
