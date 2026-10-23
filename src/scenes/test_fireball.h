#pragma once
// Analytic fireball used to test the export pipeline end to end (M0).
// It only looks roughly like a fireball; there is no physics here.

#include <cstdint>

#include "core/grid.h"
#include "io/config.h"

namespace sim {

struct TestFireballFields {
    ScalarField* density = nullptr;
    ScalarField* temperature = nullptr;
    ScalarField* flame = nullptr;
    ScalarField* vel_x = nullptr;
    ScalarField* vel_y = nullptr;
    ScalarField* vel_z = nullptr;
};

void generate_test_fireball(const TestFireballConfig& cfg, std::uint32_t seed, double time, TestFireballFields& out);

}  // namespace sim
