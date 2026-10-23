#pragma once

#include <string>

namespace sim {

struct RunOptions {
    std::string scene_path;
    std::string device;         // overrides [scene].device when non-empty
    std::string output_dir;     // overrides [output].dir when non-empty
    int max_frames = -1;        // export at most this many frames (-1 = all)
    int preview_every = -1;     // overrides [output].preview_every when >= 0
};

// Runs a scene end to end; returns a process exit code.
int run_scene(const RunOptions& opt);

}  // namespace sim
