#pragma once
// Export schedule: maps output frame numbers to simulation time.
//
// The fireball happens in milliseconds, the cloud rises for minutes, so the
// sim-time step between exported frames varies: a "ramp" segment grows the
// step geometrically (slow motion easing into real time), a "constant"
// segment keeps it fixed (e.g. 1/24 s = real time at 24 fps).

#include <string>
#include <vector>

namespace sim {

struct ScheduleSegment {
    enum class Mode { Constant, Ramp };
    Mode mode = Mode::Constant;
    int frames = 0;
    double dt_start = 0.0;  // sim seconds per frame at the segment start (Ramp)
    double dt_end = 0.0;    // ... at the segment end (Ramp); for Constant dt_start == dt_end
};

struct ExportSchedule {
    double start_time = 0.0;  // sim time of the first exported frame
    std::vector<ScheduleSegment> segments;

    // Sim time of every exported frame, strictly increasing.
    std::vector<double> frame_times() const;
    int total_frames() const;
};

// Short human readable summary (frames, covered sim time, slow-motion range).
std::string describe(const ExportSchedule& s, double fps);

}  // namespace sim
