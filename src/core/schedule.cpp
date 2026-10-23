#include "core/schedule.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace sim {

std::vector<double> ExportSchedule::frame_times() const {
    std::vector<double> times;
    times.reserve(std::size_t(total_frames()));
    double t = start_time;
    bool first = true;
    for (const ScheduleSegment& seg : segments) {
        if (seg.frames <= 0) continue;
        if (seg.dt_start <= 0.0 || seg.dt_end <= 0.0) throw std::invalid_argument("schedule: dt must be > 0");
        for (int f = 0; f < seg.frames; ++f) {
            double dt = seg.dt_start;
            if (seg.mode == ScheduleSegment::Mode::Ramp && seg.frames > 1) {
                const double u = double(f) / double(seg.frames - 1);
                dt = seg.dt_start * std::pow(seg.dt_end / seg.dt_start, u);
            }
            if (!first) t += dt;
            first = false;
            times.push_back(t);
        }
    }
    return times;
}

int ExportSchedule::total_frames() const {
    int n = 0;
    for (const ScheduleSegment& seg : segments) n += seg.frames > 0 ? seg.frames : 0;
    return n;
}

std::string describe(const ExportSchedule& s, double fps) {
    const std::vector<double> t = s.frame_times();
    if (t.empty()) return "empty schedule";
    double min_dt = 1e30, max_dt = 0.0;
    for (std::size_t i = 1; i < t.size(); ++i) {
        min_dt = std::fmin(min_dt, t[i] - t[i - 1]);
        max_dt = std::fmax(max_dt, t[i] - t[i - 1]);
    }
    if (t.size() < 2) min_dt = max_dt = 0.0;
    const double real_dt = 1.0 / fps;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%zu frames (%.1f s of video at %.0f fps) covering sim time %.4f .. %.2f s, "
                  "slow motion %.0fx .. %.2fx",
                  t.size(), double(t.size()) / fps, fps, t.front(), t.back(),
                  min_dt > 0 ? real_dt / min_dt : 0.0, max_dt > 0 ? real_dt / max_dt : 0.0);
    return buf;
}

}  // namespace sim
