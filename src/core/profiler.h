#pragma once
// Optional per-stage timing of a solver step (--profile). When enabled, every lap
// synchronises the device, which costs a little speed, so it is off by default.

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "core/backend.h"
#include "core/log.h"

namespace sim {

class StageProfiler {
public:
    void enable(Device dev) {
        dev_ = dev;
        on_ = true;
    }
    bool enabled() const { return on_; }

    // Call at the start of a step.
    void start() {
        if (!on_) return;
        device_synchronize(dev_);
        lap_.reset();
        ++steps_;
    }
    // Time since the previous lap (or start) is added to `name`.
    void lap(const char* name) {
        if (!on_) return;
        device_synchronize(dev_);
        const double t = lap_.seconds();
        lap_.reset();
        for (auto& [n, total] : totals_) {
            if (n == name) {
                total += t;
                return;
            }
        }
        totals_.emplace_back(name, t);
    }
    // "stage 12.3 ms (40%), ..." averaged per step since the last report; empty when off or idle.
    std::string report_and_reset() {
        if (!on_ || steps_ == 0) return {};
        double sum = 0.0;
        for (const auto& [n, t] : totals_) sum += t;
        std::string out;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%.1f ms/step:", 1e3 * sum / steps_);
        out = buf;
        for (const auto& [n, t] : totals_) {
            std::snprintf(buf, sizeof(buf), " %s %.1f (%.0f%%)", n.c_str(), 1e3 * t / steps_, sum > 0 ? 100.0 * t / sum : 0.0);
            out += buf;
        }
        totals_.clear();
        steps_ = 0;
        return out;
    }

private:
    Device dev_ = Device::CPU;
    bool on_ = false;
    Timer lap_;
    int steps_ = 0;
    std::vector<std::pair<std::string, double>> totals_;
};

}  // namespace sim
