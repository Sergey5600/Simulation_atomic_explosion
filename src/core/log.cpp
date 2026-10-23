#include "core/log.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace sim {

namespace {

std::mutex g_log_mutex;
const auto g_start = std::chrono::steady_clock::now();

void vlog(FILE* out, const char* level, const char* fmt, va_list args) {
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::fprintf(out, "[%9.2fs] %s ", t, level);
    std::vfprintf(out, fmt, args);
    std::fputc('\n', out);
    std::fflush(out);
}

}  // namespace

void log_info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog(stdout, "INFO ", fmt, args);
    va_end(args);
}

void log_warn(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog(stdout, "WARN ", fmt, args);
    va_end(args);
}

void log_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog(stderr, "ERROR", fmt, args);
    va_end(args);
}

std::string format_bytes(double bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int u = 0;
    while (bytes >= 1024.0 && u < 4) {
        bytes /= 1024.0;
        ++u;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f %s", bytes, units[u]);
    return buf;
}

std::string format_seconds(double s) {
    char buf[32];
    if (s < 1e-3) std::snprintf(buf, sizeof(buf), "%.1f us", s * 1e6);
    else if (s < 1.0) std::snprintf(buf, sizeof(buf), "%.1f ms", s * 1e3);
    else if (s < 120.0) std::snprintf(buf, sizeof(buf), "%.2f s", s);
    else std::snprintf(buf, sizeof(buf), "%dm %02ds", int(s) / 60, int(s) % 60);
    return buf;
}

}  // namespace sim
