#include "io/manifest.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace sim {

namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default: out += c;
        }
    }
    return out;
}

}  // namespace

void write_manifest(const std::string& path, const Manifest& m) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write " + tmp);
        char buf[1024];
        f << "{\n";
        f << "  \"format\": \"nukesim-manifest-1\",\n";
        f << "  \"scene\": \"" << json_escape(m.scene_name) << "\",\n";
        f << "  \"prefix\": \"" << json_escape(m.prefix) << "\",\n";
        std::snprintf(buf, sizeof(buf), "  \"fps\": %.6g,\n  \"world_scale\": %.9g,\n", m.fps, double(m.world_scale));
        f << buf;
        f << "  \"fields\": [";
        for (std::size_t i = 0; i < m.fields.size(); ++i) f << (i ? ", " : "") << '"' << json_escape(m.fields[i]) << '"';
        f << "],\n";
        f << "  \"frames\": [\n";
        for (std::size_t i = 0; i < m.frames.size(); ++i) {
            const ManifestFrame& fr = m.frames[i];
            std::snprintf(buf, sizeof(buf),
                          "    {\"number\": %d, \"file\": \"%s\", \"sim_time\": %.9g, \"sim_dt\": %.9g, "
                          "\"voxel_size\": %.9g, \"domain_min\": [%.6g, %.6g, %.6g], \"domain_max\": [%.6g, %.6g, %.6g], "
                          "\"cloud_min\": [%.6g, %.6g, %.6g], \"cloud_max\": [%.6g, %.6g, %.6g], "
                          "\"light_position\": [%.6g, %.6g, %.6g], \"light_power\": %.9g, "
                          "\"active_voxels\": %zu, \"file_bytes\": %zu, \"rest_weight\": %.6g, "
                          "\"vorticity_ref\": %.6g}%s\n",
                          fr.number, json_escape(fr.file).c_str(), fr.sim_time, fr.sim_dt, double(fr.voxel_size),
                          double(fr.domain_min.x), double(fr.domain_min.y), double(fr.domain_min.z),
                          double(fr.domain_max.x), double(fr.domain_max.y), double(fr.domain_max.z),
                          double(fr.cloud_min.x), double(fr.cloud_min.y), double(fr.cloud_min.z),
                          double(fr.cloud_max.x), double(fr.cloud_max.y), double(fr.cloud_max.z),
                          double(fr.light_position.x), double(fr.light_position.y), double(fr.light_position.z),
                          fr.light_power, fr.active_voxels, fr.file_bytes, double(fr.rest_weight),
                          double(fr.vorticity_ref), i + 1 < m.frames.size() ? "," : "");
            f << buf;
        }
        f << "  ]\n}\n";
    }
    std::filesystem::rename(tmp, path);
}

}  // namespace sim
