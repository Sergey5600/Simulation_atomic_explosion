#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <filesystem>
#include <map>
#include <vector>

#include "app/run.h"
#include "core/backend.h"
#include "core/log.h"
#include "io/vdb_compare.h"

namespace {

void print_usage() {
    std::printf(
        "NukeSim - nuclear explosion volume simulator\n"
        "\n"
        "usage: nukesim <scene.toml> [options]\n"
        "       nukesim --info\n"
        "       nukesim --compare A B      compare two .vdb files or two output folders\n"
        "\n"
        "options:\n"
        "  --device cpu|cuda|auto   compute device (default: from scene, usually auto)\n"
        "  --out DIR                output directory (default: from scene)\n"
        "  --frames N               export at most N frames\n"
        "  --preview-every N        write a PNG preview every N frames (0 = off)\n"
        "  --slices-every N         write density/vorticity slice PNGs every N frames (0 = off)\n"
        "  --profile                print the time of each solver stage per step\n"
        "  --info                   print available devices and exit\n");
}

int to_int(const char* s, const char* flag) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') throw std::runtime_error(std::string("invalid number for ") + flag + ": " + s);
    return int(v);
}

// Prints how far two runs differ, grid by grid; folders are compared frame by frame.
// Returns 0 when everything is identical, 1 otherwise.
int compare(const std::string& a, const std::string& b) {
    namespace fs = std::filesystem;
    std::vector<std::pair<fs::path, fs::path>> pairs;
    if (fs::is_directory(a)) {
        for (const auto& e : fs::directory_iterator(a)) {
            if (e.path().extension() != ".vdb") continue;
            const fs::path other = fs::path(b) / e.path().filename();
            if (fs::exists(other)) pairs.emplace_back(e.path(), other);
        }
        std::sort(pairs.begin(), pairs.end());
    } else {
        pairs.emplace_back(a, b);
    }
    if (pairs.empty()) {
        std::printf("no matching .vdb files\n");
        return 1;
    }
    struct Worst {
        sim::GridDifference d;
        std::string frame;
        int missing = 0;
    };
    std::map<std::string, Worst> worst;
    for (const auto& [pa, pb] : pairs) {
        for (const sim::GridDifference& d : sim::compare_vdb(pa.string(), pb.string())) {
            Worst& w = worst[d.name];
            if (d.missing) ++w.missing;
            if (d.max_abs > w.d.max_abs || w.frame.empty()) {
                w.d = d;
                w.frame = pa.filename().string();
            }
            w.d.max_value = std::max(w.d.max_value, d.max_value);
        }
    }
    bool identical = true;
    std::printf("%zu frame(s) compared\n", pairs.size());
    for (const auto& [name, w] : worst) {
        if (w.missing) {
            std::printf("  %-12s missing in %d frame(s)\n", name.c_str(), w.missing);
            identical = false;
            continue;
        }
        if (w.d.max_abs > 0.0) identical = false;
        std::printf("  %-12s max |diff| %.3g (values up to %.3g), worst in %s: %zu of %zu voxels differ\n", name.c_str(),
                    w.d.max_abs, w.d.max_value, w.frame.c_str(), w.d.differing, w.d.voxels);
    }
    std::printf(identical ? "identical\n" : "different\n");
    return identical ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        sim::RunOptions opt;
        for (int a = 1; a < argc; ++a) {
            const std::string arg = argv[a];
            auto next = [&](const char* flag) -> const char* {
                if (a + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
                return argv[++a];
            };
            if (arg == "-h" || arg == "--help") {
                print_usage();
                return 0;
            } else if (arg == "--info") {
                std::printf("CUDA compiled: %s\n", sim::cuda_compiled() ? "yes" : "no");
                std::printf("CPU : %s\n", sim::device_description(sim::Device::CPU).c_str());
                if (sim::cuda_available()) std::printf("CUDA: %s\n", sim::device_description(sim::Device::CUDA).c_str());
                else std::printf("CUDA: no device available\n");
                return 0;
            } else if (arg == "--compare") {
                if (a + 2 >= argc) throw std::runtime_error("--compare needs two paths");
                return compare(argv[a + 1], argv[a + 2]);
            } else if (arg == "--device") {
                opt.device = next("--device");
            } else if (arg == "--out") {
                opt.output_dir = next("--out");
            } else if (arg == "--frames") {
                opt.max_frames = to_int(next("--frames"), "--frames");
            } else if (arg == "--preview-every") {
                opt.preview_every = to_int(next("--preview-every"), "--preview-every");
            } else if (arg == "--slices-every") {
                opt.slices_every = to_int(next("--slices-every"), "--slices-every");
            } else if (arg == "--profile") {
                opt.profile = true;
            } else if (!arg.empty() && arg[0] == '-') {
                throw std::runtime_error("unknown option " + arg);
            } else if (opt.scene_path.empty()) {
                opt.scene_path = arg;
            } else {
                throw std::runtime_error("only one scene file may be given");
            }
        }
        if (opt.scene_path.empty()) {
            print_usage();
            return 2;
        }
        return sim::run_scene(opt);
    } catch (const std::exception& e) {
        sim::log_error("%s", e.what());
        return 1;
    }
}
