#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include "app/run.h"
#include "core/backend.h"
#include "core/log.h"

namespace {

void print_usage() {
    std::printf(
        "NukeSim - nuclear explosion volume simulator\n"
        "\n"
        "usage: nukesim <scene.toml> [options]\n"
        "       nukesim --info\n"
        "\n"
        "options:\n"
        "  --device cpu|cuda|auto   compute device (default: from scene, usually auto)\n"
        "  --out DIR                output directory (default: from scene)\n"
        "  --frames N               export at most N frames\n"
        "  --preview-every N        write a PNG preview every N frames (0 = off)\n"
        "  --info                   print available devices and exit\n");
}

int to_int(const char* s, const char* flag) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') throw std::runtime_error(std::string("invalid number for ") + flag + ": " + s);
    return int(v);
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
            } else if (arg == "--device") {
                opt.device = next("--device");
            } else if (arg == "--out") {
                opt.output_dir = next("--out");
            } else if (arg == "--frames") {
                opt.max_frames = to_int(next("--frames"), "--frames");
            } else if (arg == "--preview-every") {
                opt.preview_every = to_int(next("--preview-every"), "--preview-every");
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
