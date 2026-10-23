#include "core/backend.h"

#include <omp.h>

#include <cstdio>
#include <stdexcept>

#if SIM_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "core/cuda_check.h"

namespace sim {

const char* to_string(Device d) { return d == Device::CUDA ? "cuda" : "cpu"; }

bool cuda_compiled() { return SIM_HAS_CUDA != 0; }

bool cuda_available() {
#if SIM_HAS_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();  // clear the sticky error
        return false;
    }
    return count > 0;
#else
    return false;
#endif
}

Device parse_device(const std::string& name) {
    if (name == "cpu") return Device::CPU;
    if (name == "cuda") {
        if (!cuda_compiled()) throw std::runtime_error("this build has no CUDA backend; use --device cpu");
        if (!cuda_available()) throw std::runtime_error("no CUDA device found; use --device cpu");
        return Device::CUDA;
    }
    if (name == "auto" || name.empty()) return cuda_available() ? Device::CUDA : Device::CPU;
    throw std::runtime_error("unknown device '" + name + "' (expected cpu, cuda or auto)");
}

void device_synchronize(Device d) {
#if SIM_HAS_CUDA
    if (d == Device::CUDA) detail::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
#else
    (void)d;
#endif
}

std::string device_description(Device d) {
    if (d == Device::CPU) {
        return "CPU (" + std::to_string(omp_get_max_threads()) + " OpenMP threads)";
    }
#if SIM_HAS_CUDA
    int dev = 0;
    detail::check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
    cudaDeviceProp prop{};
    detail::check_cuda(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
    char mem[32];
    std::snprintf(mem, sizeof(mem), "%.1f GiB", double(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    return std::string(prop.name) + " (sm_" + std::to_string(prop.major) + std::to_string(prop.minor) + ", " + mem +
           ", " + std::to_string(prop.multiProcessorCount) + " SMs)";
#else
    return "cuda (not compiled)";
#endif
}

DeviceMemoryInfo device_memory_info(Device d) {
    DeviceMemoryInfo info;
#if SIM_HAS_CUDA
    if (d == Device::CUDA) {
        detail::check_cuda(cudaMemGetInfo(&info.free_bytes, &info.total_bytes), "cudaMemGetInfo");
    }
#else
    (void)d;
#endif
    return info;
}

}  // namespace sim
