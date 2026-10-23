#pragma once
// Compute backend selection shared by host-only and kernel sources.

#include <cstddef>
#include <string>

#ifndef SIM_HAS_CUDA
#define SIM_HAS_CUDA 0
#endif

#if defined(__CUDACC__)
#define SIM_HD __host__ __device__
#else
#define SIM_HD
#endif

#define SIM_INLINE SIM_HD inline
// Kernel body lambda: [=] captured by value, callable on host and device.
#define SIM_LAMBDA [=] SIM_HD

namespace sim {

enum class Device { CPU, CUDA };

const char* to_string(Device d);

// "cpu", "cuda" or "auto" (CUDA if compiled in and a GPU is present).
Device parse_device(const std::string& name);

bool cuda_compiled();
bool cuda_available();

void device_synchronize(Device d);

// Human readable description, e.g. "NVIDIA GeForce RTX 3060 (sm_86, 12.0 GiB)".
std::string device_description(Device d);

struct DeviceMemoryInfo {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
};
DeviceMemoryInfo device_memory_info(Device d);

}  // namespace sim
