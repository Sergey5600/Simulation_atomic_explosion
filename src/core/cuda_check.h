#pragma once
// Error checking for CUDA runtime calls. Safe to include from any source.

#include "core/backend.h"

#if SIM_HAS_CUDA
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace sim::detail {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
    }
}

}  // namespace sim::detail
#endif
