#pragma once
// parallel_for over a 3D index range on the CPU (OpenMP) or the GPU (CUDA).
//
// Only include this header from kernel sources (SIM_KERNEL_SOURCES in
// CMakeLists.txt): in CUDA builds they are compiled by nvcc, so the same
// lambda body is instantiated for both host and device.

#include <stdexcept>

#include "core/backend.h"
#include "core/math.h"

#if SIM_HAS_CUDA && !defined(__CUDACC__)
#error "core/parallel.h must only be included from kernel sources compiled as CUDA (see SIM_KERNEL_SOURCES)"
#endif

#if SIM_HAS_CUDA
#include "core/cuda_check.h"
#endif

namespace sim {

#if SIM_HAS_CUDA
namespace detail {
template <class F>
__global__ void parallel_for_kernel(Int3 n, F f) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int j = int(blockIdx.y * blockDim.y + threadIdx.y);
    const int k = int(blockIdx.z * blockDim.z + threadIdx.z);
    if (i < n.x && j < n.y && k < n.z) f(i, j, k);
}
}  // namespace detail
#endif

// Calls f(i, j, k) for every cell of the n.x * n.y * n.z range. Launches are
// asynchronous on CUDA; use device_synchronize() before reading results on host.
template <class F>
void parallel_for(Device dev, Int3 n, F f) {
    if (n.x <= 0 || n.y <= 0 || n.z <= 0) return;
    if (dev == Device::CUDA) {
#if SIM_HAS_CUDA
        const dim3 block(32, 4, 2);
        const dim3 grid((n.x + block.x - 1) / block.x, (n.y + block.y - 1) / block.y,
                        (n.z + block.z - 1) / block.z);
        detail::parallel_for_kernel<<<grid, block>>>(n, f);
        detail::check_cuda(cudaGetLastError(), "parallel_for launch");
#else
        throw std::runtime_error("CUDA backend is not compiled in");
#endif
    } else {
        const int rows = n.y * n.z;
#pragma omp parallel for schedule(static)
        for (int row = 0; row < rows; ++row) {
            const int j = row % n.y;
            const int k = row / n.y;
            for (int i = 0; i < n.x; ++i) f(i, j, k);
        }
    }
}

}  // namespace sim
