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

constexpr int kReduceBlock = 256;

// One 8-byte device word reused by every reduction (they run one at a time on the default stream).
inline void* reduction_scratch() {
    static void* ptr = nullptr;
    if (!ptr) check_cuda(cudaMalloc(&ptr, sizeof(double)), "cudaMalloc(reduction scratch)");
    return ptr;
}

SIM_INLINE void index_to_ijk(std::int64_t idx, Int3 n, int& i, int& j, int& k) {
    i = int(idx % n.x);
    const std::int64_t r = idx / n.x;
    j = int(r % n.y);
    k = int(r / n.y);
}

template <class F>
__global__ void sum_kernel(Int3 n, F f, double* out) {
    const std::int64_t total = n.volume();
    double v = 0.0;
    for (std::int64_t idx = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += std::int64_t(blockDim.x) * gridDim.x) {
        int i, j, k;
        index_to_ijk(idx, n, i, j, k);
        v += double(f(i, j, k));
    }
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    __shared__ double warp_sums[kReduceBlock / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < kReduceBlock / 32 ? warp_sums[lane] : 0.0;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) atomicAdd(out, v);
    }
}

// f must return values >= 0: their IEEE bit patterns then order like unsigned integers.
template <class F>
__global__ void max_kernel(Int3 n, F f, unsigned int* out) {
    const std::int64_t total = n.volume();
    float v = 0.f;
    for (std::int64_t idx = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += std::int64_t(blockDim.x) * gridDim.x) {
        int i, j, k;
        index_to_ijk(idx, n, i, j, k);
        v = fmaxf(v, float(f(i, j, k)));
    }
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    __shared__ float warp_max[kReduceBlock / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_max[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < kReduceBlock / 32 ? warp_max[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
        if (lane == 0) atomicMax(out, __float_as_uint(fmaxf(v, 0.f)));
    }
}

inline unsigned int reduce_grid_size(Int3 n) {
    const std::int64_t blocks = (n.volume() + kReduceBlock - 1) / kReduceBlock;
    return unsigned(blocks < 1024 ? (blocks > 0 ? blocks : 1) : 1024);
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

// Sum of f(i, j, k) over the range, accumulated in double precision. Synchronous.
template <class F>
double parallel_sum(Device dev, Int3 n, F f) {
    if (n.x <= 0 || n.y <= 0 || n.z <= 0) return 0.0;
    if (dev == Device::CUDA) {
#if SIM_HAS_CUDA
        double* acc = static_cast<double*>(detail::reduction_scratch());
        detail::check_cuda(cudaMemset(acc, 0, sizeof(double)), "cudaMemset");
        detail::sum_kernel<<<detail::reduce_grid_size(n), detail::kReduceBlock>>>(n, f, acc);
        detail::check_cuda(cudaGetLastError(), "parallel_sum launch");
        double result = 0.0;
        detail::check_cuda(cudaMemcpy(&result, acc, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
        return result;
#else
        throw std::runtime_error("CUDA backend is not compiled in");
#endif
    }
    const int rows = n.y * n.z;
    double sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (int row = 0; row < rows; ++row) {
        const int j = row % n.y;
        const int k = row / n.y;
        double s = 0.0;
        for (int i = 0; i < n.x; ++i) s += double(f(i, j, k));
        sum += s;
    }
    return sum;
}

// Maximum of f(i, j, k) over the range; f must return values >= 0. Synchronous.
template <class F>
float parallel_max(Device dev, Int3 n, F f) {
    if (n.x <= 0 || n.y <= 0 || n.z <= 0) return 0.f;
    if (dev == Device::CUDA) {
#if SIM_HAS_CUDA
        unsigned int* acc = static_cast<unsigned int*>(detail::reduction_scratch());
        detail::check_cuda(cudaMemset(acc, 0, sizeof(unsigned int)), "cudaMemset");
        detail::max_kernel<<<detail::reduce_grid_size(n), detail::kReduceBlock>>>(n, f, acc);
        detail::check_cuda(cudaGetLastError(), "parallel_max launch");
        float result = 0.f;
        detail::check_cuda(cudaMemcpy(&result, acc, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
        return result;
#else
        throw std::runtime_error("CUDA backend is not compiled in");
#endif
    }
    const int rows = n.y * n.z;
    float result = 0.f;
    // MSVC only supports OpenMP 2.0, which has no max reduction.
#pragma omp parallel
    {
        float local = 0.f;
#pragma omp for schedule(static)
        for (int row = 0; row < rows; ++row) {
            const int j = row % n.y;
            const int k = row / n.y;
            for (int i = 0; i < n.x; ++i) {
                const float v = float(f(i, j, k));
                local = v > local ? v : local;
            }
        }
#pragma omp critical
        result = local > result ? local : result;
    }
    return result;
}

}  // namespace sim
