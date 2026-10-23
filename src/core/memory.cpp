#include "core/memory.h"

#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"

namespace sim {

namespace {
constexpr std::align_val_t kCpuAlignment{64};
}

void* device_malloc(Device dev, std::size_t bytes) {
    if (bytes == 0) return nullptr;
    if (dev == Device::CPU) return ::operator new(bytes, kCpuAlignment);
#if SIM_HAS_CUDA
    void* ptr = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error("cudaMalloc failed for " + std::to_string(bytes / (1024 * 1024)) +
                                 " MiB: " + cudaGetErrorString(err));
    }
    return ptr;
#else
    throw std::runtime_error("CUDA backend is not compiled in");
#endif
}

void device_free(Device dev, void* ptr) noexcept {
    if (!ptr) return;
    if (dev == Device::CPU) {
        ::operator delete(ptr, kCpuAlignment);
        return;
    }
#if SIM_HAS_CUDA
    cudaFree(ptr);
#endif
}

void device_copy(void* dst, Device dst_dev, const void* src, Device src_dev, std::size_t bytes) {
    if (bytes == 0) return;
    if (dst_dev == Device::CPU && src_dev == Device::CPU) {
        std::memcpy(dst, src, bytes);
        return;
    }
#if SIM_HAS_CUDA
    cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
    if (dst_dev == Device::CPU) kind = cudaMemcpyDeviceToHost;
    else if (src_dev == Device::CPU) kind = cudaMemcpyHostToDevice;
    detail::check_cuda(cudaMemcpy(dst, src, bytes, kind), "cudaMemcpy");
#else
    throw std::runtime_error("CUDA backend is not compiled in");
#endif
}

void device_memset_zero(Device dev, void* ptr, std::size_t bytes) {
    if (bytes == 0) return;
    if (dev == Device::CPU) {
        std::memset(ptr, 0, bytes);
        return;
    }
#if SIM_HAS_CUDA
    detail::check_cuda(cudaMemset(ptr, 0, bytes), "cudaMemset");
#else
    throw std::runtime_error("CUDA backend is not compiled in");
#endif
}

}  // namespace sim
