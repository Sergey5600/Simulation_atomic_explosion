#pragma once
// Device-agnostic memory buffers.

#include <cstddef>
#include <utility>
#include <vector>

#include "core/backend.h"

namespace sim {

void* device_malloc(Device dev, std::size_t bytes);
void device_free(Device dev, void* ptr) noexcept;
void device_copy(void* dst, Device dst_dev, const void* src, Device src_dev, std::size_t bytes);
void device_memset_zero(Device dev, void* ptr, std::size_t bytes);

// Owning, move-only array of trivially copyable T living on one device.
template <class T>
class Buffer {
public:
    Buffer() = default;
    Buffer(Device dev, std::size_t count) { allocate(dev, count); }
    ~Buffer() { release(); }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept { swap(o); }
    Buffer& operator=(Buffer&& o) noexcept {
        if (this != &o) {
            release();
            swap(o);
        }
        return *this;
    }

    // Reallocates; previous contents are lost. New memory is zeroed.
    void allocate(Device dev, std::size_t count) {
        release();
        dev_ = dev;
        size_ = count;
        if (count) {
            ptr_ = static_cast<T*>(device_malloc(dev, count * sizeof(T)));
            device_memset_zero(dev, ptr_, count * sizeof(T));
        }
    }

    void release() noexcept {
        if (ptr_) device_free(dev_, ptr_);
        ptr_ = nullptr;
        size_ = 0;
    }

    void zero() {
        if (size_) device_memset_zero(dev_, ptr_, size_ * sizeof(T));
    }

    void upload(const T* host, std::size_t count) {
        if (count > size_) count = size_;
        device_copy(ptr_, dev_, host, Device::CPU, count * sizeof(T));
    }

    void download(T* host, std::size_t count) const {
        if (count > size_) count = size_;
        device_copy(host, Device::CPU, ptr_, dev_, count * sizeof(T));
    }

    std::vector<T> to_host() const {
        std::vector<T> out(size_);
        download(out.data(), size_);
        return out;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }
    std::size_t bytes() const { return size_ * sizeof(T); }
    Device device() const { return dev_; }

private:
    void swap(Buffer& o) noexcept {
        std::swap(ptr_, o.ptr_);
        std::swap(size_, o.size_);
        std::swap(dev_, o.dev_);
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
    Device dev_ = Device::CPU;
};

}  // namespace sim
