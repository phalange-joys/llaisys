#pragma once
#include "../core/llaisys_core.hpp"
#include "../utils/check.hpp"

#include <vector>
namespace llaisys {
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;

struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
};

class Tensor {
private:
    TensorMeta _meta;
    core::storage_t _storage;
    size_t _offset;
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);

public:
    static tensor_t create(
        const std::vector<size_t> &shape,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        int device = 0);
    ~Tensor() = default;
    // Info
    std::byte *data();
    const std::byte *data() const;
    size_t ndim() const;
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    llaisysDataType_t dtype() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    size_t numel() const;
    size_t elementSize() const;

    std::string info() const;
    void debug() const;

    bool isContiguous() const;

    // Meta Transform
    tensor_t permute(const std::vector<size_t> &order) const;
    tensor_t slice(size_t dim, size_t start, size_t end) const;
    tensor_t view(const std::vector<size_t> &shape) const;

    // Load data from host memory
    void load(const void *src);

    // Challenging features
    tensor_t contiguous() const;
    tensor_t reshape(const std::vector<size_t> &shape) const;
    tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;

    // Additional features
    // Device-agnostic tensor copy (src can be on any device)
    void copyFrom(tensor_t src);

    // Device-agnostic scalar read (D2H if needed, then dereference)
    template <typename T>
    T toScalar() const;
};

// Device-agnostic scalar read (D2H if needed, then dereference)
template <typename T>
T Tensor::toScalar() const {
    CHECK_ARGUMENT(this->elementSize() == sizeof(T),
                   "toScalar type size does not match tensor element size");
    if (this->deviceType() != LLAISYS_DEVICE_CPU) {
        auto cpu_tensor = this->to(LLAISYS_DEVICE_CPU, 0);
        return *reinterpret_cast<const T *>(cpu_tensor->data());
    }
    return *reinterpret_cast<const T *>(this->data());
}

} // namespace llaisys
