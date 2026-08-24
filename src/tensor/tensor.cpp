#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>
#include <string>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    auto &shape = this->shape();
    auto &strides = this->strides();

    if (shape.empty()) {
        return true; // Scalars are contiguous
    }

    ptrdiff_t expected_stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        if (strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= shape[i]; // stride[i] = product of shape[i+1:]
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    const std::vector<size_t> &shape = this->shape();
    const std::vector<ptrdiff_t> &strides = this->strides();
    if (order.size() != shape.size()) {
        throw std::runtime_error("Tensor::permute order size does not match tensor dimensions");
    }
    std::vector<size_t> new_shape(shape.size());
    std::vector<ptrdiff_t> new_strides(shape.size());
    for (size_t i = 0; i < order.size(); i++) {
        if (order[i] >= shape.size()) {
            throw std::runtime_error("Tensor::permute order index out of bounds");
        }
        new_shape[i] = shape[order[i]];
        new_strides[i] = strides[order[i]]; // 物理地址 = 基地址 + Σ(索引[i] × stride[i])
    }
    TensorMeta new_meta = _meta; // Copy the existing meta
    new_meta.shape = new_shape;
    new_meta.strides = new_strides;
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_numel != this->numel()) {
        throw std::runtime_error("Tensor::view shape does not match total number of elements");
    }
    if (!this->isContiguous()) {
        throw std::runtime_error("Tensor::view requires a contiguous tensor");
    }
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        new_strides[i] = stride;
        stride *= shape[i];
    }
    TensorMeta new_meta = _meta; // Copy the existing meta
    new_meta.shape = shape;
    new_meta.strides = new_strides;
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    const std::vector<size_t> &shape = this->shape();
    if (dim >= shape.size()) {
        throw std::runtime_error("Tensor::slice dimension out of bounds");
    }
    if (start >= end || end > shape[dim]) {
        throw std::runtime_error("Tensor::slice invalid start or end indices");
    }
    TensorMeta new_meta = _meta; // Copy the existing meta
    new_meta.shape[dim] = end - start;
    size_t new_offset = _offset + start * _meta.strides[dim] * this->elementSize(); // slice causes offset change; new offset = old offset + start * stride[dim] * element size;
    new_meta.strides[dim] = _meta.strides[dim];                                     // The stride remains the same for the sliced dimension
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));     //
}

void Tensor::load(const void *src_) {
    size_t bytes = this->numel() * this->elementSize();
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(this->data(), src_, bytes);
    } else {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            bytes,
            LLAISYS_MEMCPY_H2D);
        core::context().runtime().api()->device_synchronize();
    }
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    // create new tensor on the same device
    const std::vector<size_t> &shape = this->shape();
    tensor_t new_tensor = Tensor::create(shape, this->dtype(), this->deviceType(), this->deviceId());

    const size_t total = this->numel();
    const size_t elementSize = this->elementSize();
    const size_t bytes = total * elementSize;
    const std::vector<ptrdiff_t> &strides = this->strides();

    std::vector<std::byte> host_src;
    const std::byte *src = this->data();

    const bool is_cpu = this->deviceType() == LLAISYS_DEVICE_CPU;
    // bring source data to host (no-op on CPU)
    if (!is_cpu) {
        host_src.resize(bytes);
        core::context().runtime().api()->memcpy_sync(
            host_src.data(), src, bytes, LLAISYS_MEMCPY_D2H);
        core::context().runtime().api()->device_synchronize();
        src = host_src.data();
    }

    // rearrange into contiguous layout on host
    std::vector<std::byte> host_dst;
    std::byte *dst = new_tensor->data();
    if (!is_cpu) {
        host_dst.resize(bytes);
        dst = host_dst.data();
    }

    // linear_idx = (i * dim[0] + j) * dim[1] + k for (i, j, k), only related to shape
    for (size_t i = 0; i < total; i++) {
        // compute offset from last dim
        size_t offset = 0;
        size_t remaining = i;
        for (size_t dim = shape.size(); dim-- > 0;) {
            const size_t coord = remaining % shape[dim];
            remaining /= shape[dim];
            offset += coord * static_cast<size_t>(strides[dim]);
        }
        // copy element i
        std::memcpy(dst + i * elementSize, src + offset * elementSize, elementSize);
    }

    if (!is_cpu) {
        core::context().runtime().api()->memcpy_sync(
            new_tensor->data(), host_dst.data(), bytes, LLAISYS_MEMCPY_H2D);
        core::context().runtime().api()->device_synchronize();
    }

    return new_tensor;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    size_t numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (numel != this->numel()) {
        throw std::runtime_error("reshape: total elements mismatch");
    }
    try {
        // if contiguous then view success
        return this->view(shape);
    } catch (const std::runtime_error &e) {
        // if not contiguous then create a contiguous tensor and view(shape)
        if (std::string(e.what()).find("Tensor::view") == 0) {
            auto contiguous_tensor = this->contiguous();
            return contiguous_tensor->view(shape);
        }
        throw;
    }
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (this->deviceType() == device_type && this->deviceId() == device) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    tensor_t src = this->contiguous();
    size_t bytes = src->numel() * src->elementSize();
    core::storage_t new_storage;

    // allocate storage
    if (device_type == LLAISYS_DEVICE_CPU) {
        new_storage = core::context().runtime().allocateHostStorage(bytes);
    } else {
        core::context().setDevice(device_type, device);
        new_storage = core::context().runtime().allocateDeviceStorage(bytes);
    }

    if (src->deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        // cpu to cpu
        std::memcpy(new_storage->memory(), src->data(), bytes);
    } else if (src->deviceType() == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) {
        // cpu to device
        core::context().runtime().api()->memcpy_sync(
            new_storage->memory(),
            src->data(),
            bytes,
            LLAISYS_MEMCPY_H2D);
        core::context().runtime().api()->device_synchronize();
    } else if (src->deviceType() != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        // device to cpu
        core::context().runtime().api()->memcpy_sync(
            new_storage->memory(),
            src->data(),
            bytes,
            LLAISYS_MEMCPY_D2H);
        core::context().runtime().api()->device_synchronize();
    } else {
        // device to device
        core::context().runtime().api()->memcpy_sync(
            new_storage->memory(),
            src->data(),
            bytes,
            LLAISYS_MEMCPY_D2D);
        core::context().runtime().api()->device_synchronize();
    }
    return std::shared_ptr<Tensor>(new Tensor(src->_meta, new_storage, 0));
}

void Tensor::copyFrom(tensor_t src) {
    // Use to() to handle any device combination (H2D, D2D, D2H, H2H)
    tensor_t src_on_same_device = src->to(this->deviceType(), this->deviceId());
    size_t bytes = this->numel() * this->elementSize();
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(this->data(), src_on_same_device->data(), bytes);
    } else {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_on_same_device->data(),
            bytes,
            LLAISYS_MEMCPY_D2D);
        core::context().runtime().api()->device_synchronize();
    }
}
} // namespace llaisys
