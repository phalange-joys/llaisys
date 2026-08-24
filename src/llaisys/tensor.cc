#include "llaisys_tensor.hpp"

#include "../utils/check.hpp"

#include <cstring>
#include <vector>

__C {
    llaisysTensor_t tensorCreate(
        size_t * shape,
        size_t ndim,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id) {
        std::vector<size_t> shape_vec(shape, shape + ndim);
        return new LlaisysTensor{llaisys::Tensor::create(shape_vec, dtype, device_type, device_id)};
    }

    void tensorDestroy(
        llaisysTensor_t tensor) {
        delete tensor;
    }

    void *tensorGetData(
        llaisysTensor_t tensor) {
        return tensor->tensor->data();
    }

    size_t tensorGetNdim(
        llaisysTensor_t tensor) {
        return tensor->tensor->ndim();
    }

    void tensorGetShape(
        llaisysTensor_t tensor,
        size_t * shape) {
        std::copy(tensor->tensor->shape().begin(), tensor->tensor->shape().end(), shape);
    }

    void tensorGetStrides(
        llaisysTensor_t tensor,
        ptrdiff_t * strides) {
        std::copy(tensor->tensor->strides().begin(), tensor->tensor->strides().end(), strides);
    }

    llaisysDataType_t tensorGetDataType(
        llaisysTensor_t tensor) {
        return tensor->tensor->dtype();
    }

    llaisysDeviceType_t tensorGetDeviceType(
        llaisysTensor_t tensor) {
        return tensor->tensor->deviceType();
    }

    int tensorGetDeviceId(
        llaisysTensor_t tensor) {
        return tensor->tensor->deviceId();
    }

    void tensorDebug(
        llaisysTensor_t tensor) {
        tensor->tensor->debug();
    }

    uint8_t tensorIsContiguous(
        llaisysTensor_t tensor) {
        return uint8_t(tensor->tensor->isContiguous());
    }

    void tensorLoad(
        llaisysTensor_t tensor,
        const void *data) {
        tensor->tensor->load(data);
    }

    llaisysTensor_t tensorView(
        llaisysTensor_t tensor,
        size_t * shape,
        size_t ndim) {
        std::vector<size_t> shape_vec(shape, shape + ndim);
        return new LlaisysTensor{tensor->tensor->view(shape_vec)};
    }

    llaisysTensor_t tensorPermute(
        llaisysTensor_t tensor,
        size_t * order) {
        std::vector<size_t> order_vec(order, order + tensor->tensor->ndim());
        return new LlaisysTensor{tensor->tensor->permute(order_vec)};
    }

    llaisysTensor_t tensorSlice(
        llaisysTensor_t tensor,
        size_t dim,
        size_t start,
        size_t end) {
        return new LlaisysTensor{tensor->tensor->slice(dim, start, end)};
    }

    llaisysTensor_t tensorContiguous(
        llaisysTensor_t tensor) {
        return new LlaisysTensor{tensor->tensor->contiguous()};
    }

    llaisysTensor_t tensorReshape(
        llaisysTensor_t tensor,
        size_t * shape,
        size_t ndim) {
        std::vector<size_t> shape_vec(shape, shape + ndim);
        return new LlaisysTensor{tensor->tensor->reshape(shape_vec)};
    }

    llaisysTensor_t tensorTo(
        llaisysTensor_t tensor,
        llaisysDeviceType_t device_type,
        int device_id) {
        return new LlaisysTensor{tensor->tensor->to(device_type, device_id)};
    }

    void tensorCopyFrom(
        llaisysTensor_t tensor,
        llaisysTensor_t src) {
        tensor->tensor->copyFrom(src->tensor);
    }

    void tensorToScalar(
        llaisysTensor_t tensor,
        void *out) {
        const llaisys::tensor_t &t = tensor->tensor;
        CHECK_ARGUMENT(t->numel() == 1, "tensorToScalar requires a tensor with exactly one element");
        switch (t->dtype()) {
        case LLAISYS_DTYPE_BYTE:
            *static_cast<char *>(out) = t->toScalar<char>();
            break;
        case LLAISYS_DTYPE_BOOL:
            *static_cast<bool *>(out) = t->toScalar<bool>();
            break;
        case LLAISYS_DTYPE_I8:
            *static_cast<int8_t *>(out) = t->toScalar<int8_t>();
            break;
        case LLAISYS_DTYPE_I16:
            *static_cast<int16_t *>(out) = t->toScalar<int16_t>();
            break;
        case LLAISYS_DTYPE_I32:
            *static_cast<int32_t *>(out) = t->toScalar<int32_t>();
            break;
        case LLAISYS_DTYPE_I64:
            *static_cast<int64_t *>(out) = t->toScalar<int64_t>();
            break;
        case LLAISYS_DTYPE_U8:
            *static_cast<uint8_t *>(out) = t->toScalar<uint8_t>();
            break;
        case LLAISYS_DTYPE_U16:
            *static_cast<uint16_t *>(out) = t->toScalar<uint16_t>();
            break;
        case LLAISYS_DTYPE_U32:
            *static_cast<uint32_t *>(out) = t->toScalar<uint32_t>();
            break;
        case LLAISYS_DTYPE_U64:
            *static_cast<uint64_t *>(out) = t->toScalar<uint64_t>();
            break;
        case LLAISYS_DTYPE_F16:
            *static_cast<llaisys::fp16_t *>(out) = t->toScalar<llaisys::fp16_t>();
            break;
        case LLAISYS_DTYPE_BF16:
            *static_cast<llaisys::bf16_t *>(out) = t->toScalar<llaisys::bf16_t>();
            break;
        case LLAISYS_DTYPE_F32:
            *static_cast<float *>(out) = t->toScalar<float>();
            break;
        case LLAISYS_DTYPE_F64:
            *static_cast<double *>(out) = t->toScalar<double>();
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(t->dtype());
        }
    }
}
