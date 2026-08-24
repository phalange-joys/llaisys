#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t seqlen, size_t in_features) {
#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(seqlen); i++) {
        float sum_square = 0.0f;
        for (size_t j = 0; j < in_features; j++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[i * in_features + j]);
            } else {
                val = in[i * in_features + j];
            }
            sum_square += val * val;
        }
        float mean_square = sum_square / static_cast<float>(in_features) + eps;
        float rms = std::sqrt(mean_square);
        float inv_rms = 1.0f / rms;

        for (size_t j = 0; j < in_features; j++) {
            float result = 0.0f;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                result = llaisys::utils::cast<float>(in[i * in_features + j]) * llaisys::utils::cast<float>(weight[j]) / rms;
            } else {
                result = in[i * in_features + j] * weight[j] * inv_rms;
            }

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * in_features + j] = llaisys::utils::cast<T>(result);
            } else {
                out[i * in_features + j] = result;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t sesqlen, size_t in_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, sesqlen, in_features);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), eps, sesqlen, in_features);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), eps, sesqlen, in_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu