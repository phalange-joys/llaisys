#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void decode_linear_(T *out, const T *in, const T *weight, const T *bias, size_t seqlen, size_t in_features, size_t out_features) {
#pragma omp parallel for
    for (int64_t j = 0; j < (int64_t)out_features; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < in_features; k++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                sum += llaisys::utils::cast<float>(in[k]) * llaisys::utils::cast<float>(weight[j * in_features + k]);
            } else {
                sum += in[k] * weight[j * in_features + k];
            }
        }

        float bias_val = 0.0f;
        if (bias != nullptr) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                bias_val = llaisys::utils::cast<float>(bias[j]);
            } else {
                bias_val = bias[j];
            }
        }
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            out[j] = llaisys::utils::cast<T>(sum + bias_val);
        } else {
            out[j] = sum + bias_val;
        }
    }
}

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t seqlen, size_t in_features, size_t out_features) {
    if (seqlen == 1) {
        decode_linear_(out, in, weight, bias, seqlen, in_features, out_features);
        return;
    }
#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(seqlen); i++) {
        // row cahce
        std::vector<float> in_row(in_features);
        for (size_t k = 0; k < in_features; k++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                in_row[k] = llaisys::utils::cast<float>(in[i * in_features + k]);
            } else {
                in_row[k] = in[i * in_features + k];
            }
        }

        for (size_t j = 0; j < out_features; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < in_features; k++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum += in_row[k] * llaisys::utils::cast<float>(weight[j * in_features + k]);
                } else {
                    sum += in_row[k] * weight[j * in_features + k];
                }
            }

            float bias_val = 0.0f;
            if (bias != nullptr) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    bias_val = llaisys::utils::cast<float>(bias[j]);
                } else {
                    bias_val = bias[j];
                }
            }

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * out_features + j] = llaisys::utils::cast<T>(sum + bias_val);
            } else {
                out[i * out_features + j] = sum + bias_val;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t seqlen, size_t in_features, size_t out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), seqlen, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), seqlen, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), seqlen, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu