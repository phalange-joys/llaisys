#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seqlen, size_t nhead, size_t d) {
#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(seqlen); i++) {
        for (size_t j = 0; j < d / 2; j++) {
            float angle = static_cast<float>(pos_ids[i]) / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(d)); // avoid MSVC C4244 under /WX
            float cos_angle = std::cos(angle);                                                                                    // TODO precompute a table
            float sin_angle = std::sin(angle);
            for (size_t h = 0; h < nhead; h++) {
                size_t idx = i * nhead * d + h * d + j;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    float a_ij = llaisys::utils::cast<float>(in[idx]);
                    float b_ij = llaisys::utils::cast<float>(in[idx + d / 2]);
                    out[idx] = llaisys::utils::cast<T>(a_ij * cos_angle - b_ij * sin_angle);
                    out[idx + d / 2] = llaisys::utils::cast<T>(a_ij * sin_angle + b_ij * cos_angle);
                } else {
                    out[idx] = in[idx] * cos_angle - in[idx + d / 2] * sin_angle;
                    out[idx + d / 2] = in[idx] * sin_angle + in[idx + d / 2] * cos_angle;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu