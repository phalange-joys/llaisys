#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// arm64 torch uses NEON f16 arithmetic — every intermediate is rounded to
// f16. We simulate this by rounding through f16 cast on arm64 only.
// On x86 torch uses f32 intermediates, so f16_round is a compile-time no-op.
#if defined(__aarch64__)
#define LLAISYS_ARM64_F16_ROUND 1
#else
#define LLAISYS_ARM64_F16_ROUND 0
#endif

namespace {
// f32 → f16 → f32. No-op on non-arm64 (compile-time eliminated).
template <bool Enable>
inline float f16_round(float v) {
    if constexpr (Enable) {
        return llaisys::utils::cast<float>(llaisys::utils::cast<llaisys::fp16_t>(v));
    } else {
        return v;
    }
}
} // namespace

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t seqlen, size_t in_features) {

    constexpr bool is_f16 = std::is_same_v<T, llaisys::fp16_t>;
    constexpr bool do_round = is_f16 && LLAISYS_ARM64_F16_ROUND;

#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(seqlen); i++) {
        // --- Step 1: mean of squares ---
        float sum_square = 0.0f;
        for (size_t j = 0; j < in_features; j++) {
            float val;
            if constexpr (is_f16 || std::is_same_v<T, llaisys::bf16_t>) {
                val = llaisys::utils::cast<float>(in[i * in_features + j]);
            } else {
                val = in[i * in_features + j];
            }
            sum_square += val * val;
        }

        float mean_sq = sum_square / static_cast<float>(in_features);
        mean_sq = f16_round<do_round>(mean_sq);  // match torch.mean f16 rounding on arm64
        mean_sq += eps;

        // --- Step 2: rsqrt ---
        float inv_rms = 1.0f / std::sqrt(mean_sq);
        inv_rms = f16_round<do_round>(inv_rms);  // match torch.rsqrt f16 rounding on arm64

        // --- Step 3: x * inv_rms * w ---
        for (size_t j = 0; j < in_features; j++) {
            float result;
            if constexpr (is_f16) {
                float x_f32  = llaisys::utils::cast<float>(in[i * in_features + j]);
                float w_f32  = llaisys::utils::cast<float>(weight[j]);
                result = x_f32 * inv_rms;
                result = f16_round<do_round>(result); // match torch.mul f16 rounding
                result *= w_f32;
                result = f16_round<do_round>(result); // match torch.mul_ f16 rounding
                out[i * in_features + j] = llaisys::utils::cast<llaisys::fp16_t>(result);
            } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
                result = llaisys::utils::cast<float>(in[i * in_features + j])
                       * llaisys::utils::cast<float>(weight[j])
                       * inv_rms;
                out[i * in_features + j] = llaisys::utils::cast<llaisys::bf16_t>(result);
            } else {
                result = in[i * in_features + j] * weight[j] * inv_rms;
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
