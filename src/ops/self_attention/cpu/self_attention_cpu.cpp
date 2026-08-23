#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

#include <vector>

// causal mask via triangular loop bound: k_idx < total_len - seqlen + i + 1
template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale, size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
    #pragma omp parallel for
    for (int64_t j = 0; j < static_cast<int64_t>(nhead); j++) {
        size_t head_idx = j * nkvhead / nhead;
        for (size_t i = 0; i < seqlen; i++) {
            // Pass 1: compute all dot-product scores and find max for numerical stability
            size_t valid_end = total_len - seqlen + i + 1;
            std::vector<float> scores(valid_end);
            float max_score = -INFINITY;
            for (size_t k_idx = 0; k_idx < valid_end; k_idx++) {
                float dot_product = 0.0f;
                for (size_t n = 0; n < d; n++) {
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        dot_product += llaisys::utils::cast<float>(q[i * nhead * d + j * d + n]) * llaisys::utils::cast<float>(k[k_idx * nkvhead * d + head_idx * d + n]);
                    } else {
                        dot_product += q[i * nhead * d + j * d + n] * k[k_idx * nkvhead * d + head_idx * d + n];
                    }
                }
                scores[k_idx] = dot_product * scale;
                if (scores[k_idx] > max_score) {
                    max_score = scores[k_idx];
                }
            }

            // Pass 2: compute stable softmax (subtract max before exp)
            float sum_exp_scores = 0.0f;
            std::vector<float> softmax_scores(valid_end);
            for (size_t k_idx = 0; k_idx < valid_end; k_idx++) {
                softmax_scores[k_idx] = std::exp(scores[k_idx] - max_score);
                sum_exp_scores += softmax_scores[k_idx];
            }

            // Pass 3: compute weighted sum
            for (size_t nv = 0; nv < dv; nv++) {
                float attn_value = 0.0f;
                for (size_t v_idx = 0; v_idx < valid_end; v_idx++) {
                    float softmax_score = softmax_scores[v_idx] / sum_exp_scores;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        attn_value += softmax_score * llaisys::utils::cast<float>(v[v_idx * nkvhead * dv + head_idx * dv + nv]);
                    } else {
                        attn_value += softmax_score * v[v_idx * nkvhead * dv + head_idx * dv + nv];
                    }
                }
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[i * nhead * dv + j * dv + nv] = llaisys::utils::cast<T>(attn_value);
                } else {
                    attn_val[i * nhead * dv + j * dv + nv] = attn_value;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, llaisysDataType_t type,
                    size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), scale, seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), scale, seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), scale, seqlen, nhead, d, total_len, nkvhead, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu