#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>

#include <vector>

// causal mask via triangular loop bound: k_idx < total_len - seqlen + i + 1
template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale, size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
    const size_t group_size = nhead / nkvhead;

#pragma omp parallel for
    for (int64_t kv_head = 0; kv_head < static_cast<int64_t>(nkvhead); kv_head++) {
        for (size_t i = 0; i < seqlen; i++) {
            size_t valid_end = total_len - seqlen + i + 1;
            std::vector<float> acc(group_size * valid_end, 0.0f);
            std::vector<float> maxs(group_size, -INFINITY);
            for (size_t k_idx = 0; k_idx < valid_end; k_idx++) { // triangular
                // Pass 1: compute group scores
                for (size_t n = 0; n < d; n++) {
                    float k_val = 0.0f;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        k_val += llaisys::utils::cast<float>(k[(k_idx * nkvhead + kv_head) * d + n]);
                    } else {
                        k_val += k[(k_idx * nkvhead + kv_head) * d + n];
                    }
                    // k_val for group
                    for (size_t group_head = 0; group_head < group_size; group_head++) {
                        size_t q_head = kv_head * group_size + group_head;
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            acc[group_head * valid_end + k_idx] += llaisys::utils::cast<float>(q[(i * nhead + q_head) * d + n]) * k_val;
                        } else {
                            acc[group_head * valid_end + k_idx] += q[(i * nhead + q_head) * d + n] * k_val;
                        }
                    }
                }
                // update max for each head on the process of go through k_idx
                for (size_t group_head = 0; group_head < group_size; group_head++) {
                    acc[group_head * valid_end + k_idx] *= scale;
                    maxs[group_head] = std::max(maxs[group_head], acc[group_head * valid_end + k_idx]);
                }
            }

            // Pass 2: compute stable softmax (subtract max before exp)
            // local update
            for (size_t group_head = 0; group_head < group_size; group_head++) {
                float sum_exp_scores = 0.0f;
                for (size_t k_idx = 0; k_idx < valid_end; k_idx++) {
                    acc[group_head * valid_end + k_idx] = std::exp(acc[group_head * valid_end + k_idx] - maxs[group_head]);
                    sum_exp_scores += acc[group_head * valid_end + k_idx];
                }
                for (size_t k_idx = 0; k_idx < valid_end; k_idx++) {
                    acc[group_head * valid_end + k_idx] /= sum_exp_scores;
                }
            }

            // Pass 3: compute weighted sum
            for (size_t nv = 0; nv < dv; nv++) {
                std::vector<float> attn_vals(group_size, 0.0f);
                for (size_t v_idx = 0; v_idx < valid_end; v_idx++) {
                    float v_val = 0.0f;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[(v_idx * nkvhead + kv_head) * dv + nv]);
                    } else {
                        v_val = v[(v_idx * nkvhead + kv_head) * dv + nv];
                    }
                    for (size_t group_head = 0; group_head < group_size; group_head++) {
                        float softmax_score = acc[group_head * valid_end + v_idx];
                        attn_vals[group_head] += softmax_score * v_val;
                    }
                }
                for (size_t group_head = 0; group_head < group_size; group_head++) {
                    size_t a_head = kv_head * group_size + group_head;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        attn_val[(i * nhead + a_head) * dv + nv] = llaisys::utils::cast<T>(attn_vals[group_head]);
                    } else {
                        attn_val[(i * nhead + a_head) * dv + nv] = attn_vals[group_head];
                    }
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