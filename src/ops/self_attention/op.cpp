#include "op.hpp"

#include "../../core/core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    // Only support contiguous inputs for now.
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "Self-Attention: all tensors must be contiguous.");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    // atten_val [seqlen, nhead, dv]
    // q [seqlen, nhead, d], k  [total_len, nkvhead, d], v [total_len, nkvhead, dv]
    // total_len = seqlen + past_len, nkvhead = nhead * nkv_ratio
    // (now qwen2) GQA(Group Query Attention) more than one q-heads share the same k-head
    // future tokens: no need to compute, since softmax value = exp(-inf) ~ 0
    CHECK_ARGUMENT(k->shape() == v->shape(), "Self -Attention: key and value tensors must have the same shape.");
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0], "Self-Attention: attention value and query tensors must have the same sequence length.");
    CHECK_ARGUMENT(attn_val->shape()[1] == q->shape()[1], "Self-Attention: attention value and query tensors must have the same number of heads.");
    CHECK_ARGUMENT(attn_val->shape()[2] == v->shape()[2], "Self-Attention: attention value and value tensors must have the same dimension.");
    // less kvhead less attention compute, less k/v cache
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0, "Self-Attention: number of heads in query must be divisible by number of heads in key/value.");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                   q->shape()[0], q->shape()[1], q->shape()[2], k->shape()[0], k->shape()[1], k->shape()[2]);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                   q->shape()[0], q->shape()[1], q->shape()[2], k->shape()[0], k->shape()[1], k->shape()[2]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
