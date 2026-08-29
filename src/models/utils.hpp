#include "../tensor/tensor.hpp"

#include "../ops/ops.hpp"

#include "utils/sampler.hpp"

namespace llaisys::model::utils {

static tensor_t make_pos_ids(size_t start, size_t len, llaisysDeviceType_t dev, int id) {
    // Build on CPU first, then transfer to target device
    auto t_cpu = Tensor::create({len}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    auto *d = reinterpret_cast<int64_t *>(t_cpu->data());
    for (size_t i = 0; i < len; i++) {
        d[i] = static_cast<int64_t>(start + i);
    }
    if (dev == LLAISYS_DEVICE_CPU) {
        return t_cpu;
    }

    return t_cpu->to(dev, id);
}
tensor_t get_token_index(tensor_t in_embed, int64_t *token_ids, size_t ntoken) {
    tensor_t out = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, in_embed->deviceType(), in_embed->deviceId());
    out->load(token_ids);
    return out;
}

tensor_t apply_add(tensor_t a, tensor_t b) {
    tensor_t out = Tensor::create(a->shape(), a->dtype(), a->deviceType(), a->deviceId());
    llaisys::ops::add(out, a, b);
    return out;
}

tensor_t apply_embedding(tensor_t token_index, tensor_t in_embed) {
    tensor_t out = Tensor::create({token_index->shape()[0], in_embed->shape()[1]}, in_embed->dtype(), in_embed->deviceType(), in_embed->deviceId());
    llaisys::ops::embedding(out, token_index, in_embed);
    return out;
}

tensor_t apply_linear(tensor_t input, tensor_t weight, tensor_t bias) {
    tensor_t out = Tensor::create({input->shape()[0], weight->shape()[0]}, input->dtype(), input->deviceType(), input->deviceId());
    llaisys::ops::linear(out, input, weight, bias);
    return out;
}

tensor_t apply_rms_norm(tensor_t input, tensor_t norm_w, float rms_epsilon) {
    tensor_t out = Tensor::create(input->shape(), input->dtype(), input->deviceType(), input->deviceId());
    llaisys::ops::rms_norm(out, input, norm_w, rms_epsilon);
    return out;
}

tensor_t apply_rope(tensor_t input, tensor_t pos_ids, float rope_theta) {
    tensor_t out = Tensor::create(input->shape(), input->dtype(), input->deviceType(), input->deviceId());
    llaisys::ops::rope(out, input, pos_ids, rope_theta);
    return out;
}

tensor_t compute_self_attention(tensor_t q, tensor_t k, tensor_t v, float scale) {
    tensor_t out = Tensor::create({q->shape()[0], q->shape()[1], v->shape()[2]}, q->dtype(), q->deviceType(), q->deviceId());
    llaisys::ops::self_attention(out, q, k, v, scale);
    return out;
}

tensor_t apply_swiglu(tensor_t gate, tensor_t up) {
    tensor_t out = Tensor::create(gate->shape(), gate->dtype(), gate->deviceType(), gate->deviceId());
    llaisys::ops::swiglu(out, gate, up);
    return out;
}

int64_t get_next_token_id(tensor_t logits, Sampler &sampler, int64_t top_k, float top_p, float temperature) {
    if (top_k == 1) {
        tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, logits->deviceType(), logits->deviceId());
        tensor_t max_val = Tensor::create({1}, logits->dtype(), logits->deviceType(), logits->deviceId());
        llaisys::ops::argmax(max_idx, max_val, logits);
        return max_idx->toScalar<int64_t>();
    }

    int64_t next_token_id = sampler.sample(
        logits,
        top_k,
        top_p,
        temperature);
    return next_token_id;
}

} // namespace llaisys::model::utils
