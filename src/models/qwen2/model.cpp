#include "model.hpp"

#include "../utils.hpp"

#include <cmath>
#include <cstring>
#include <numeric>
#include <string> // std::stoi

namespace llaisys::models::qwen2 {

Model::Model(ModelMeta meta, ModelWeights weights)
    : _meta(std::move(meta)), _weights(std::move(weights)) {}

// forward declarations for helper functions used in infer
tensor_t get_tensor_by_name_(const ModelWeights &weights, const std::string &name);
int64_t infer_use_cache_(model_t model, int64_t *token_ids, size_t ntoken, int64_t top_k, float top_p, float temperature);
int select_device_id_(llaisysDeviceType_t device_type, const std::vector<int> &device_ids, int ndevice);
static void init_kv_caches_(model_t &model, llaisysDeviceType_t device_type, int device_id);

model_t Model::create_model(
    const ModelMeta &meta,
    llaisysDeviceType_t device_type,
    const std::vector<int> &device_ids,
    int ndevice) {
    CHECK_ARGUMENT(meta.hidden_size > 0, "hidden_size must be positive");
    CHECK_ARGUMENT(meta.nlayer > 0, "nlayer must be positive");
    CHECK_ARGUMENT(meta.vocab_size > 0, "vocab_size must be positive");
    CHECK_ARGUMENT(meta.nhead > 0, "nhead must be positive");
    CHECK_ARGUMENT(meta.nkvhead > 0, "nkvhead must be positive");
    CHECK_ARGUMENT(meta.d_intermediate > 0, "d_intermediate must be positive");

    const int device_id = select_device_id_(device_type, device_ids, ndevice);

    ModelWeights weights;

    // global weights
    weights.in_embed = Tensor::create({meta.vocab_size, meta.hidden_size}, meta.dtype, device_type, device_id);
    weights.out_embed = Tensor::create({meta.vocab_size, meta.hidden_size}, meta.dtype, device_type, device_id);
    weights.out_norm_w = Tensor::create({meta.hidden_size}, meta.dtype, device_type, device_id);

    // layer weights
    weights.layers.resize(meta.nlayer);
    for (size_t layer_idx = 0; layer_idx < meta.nlayer; ++layer_idx) {
        auto &layer = weights.layers[layer_idx];

        size_t d_kv = meta.nkvhead * (meta.hidden_size / meta.nhead);

        // attention weights
        layer.attn_norm_w = Tensor::create({meta.hidden_size}, meta.dtype, device_type, device_id);

        layer.attn_q_w = Tensor::create({meta.hidden_size, meta.hidden_size}, meta.dtype, device_type, device_id);
        layer.attn_q_b = Tensor::create({meta.hidden_size}, meta.dtype, device_type, device_id);

        layer.attn_k_w = Tensor::create({d_kv, meta.hidden_size}, meta.dtype, device_type, device_id);
        layer.attn_k_b = Tensor::create({d_kv}, meta.dtype, device_type, device_id);

        layer.attn_v_w = Tensor::create({d_kv, meta.hidden_size}, meta.dtype, device_type, device_id);
        layer.attn_v_b = Tensor::create({d_kv}, meta.dtype, device_type, device_id);

        layer.attn_o_w = Tensor::create({meta.hidden_size, meta.hidden_size}, meta.dtype, device_type, device_id);

        // mlp weights
        layer.mlp_norm_w = Tensor::create({meta.hidden_size}, meta.dtype, device_type, device_id);

        layer.mlp_gate_w = Tensor::create({meta.d_intermediate, meta.hidden_size}, meta.dtype, device_type, device_id);
        layer.mlp_up_w = Tensor::create({meta.d_intermediate, meta.hidden_size}, meta.dtype, device_type, device_id);
        layer.mlp_down_w = Tensor::create({meta.hidden_size, meta.d_intermediate}, meta.dtype, device_type, device_id);
    }

    auto m = model_t(new Model(meta, std::move(weights)));
    init_kv_caches_(m, device_type, device_id);
    return m;
}

static void init_kv_caches_(model_t &model, llaisysDeviceType_t device_type, int device_id) {
    const auto &meta = model->meta();
    auto &caches = model->kv_caches();
    size_t d_kv = meta.nkvhead * (meta.hidden_size / meta.nhead);
    for (size_t i = 0; i < meta.nlayer; ++i) {
        caches.emplace_back(meta.maxseq, d_kv, meta.dtype, device_type, device_id);
    }
}

model_t create(
    const ModelMeta &meta,
    llaisysDeviceType_t device_type,
    const std::vector<int> &device_ids,
    int ndevice) {
    return Model::create_model(meta, device_type, device_ids, ndevice);
}

model_weights_t getWeights(const model_t &model) {
    if (!model) {
        throw std::runtime_error("Qwen2 model is null");
    }
    return std::make_shared<llaisys::models::qwen2::ModelWeights>(model->weights());
}

void loadWeights(
    const model_t &model,
    const std::string &name,
    const void *data,
    const size_t numel) {
    if (!model) {
        throw std::runtime_error("Qwen2 model is null");
    }

    tensor_t tensor = get_tensor_by_name_(model->weights(), name);
    if (!tensor) {
        throw std::runtime_error("Tensor with name " + name + " not found in model weights");
    }

    if (tensor->numel() != numel) {
        throw std::runtime_error("Number of elements mismatch for tensor " + name);
    }

    tensor->load(data);

    // std::cout << tensor->info() << std::endl; // debug
}

int64_t infer(model_t model, int64_t *token_ids, size_t ntoken, int64_t top_k, float top_p, float temperature) {
    if (!model || !token_ids || ntoken == 0) {
        return -1;
    }
    if (model->meta().use_cache) {
        return infer_use_cache_(model, token_ids, ntoken, top_k, top_p, temperature);
    }

    const auto &model_weights = model->weights();
    const auto &meta = model->meta();

    tensor_t token_index = llaisys::model::utils::get_token_index(model_weights.in_embed, token_ids, ntoken);
    tensor_t input = llaisys::model::utils::apply_embedding(token_index, model_weights.in_embed);

    // layers
    size_t nlayer = meta.nlayer;
    tensor_t layer_input = input;
    for (size_t i_layer = 0; i_layer < nlayer; i_layer++) {
        // attention block
        const auto &layer_weights = model_weights.get_layer(i_layer);
        tensor_t attn_norm = llaisys::model::utils::apply_rms_norm(layer_input, layer_weights.attn_norm_w, meta.rms_epsilon);
        tensor_t attn_q = llaisys::model::utils::apply_linear(attn_norm, layer_weights.attn_q_w, layer_weights.attn_q_b);
        tensor_t attn_k = llaisys::model::utils::apply_linear(attn_norm, layer_weights.attn_k_w, layer_weights.attn_k_b);
        tensor_t attn_v = llaisys::model::utils::apply_linear(attn_norm, layer_weights.attn_v_w, layer_weights.attn_v_b);
        // multi-head
        size_t dhead = meta.hidden_size / meta.nhead;
        tensor_t head_q = attn_q->view({attn_q->shape()[0], meta.nhead, dhead});
        tensor_t head_k = attn_k->view({attn_k->shape()[0], meta.nkvhead, dhead});
        tensor_t head_v = attn_v->view({attn_v->shape()[0], meta.nkvhead, dhead});
        // rope
        tensor_t pos_ids = llaisys::model::utils::make_pos_ids(0, ntoken, head_q->deviceType(), head_q->deviceId());
        tensor_t rope_q = llaisys::model::utils::apply_rope(head_q, pos_ids, meta.rope_theta);
        tensor_t rope_k = llaisys::model::utils::apply_rope(head_k, pos_ids, meta.rope_theta);
        // self-attention
        float scale = 1.0f / std::sqrt(static_cast<float>(dhead)); // avoid C4244 (double -> float) under MSVC /WX
        tensor_t attn_val = llaisys::model::utils::compute_self_attention(rope_q, rope_k, head_v, scale);
        // attention output projection
        tensor_t attn_proj = llaisys::model::utils::apply_linear(attn_val->view(layer_input->shape()), layer_weights.attn_o_w, nullptr);
        // add residual
        tensor_t attn_out = llaisys::model::utils::apply_add(layer_input, attn_proj);

        // multi-layer perceptron block
        tensor_t mlp_norm = llaisys::model::utils::apply_rms_norm(attn_out, layer_weights.mlp_norm_w, meta.rms_epsilon);
        // swiglu
        tensor_t mlp_gate = llaisys::model::utils::apply_linear(mlp_norm, layer_weights.mlp_gate_w, nullptr);
        tensor_t mlp_up = llaisys::model::utils::apply_linear(mlp_norm, layer_weights.mlp_up_w, nullptr);
        tensor_t mlp_swiglu = llaisys::model::utils::apply_swiglu(mlp_gate, mlp_up);
        // block out
        tensor_t mlp_out = llaisys::model::utils::apply_linear(mlp_swiglu, layer_weights.mlp_down_w, nullptr);

        // add residual
        tensor_t layer_out = llaisys::model::utils::apply_add(attn_out, mlp_out);

        // next layer
        layer_input = layer_out;
        // std::cout << "layer " << i_layer << "done" << std::endl; // debug
    }

    // output embedding
    tensor_t out_rms_norm = llaisys::model::utils::apply_rms_norm(layer_input, model_weights.out_norm_w, meta.rms_epsilon);
    // LM head
    tensor_t logits = llaisys::model::utils::apply_linear(out_rms_norm, model_weights.out_embed, nullptr);

    // find next token
    tensor_t last_logits = logits->slice(0, ntoken - 1, ntoken); // last line

    // return scale
    int64_t next_token_id = llaisys::model::utils::get_next_token_id(last_logits, model->sampler(), top_k, top_p, temperature);

    return next_token_id;
}

int64_t infer_use_cache_(model_t model, int64_t *token_ids, size_t ntoken, int64_t top_k, float top_p, float temperature) {
    if (!model || !token_ids || ntoken == 0) {
        return -1;
    }

    const auto &model_weights = model->weights();
    const auto &meta = model->meta();
    auto &kv_caches = model->kv_caches();

    // Determine past_len from cache (0 for prefill, >0 for decode)
    size_t past_len = kv_caches.empty() ? 0 : kv_caches[0].curLen();

    // Prefill: process all tokens; Decode: only process the last token
    int64_t *current_ids = token_ids;
    size_t current_ntoken = ntoken;
    if (past_len > 0) {
        current_ids = &token_ids[ntoken - 1];
        current_ntoken = 1;
    }

    // Embedding
    tensor_t token_index = llaisys::model::utils::get_token_index(model_weights.in_embed, current_ids, current_ntoken);
    tensor_t input = llaisys::model::utils::apply_embedding(token_index, model_weights.in_embed);

    // Layers
    size_t nlayer = meta.nlayer;
    tensor_t layer_input = input;
    for (size_t i_layer = 0; i_layer < nlayer; i_layer++) {
        const auto &layer_weights = model_weights.get_layer(i_layer);
        auto &kv_cache = kv_caches[i_layer];

        // Attention block
        tensor_t attn_norm = llaisys::model::utils::apply_rms_norm(layer_input, layer_weights.attn_norm_w, meta.rms_epsilon);
        tensor_t attn_q = llaisys::model::utils::apply_linear(attn_norm, layer_weights.attn_q_w, layer_weights.attn_q_b);
        tensor_t attn_k = llaisys::model::utils::apply_linear(attn_norm, layer_weights.attn_k_w, layer_weights.attn_k_b);
        tensor_t attn_v = llaisys::model::utils::apply_linear(attn_norm, layer_weights.attn_v_w, layer_weights.attn_v_b);

        // Multi-head
        size_t dhead = meta.hidden_size / meta.nhead;
        tensor_t head_q = attn_q->view({attn_q->shape()[0], meta.nhead, dhead});
        tensor_t head_k = attn_k->view({attn_k->shape()[0], meta.nkvhead, dhead});
        tensor_t head_v = attn_v->view({attn_v->shape()[0], meta.nkvhead, dhead});

        // RoPE (positions start from past_len for correct positional encoding)
        tensor_t pos_ids = llaisys::model::utils::make_pos_ids(past_len, current_ntoken, head_q->deviceType(), head_q->deviceId());
        tensor_t rope_q = llaisys::model::utils::apply_rope(head_q, pos_ids, meta.rope_theta);
        tensor_t rope_k = llaisys::model::utils::apply_rope(head_k, pos_ids, meta.rope_theta);

        // Append K/V to cache (flatten to 2D: {seqlen, nkvhead * dhead})
        size_t d_kv_flat = meta.nkvhead * dhead;
        tensor_t rope_k_2d = rope_k->view({rope_k->shape()[0], d_kv_flat});
        tensor_t head_v_2d = head_v->view({head_v->shape()[0], d_kv_flat});
        kv_cache.append(rope_k_2d, head_v_2d);

        // Self-attention with cached K/V
        float scale = 1.0f / std::sqrt(static_cast<float>(dhead)); // avoid C4244 (double -> float) under MSVC /WX
        size_t total_len = kv_cache.curLen();
        tensor_t cached_k_2d = kv_cache.getK(total_len);
        tensor_t cached_v_2d = kv_cache.getV(total_len);
        tensor_t cached_k = cached_k_2d->view({total_len, meta.nkvhead, dhead});
        tensor_t cached_v = cached_v_2d->view({total_len, meta.nkvhead, dhead});
        tensor_t attn_val = llaisys::model::utils::compute_self_attention(rope_q, cached_k, cached_v, scale);

        // Attention output projection
        tensor_t attn_proj = llaisys::model::utils::apply_linear(attn_val->view(layer_input->shape()), layer_weights.attn_o_w, nullptr);
        tensor_t attn_out = llaisys::model::utils::apply_add(layer_input, attn_proj);

        // MLP block
        tensor_t mlp_norm = llaisys::model::utils::apply_rms_norm(attn_out, layer_weights.mlp_norm_w, meta.rms_epsilon);
        tensor_t mlp_gate = llaisys::model::utils::apply_linear(mlp_norm, layer_weights.mlp_gate_w, nullptr);
        tensor_t mlp_up = llaisys::model::utils::apply_linear(mlp_norm, layer_weights.mlp_up_w, nullptr);
        tensor_t mlp_swiglu = llaisys::model::utils::apply_swiglu(mlp_gate, mlp_up);
        tensor_t mlp_out = llaisys::model::utils::apply_linear(mlp_swiglu, layer_weights.mlp_down_w, nullptr);
        tensor_t layer_out = llaisys::model::utils::apply_add(attn_out, mlp_out);

        layer_input = layer_out;
    }

    // Output embedding

    tensor_t out_rms_norm = llaisys::model::utils::apply_rms_norm(layer_input, model_weights.out_norm_w, meta.rms_epsilon);

    tensor_t logits = llaisys::model::utils::apply_linear(out_rms_norm, model_weights.out_embed, nullptr);

    // Find next token (last position in logits)
    size_t logit_rows = logits->shape()[0];
    tensor_t last_logits = logits->slice(0, logit_rows - 1, logit_rows);

    // return scale
    int64_t next_token_id = llaisys::model::utils::get_next_token_id(last_logits, model->sampler(), top_k, top_p, temperature);

    return next_token_id;
}

tensor_t get_tensor_by_name_(const ModelWeights &weights, const std::string &name) {
    // Global weights
    if (name == "model.embed_tokens.weight") {
        return weights.in_embed;
    }
    if (name == "model.norm.weight") {
        return weights.out_norm_w;
    }
    if (name == "lm_head.weight") {
        return weights.out_embed;
    }

    // Layer weights: "model.layers.{idx}.{field}"
    const std::string prefix = "model.layers.";
    if (name.find(prefix) == 0) {
        size_t idx_start = prefix.length();
        size_t idx_end = name.find('.', idx_start);
        if (idx_end == std::string::npos) {
            return nullptr;
        }

        int layer_idx = std::stoi(name.substr(idx_start, idx_end - idx_start));
        if (layer_idx < 0 || layer_idx >= (int)weights.layers.size()) {
            return nullptr;
        }

        const auto &layer = weights.layers[layer_idx];
        std::string field = name.substr(idx_end + 1);

        if (field == "input_layernorm.weight") {
            return layer.attn_norm_w;
        }

        if (field == "self_attn.q_proj.weight") {
            return layer.attn_q_w;
        }
        if (field == "self_attn.q_proj.bias") {
            return layer.attn_q_b;
        }

        if (field == "self_attn.k_proj.weight") {
            return layer.attn_k_w;
        }
        if (field == "self_attn.k_proj.bias") {
            return layer.attn_k_b;
        }

        if (field == "self_attn.v_proj.weight") {
            return layer.attn_v_w;
        }
        if (field == "self_attn.v_proj.bias") {
            return layer.attn_v_b;
        }

        if (field == "post_attention_layernorm.weight") {
            return layer.mlp_norm_w;
        }

        if (field == "self_attn.o_proj.weight") {
            return layer.attn_o_w;
        }

        if (field == "mlp.gate_proj.weight") {
            return layer.mlp_gate_w;
        }
        if (field == "mlp.up_proj.weight") {
            return layer.mlp_up_w;
        }
        if (field == "mlp.down_proj.weight") {
            return layer.mlp_down_w;
        }
    }

    return nullptr;
}

int select_device_id_(llaisysDeviceType_t device_type, const std::vector<int> &device_ids, int ndevice) {
    if (device_type == LLAISYS_DEVICE_CPU) {
        return 0;
    }
    if (!device_ids.empty()) {
        return device_ids[0];
    }
    if (ndevice > 0) {
        return 0;
    }
    return 0;
}

} // namespace llaisys::models::qwen2