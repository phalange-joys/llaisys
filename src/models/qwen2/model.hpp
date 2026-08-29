#pragma once

#include "../../core/llaisys_core.hpp"
#include "../../tensor/tensor.hpp"
#include "../kv_cache.hpp"
#include "../utils/sampler.hpp"

#include <memory>
#include <vector>

namespace llaisys::models::qwen2 {
class Model;
using model_t = std::shared_ptr<Model>;
using model_weights_t = std::shared_ptr<struct ModelWeights>;

struct LayerWeights {
    tensor_t attn_norm_w; // rms norm weight
    tensor_t attn_q_w;
    tensor_t attn_q_b;
    tensor_t attn_k_w;
    tensor_t attn_k_b;
    tensor_t attn_v_w;
    tensor_t attn_v_b;
    tensor_t attn_o_w;
    tensor_t mlp_norm_w; // rms norm weight post attention
    tensor_t mlp_gate_w;
    tensor_t mlp_up_w;
    tensor_t mlp_down_w;
};

struct ModelWeights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<LayerWeights> layers;

    size_t num_layers() const { return layers.size(); }
    const LayerWeights &get_layer(size_t idx) const {
        if (idx >= layers.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        return layers[idx];
    }
};

struct ModelMeta {
    // config.json
    llaisysDataType_t dtype = LLAISYS_DTYPE_BF16;
    size_t nlayer = 0;
    size_t hidden_size = 0;
    size_t nhead = 0;
    size_t nkvhead = 0;
    size_t d_intermediate = 0;
    size_t maxseq = 0;
    size_t vocab_size = 0;
    float rms_epsilon = 1e-6f;
    float rope_theta = 10000.0f;
    int64_t end_token = -1;
    // additional config parameters
    bool use_sliding_window = false;
    bool use_cache = true;
};

class Model {
private:
    ModelMeta _meta;
    ModelWeights _weights;
    std::vector<KVCache> _kv_caches;
    llaisys::model::utils::Sampler _sampler;
    Model(ModelMeta meta, ModelWeights weights);

public:
    static model_t create_model(
        const ModelMeta &meta,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        const std::vector<int> &device_ids = {},
        int ndevice = 0);
    ~Model() = default;

    const ModelMeta &meta() const { return _meta; }
    const ModelWeights &weights() const { return _weights; }
    std::vector<KVCache> &kv_caches() { return _kv_caches; }
    llaisys::model::utils::Sampler &sampler() { return _sampler; }
};

model_t create(
    const ModelMeta &meta,
    llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
    const std::vector<int> &device_ids = {},
    int ndevice = 0);

model_weights_t getWeights(const model_t &model);

void loadWeights(
    const model_t &model,
    const std::string &name,
    const void *data,
    const size_t numel);

int64_t infer(model_t model, int64_t *token_ids, size_t ntoken, int64_t top_k, float top_p, float temperature);

} // namespace llaisys::models::qwen2
