#include "llaisys/models/qwen2.h"

#include "../llaisys_models.hpp"
#include "../llaisys_tensor.hpp"

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        auto model = new LlaisysQwen2Model();

        if (!meta) {
            return model;
        }

        std::vector<int> ids;
        if (device_ids && ndevice > 0) {
            ids.assign(device_ids, device_ids + ndevice);
        }
        try {
            model->model = llaisys::models::qwen2::create(meta->meta, device, ids, ndevice);
        } catch (const std::exception &e) {
            printf("Error creating Qwen2 model: %s\n", e.what());
            delete model;
            return nullptr;
        }
        return model;
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model * model) {
        if (!model) {
            return;
        }
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model * model) {
        if (!model || !model->model) {
            return nullptr;
        }
        auto *out = new LlaisysQwen2Weights();
        out->weights = llaisys::models::qwen2::getWeights(model->model);
        return out;
    }

    int llaisysQwen2ModelLoadWeights(LlaisysQwen2Model * model, const char *name, const void *data, size_t numel) {
        if (!model || !model->model || !name || !data) {
            return -1;
        }
        try {
            llaisys::models::qwen2::loadWeights(model->model, name, data, numel);
        } catch (const std::exception &e) {
            printf("Error loading weights: %s\n", e.what());
            return -1; // Return an error code if an exception occurs
        }
        return 0;
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken, int64_t top_k, float top_p, float temperature) {
        if (!model) {
            return 0;
        }
        return llaisys::models::qwen2::infer(model->model, token_ids, ntoken, top_k, top_p, temperature);
    }
}
