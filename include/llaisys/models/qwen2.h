#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    typedef struct LlaisysQwen2Meta LlaisysQwen2Meta;
    typedef struct LlaisysQwen2Weights LlaisysQwen2Weights;
    typedef struct LlaisysQwen2Model LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    // load single tensor weights
    __export int llaisysQwen2ModelLoadWeights(struct LlaisysQwen2Model * model, const char *name, const void *data, size_t numel);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken, int64_t top_k, float top_p, float temperature);
}
#endif // LLAISYS_MODELS_QWEN2_H
