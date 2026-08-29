// src/models/utils/sampler.hpp
#pragma once

#include <vector>

// Temporarily undef __C to avoid conflict with GCC's emmintrin.h
// (included by <random>, which uses __C as internal identifier)
#ifdef __C
#pragma push_macro("__C")
#undef __C
#define _SAMPLER_RESTORE___C
#endif

#include <random>

#ifdef _SAMPLER_RESTORE___C
#pragma pop_macro("__C")
#undef _SAMPLER_RESTORE___C
#endif

#include "../../tensor/tensor.hpp"

namespace llaisys::model::utils {

class Sampler {

public:
    Sampler() : rng_(std::random_device{}()) {}

    int64_t sample(const tensor_t &logits,
                   int64_t top_k,
                   float top_p,
                   float temperature);

private:
    int64_t random_sample(float *data, size_t n);

    std::vector<float> float_buffer_;  // float or fp32 for cpu compute
    std::vector<uint16_t> raw_buffer_; // raw data from device
    std::mt19937 rng_;
};

} // namespace llaisys::model::utils