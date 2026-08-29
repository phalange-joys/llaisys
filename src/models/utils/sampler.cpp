#include "sampler.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../tensor/tensor.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace llaisys::model::utils {

// Forward declarations
template <typename T>
void copy_and_convert_(T *data, llaisysDataType_t dtype, size_t numel, bool is_host,
                       std::vector<float> &float_buffer, std::vector<uint16_t> &raw_buffer);
void copy_and_convert(const tensor_t &src, std::vector<float> &float_buffer, std::vector<uint16_t> &raw_buffer);
void apply_temperature(float *data, size_t n, float temperature);
void apply_top_k(float *data, size_t n, int64_t k);
void apply_softmax(float *data, size_t n);
void apply_top_p(float *data, size_t n, float p);

int64_t Sampler::sample(const tensor_t &logits,
                        int64_t top_k,
                        float top_p,
                        float temperature) {
    size_t n = logits->numel();
    if (float_buffer_.size() != n) {
        float_buffer_.resize(n);
    }
    copy_and_convert(logits, float_buffer_, raw_buffer_);

    apply_temperature(float_buffer_.data(), n, temperature);
    apply_top_k(float_buffer_.data(), n, top_k);
    apply_softmax(float_buffer_.data(), n);

    apply_top_p(float_buffer_.data(), n, top_p);
    return random_sample(float_buffer_.data(), n);
}

int64_t Sampler::random_sample(float *data, size_t n) {
    std::vector<float> cumsum(n);
    std::partial_sum(data, data + n, cumsum.begin());

    std::uniform_real_distribution<float> dist(0.0f, cumsum.back());
    float r = dist(rng_);

    auto it = std::lower_bound(cumsum.begin(), cumsum.end(), r);
    return static_cast<int64_t>(std::distance(cumsum.begin(), it));
}

// nucleus sampling
void apply_top_p(float *data, size_t n, float p) {
    if (p >= 1.0f) {
        return;
    }
    // sort indices by probability descending
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), static_cast<size_t>(0));
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return data[a] > data[b];
    });
    // find cutoff where cumsum exceeds p
    float cumsum = 0.0f;
    size_t cutoff = n;
    for (size_t i = 0; i < n; i++) {
        cumsum += data[indices[i]]; // descending
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }
    // zero out beyond cutoff and renormalize
    for (size_t i = cutoff; i < n; i++) {
        data[indices[i]] = 0.0f;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    if (sum > 0.0f) {
        for (size_t i = 0; i < n; i++) {
            data[i] /= sum;
        }
    }
}

void apply_softmax(float *data, size_t n) {
    float max_val = *std::max_element(data, data + n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    for (size_t i = 0; i < n; i++) {
        data[i] /= sum;
    }
}

void apply_top_k(float *data, size_t n, int64_t k) {
    if (k <= 0 || static_cast<size_t>(k) >= n) {
        return;
    }
    size_t uk = static_cast<size_t>(k);
    // find kth largest value
    std::vector<float> temp(data, data + n);
    std::nth_element(temp.begin(), temp.begin() + static_cast<ptrdiff_t>(uk) - 1, temp.end(), std::greater<float>());
    float threshold = temp[uk - 1];

    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; i++) {
        if (data[i] < threshold) {
            data[i] = neg_inf;
        }
    }
}

void apply_temperature(float *data, size_t n, float temperature) {
    if (temperature <= 0.0f) {
        return;
    }
    for (size_t i = 0; i < n; i++) {
        data[i] /= temperature;
    }
}

template <typename T>
void copy_and_convert_(T *data, llaisysDataType_t dtype, size_t numel, bool is_host,
                       std::vector<float> &float_buffer, std::vector<uint16_t> &raw_buffer) {
    // copy raw data to cpu
    size_t total_bytes = numel * llaisys::utils::dsize(dtype);

    if (dtype == LLAISYS_DTYPE_F32) {
        if (is_host) {
            std::memcpy(float_buffer.data(), data, numel * sizeof(float));
        } else {
            core::context()
                .runtime()
                .api()
                ->memcpy_sync(float_buffer.data(), data, total_bytes, LLAISYS_MEMCPY_D2H);
        }
        return;
    }

    // For device tensors, copy raw bytes to host first via raw_buffer
    T *src_typed = data;
    if (!is_host) {
        if (raw_buffer.size() != total_bytes) {
            raw_buffer.resize(total_bytes);
        }
        core::context().runtime().api()->memcpy_sync(
            raw_buffer.data(),
            data,
            total_bytes,
            LLAISYS_MEMCPY_D2H);
        src_typed = reinterpret_cast<T *>(raw_buffer.data());
    }

    for (size_t i = 0; i < numel; i++) {
        float_buffer[i] = llaisys::utils::cast<float>(src_typed[i]);
    }
}

void copy_and_convert(const tensor_t &src, std::vector<float> &float_buffer, std::vector<uint16_t> &raw_buffer) {
    llaisysDataType_t dtype = src->dtype();
    std::byte *src_data = src->data();
    size_t numel = src->numel();
    bool is_host = (src->deviceType() == LLAISYS_DEVICE_CPU);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return copy_and_convert_(reinterpret_cast<float *>(src_data), dtype, numel, is_host, float_buffer, raw_buffer);
    case LLAISYS_DTYPE_BF16:
        return copy_and_convert_(reinterpret_cast<llaisys::bf16_t *>(src_data), dtype, numel, is_host, float_buffer, raw_buffer);
    case LLAISYS_DTYPE_F16:
        return copy_and_convert_(reinterpret_cast<llaisys::fp16_t *>(src_data), dtype, numel, is_host, float_buffer, raw_buffer);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::model::utils
