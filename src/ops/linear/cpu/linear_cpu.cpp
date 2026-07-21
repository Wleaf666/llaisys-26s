#include "linear_cpu.hpp"
#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t batch_size, size_t in_features, size_t out_features)
{
    for (size_t row = 0; row < batch_size;row++)
    {
        for (size_t col = 0; col < out_features;col++)
        {
            float res = bias != nullptr
                          ? llaisys::utils::cast<float>(bias[col])
                          : 0.0f;
            for (size_t k = 0; k < in_features;k++) {
                res += llaisys::utils::cast<float>(
                           in[row * in_features + k])
                     * llaisys::utils::cast<float>(
                           weight[col * in_features + k]);
            }
            out[row * out_features+col] = llaisys::utils::cast<T>(res);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), batch_size, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), batch_size, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}