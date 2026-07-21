#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t feature_size,float eps) {
    for (size_t row = 0; row < rows;row++)
    {
        float sum_square = 0.0f;
        for (size_t col = 0; col < feature_size; col++) {
            float value = llaisys::utils::cast<float>(in[row * feature_size + col]);
            sum_square += value * value;
        }
        float inv_denominator = 1.0f/std::sqrt(sum_square / feature_size + eps);
        for (size_t col = 0; col < feature_size; col++) {
            float res = llaisys::utils::cast<float>(weight[col]) * llaisys::utils::cast<float> (in[row * feature_size + col])*inv_denominator;
            out[row * feature_size + col] = llaisys::utils::cast<T>(res);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t rows, size_t feature_size, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), rows, feature_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), rows, feature_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), rows, feature_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}