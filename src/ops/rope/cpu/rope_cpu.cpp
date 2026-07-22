#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>


template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t num_heads, size_t head_dim, float theta)
{
    for (size_t index = 0; index < seq_len;index++)
    {
        for (size_t cur_head_index = 0; cur_head_index < num_heads;cur_head_index++) {
            for (size_t i = 0, j = head_dim / 2; i < (head_dim / 2);i++,j++){
                float fyi =pos_ids[index] / std::powf(theta, 2.0f * i / head_dim);
                size_t base =  head_dim * num_heads*index+head_dim*cur_head_index;
                const float a=llaisys::utils::cast<float>(in[base + i]);
                const float b=llaisys::utils::cast<float>(in[base + j]);
                const float sin_val=sin(fyi);
                const float cos_val=cos(fyi);
                out[base + i] = llaisys::utils::cast<T>( a * cos_val - b * sin_val);
                out[base + j] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte * out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seq_len, size_t num_heads, size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t  *>(in), reinterpret_cast<const int64_t *>(pos_ids), seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), seq_len, num_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    }
}