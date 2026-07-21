#include "embedding_cpu.hpp"
#include "../../../utils.hpp"

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t index_count, size_t embedding_dim) {
    for (size_t i = 0; i < index_count;i++) {
        int64_t row = index[i];
        std::memcpy(out + i * embedding_dim, weight + row * embedding_dim, embedding_dim*sizeof(T));

    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const int64_t *index, const std::byte *weight, llaisysDataType_t type, size_t index_count, size_t embedding_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out),index,reinterpret_cast<const float *>(weight),index_count,embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t*>(out), index, reinterpret_cast<const llaisys::bf16_t*>(weight), index_count, embedding_dim);

    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), index, reinterpret_cast<const llaisys::fp16_t *>(weight), index_count, embedding_dim);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}