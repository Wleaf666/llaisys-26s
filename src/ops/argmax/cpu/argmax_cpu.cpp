#include "argmax_cpu.hpp"

#include "../../../utils.hpp"



namespace llaisys::ops::cpu {
template <typename T>
void argmax_(tensor_t max_idx, tensor_t max_val, T *vals, size_t numel) {
    size_t _index = 0;
    float max = llaisys::utils::cast<float>(vals[0]);
    for (size_t i = 0; i < numel; i++) {
        if (max < llaisys::utils::cast<float>(vals[i])) {
                max = llaisys::utils::cast<float>(vals[i]);
                _index = i;
            }
        } 
    T max_val_ = vals[_index];
    max_val->load(&max_val_);
    max_idx->load(&_index);
}
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals,llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(max_idx, max_val, reinterpret_cast<float *>(vals->data()),numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(max_idx, max_val, reinterpret_cast<llaisys::bf16_t *>(vals->data()), numel);

        case LLAISYS_DTYPE_F16:
            return argmax_(max_idx, max_val, reinterpret_cast<llaisys::fp16_t *>(vals->data()), numel);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}