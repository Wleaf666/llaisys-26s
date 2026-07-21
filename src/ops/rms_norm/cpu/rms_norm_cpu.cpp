#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t rows, size_t feature_size, float eps) {
    (void)out; (void)in; (void)weight; (void)type; (void)rows; (void)feature_size; (void)eps;
    TO_BE_IMPLEMENTED();
}
}