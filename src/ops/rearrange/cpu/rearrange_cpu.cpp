#include "rearrange_cpu.hpp"
#include "../../../utils.hpp"
namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, const size_t *shape, const ptrdiff_t *strides, size_t ndim, size_t numel) {
    (void)out; (void)in; (void)type; (void)shape; (void)strides; (void)ndim; (void)numel;
    TO_BE_IMPLEMENTED();
}
}