#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    (void)out; (void)gate; (void)up; (void)type; (void)numel;
    TO_BE_IMPLEMENTED();
}
}