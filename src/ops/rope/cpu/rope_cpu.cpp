#include "rope_cpu.hpp"
#include "../../../utils.hpp"
namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seq_len, size_t num_heads, size_t head_dim, float theta) {
    (void)out; (void)in; (void)pos_ids; (void)type; (void)seq_len; (void)num_heads; (void)head_dim; (void)theta;
    TO_BE_IMPLEMENTED();
}
}