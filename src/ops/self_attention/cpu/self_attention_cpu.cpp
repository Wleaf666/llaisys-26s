#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, size_t query_len, size_t kv_len, size_t num_heads, size_t num_kv_heads, size_t head_dim, float scale) {
    (void)attn_val; (void)q; (void)k; (void)v; (void)type; (void)query_len; (void)kv_len; (void)num_heads; (void)num_kv_heads; (void)head_dim; (void)scale;
    TO_BE_IMPLEMENTED();
}
}