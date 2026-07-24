#include "self_attention_cpu.hpp"
#include "../../../tensor/tensor.hpp"
#include <vector>

void casual_softmax(std::vector<float> &score, size_t query_index, size_t query_len, size_t kv_len) {
    size_t past_len = kv_len - query_len;
    size_t max_key = past_len + query_index;
    float max_scores = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < max_key + 1; i++) {
        max_scores = std::max(max_scores, score[i]);
    }
    float exp_sum = 0.0f;
    for (size_t i = 0; i < max_key + 1; i++) {
        score[i] = exp(score[i] - max_scores);
        exp_sum += score[i];
    }
    for (size_t i = 0; i < max_key + 1; i++) {
        score[i] /= exp_sum;
    }
    for (size_t i = max_key+1; i < kv_len; i++) {
        score[i] = 0.0f;
    }
}

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, size_t query_len, size_t kv_len, size_t num_heads, size_t num_kv_heads, size_t head_dim, float scale) {
    std::vector<float> score(kv_len);
    size_t past_len = kv_len - query_len;
    size_t group_size = num_heads / num_kv_heads;
    for (size_t qi = 0; qi < query_len; qi++) {
        for (size_t headi = 0; headi < num_heads; headi++) {
            size_t kv_headi = headi / group_size;
            size_t q_offset = qi * head_dim * num_heads + headi * head_dim;
            for (size_t leni = 0; leni < kv_len; leni++) {
                float res = 0;
                size_t k_offset = leni * head_dim * num_kv_heads + kv_headi * head_dim;
                for (size_t di = 0; di < head_dim; di++) {
                    res += llaisys::utils::cast<float>(q[q_offset + di]) * llaisys::utils::cast<float>(k[k_offset +  di]);
                }
                score[leni] = res * scale;
            }
            casual_softmax(score, qi, query_len, kv_len);
            for (size_t di = 0; di < head_dim;di++)
            {
                float res = 0.0f;
                size_t out_offset = qi * head_dim * num_heads + headi * head_dim;
                for (size_t leni = 0; leni < kv_len; leni++) {
                    size_t v_offset = leni * num_kv_heads * head_dim + kv_headi * head_dim;
                    res += score[leni] * llaisys::utils::cast<float> (v[v_offset + di]);
                }
                attn_val[out_offset+di] = llaisys::utils::cast<T>(res);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, size_t query_len, size_t kv_len, size_t num_heads, size_t num_kv_heads, size_t head_dim, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), query_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), query_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), query_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu