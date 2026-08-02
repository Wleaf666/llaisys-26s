#pragma once

#include "../core/llaisys_core.hpp"
#include "../tensor/tensor.hpp"

namespace llaisys::model {
class Qwen2Model;
using tensor_t = std::shared_ptr<Tensor>;

struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;   // a.k.a. model.norm.weight
    std::vector<tensor_t> attn_norm_w; // a.k.a. input_layernorm.weight
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w; // a.k.a. post_attention_layernorm.weight
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

class Qwen2Model {
private:
    Qwen2Meta _meta;
    Qwen2Weights _weights;
    llaisysDeviceType_t _device;
    int _device_id;
    int _ndevice;
    size_t _past_len;

    size_t _cache_capacity;
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;

public:
    Qwen2Model(const Qwen2Meta &meta, llaisysDeviceType_t device, int device_id, int ndevice);
    ~Qwen2Model() = default;
    const Qwen2Weights &weights() const;
    const Qwen2Meta &meta() const;
    int64_t infer( int64_t *token_ids, size_t ntoken);

    void reset();
};

} // namespace llasisy::model
