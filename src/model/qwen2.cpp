#include "qwen2.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <stdexcept>
#include <vector>
#include <cmath>

namespace llaisys::model {

    const Qwen2Weights &Qwen2Model::weights() const {
        return _weights;
    }

    const Qwen2Meta &Qwen2Model::meta() const {
        return _meta;
    }

    Qwen2Model::Qwen2Model(const Qwen2Meta &meta,
                        llaisysDeviceType_t device,
                        int device_id,
                        int ndevice)
        : _meta(meta),
        _device(device),
        _device_id(device_id),
        _ndevice(ndevice),
        _past_len(0) {
        if (_ndevice != 1) {
            throw std::invalid_argument("Qwen2Model currently supports exactly one device.");
        }
        if (_meta.nlayer == 0 || _meta.hs == 0 || _meta.nh == 0 ||
            _meta.nkvh == 0 || _meta.dh == 0 || _meta.di == 0 ||
            _meta.maxseq == 0 || _meta.voc == 0) {
            throw std::invalid_argument("Qwen2Model metadata dimensions must be greater than zero.");
        }
        if (_meta.hs != _meta.nh * _meta.dh) {
            throw std::invalid_argument("Qwen2Model requires hs == nh * dh.");
        }
        if (_meta.nh % _meta.nkvh != 0) {
            throw std::invalid_argument("Qwen2Model requires nh to be divisible by nkvh.");
        }

        auto make_tensor = [this](std::vector<size_t> shape) {
            return Tensor::create(shape, _meta.dtype, _device, _device_id);
        };

        // Model-wide parameters.
        _weights.in_embed = make_tensor({_meta.voc, _meta.hs});
        _weights.out_embed = make_tensor({_meta.voc, _meta.hs});
        _weights.out_norm_w = make_tensor({_meta.hs});

        // Each vector contains one tensor for every Transformer layer.
        _weights.attn_norm_w.resize(_meta.nlayer);
        _weights.attn_q_w.resize(_meta.nlayer);
        _weights.attn_q_b.resize(_meta.nlayer);
        _weights.attn_k_w.resize(_meta.nlayer);
        _weights.attn_k_b.resize(_meta.nlayer);
        _weights.attn_v_w.resize(_meta.nlayer);
        _weights.attn_v_b.resize(_meta.nlayer);
        _weights.attn_o_w.resize(_meta.nlayer);
        _weights.mlp_norm_w.resize(_meta.nlayer);
        _weights.mlp_gate_w.resize(_meta.nlayer);
        _weights.mlp_up_w.resize(_meta.nlayer);
        _weights.mlp_down_w.resize(_meta.nlayer);

        const size_t q_features = _meta.nh * _meta.dh;
        const size_t kv_features = _meta.nkvh * _meta.dh;

        for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
            _weights.attn_norm_w[layer] = make_tensor({_meta.hs});

            _weights.attn_q_w[layer] = make_tensor({q_features, _meta.hs});
            _weights.attn_q_b[layer] = make_tensor({q_features});
            _weights.attn_k_w[layer] = make_tensor({kv_features, _meta.hs});
            _weights.attn_k_b[layer] = make_tensor({kv_features});
            _weights.attn_v_w[layer] = make_tensor({kv_features, _meta.hs});
            _weights.attn_v_b[layer] = make_tensor({kv_features});
            _weights.attn_o_w[layer] = make_tensor({_meta.hs, q_features});

            _weights.mlp_norm_w[layer] = make_tensor({_meta.hs});
            _weights.mlp_gate_w[layer] = make_tensor({_meta.di, _meta.hs});
            _weights.mlp_up_w[layer] = make_tensor({_meta.di, _meta.hs});
            _weights.mlp_down_w[layer] = make_tensor({_meta.hs, _meta.di});
        }

        _k_cache.resize(_meta.nlayer);
        _v_cache.resize(_meta.nlayer);
        _cache_capacity = _meta.maxseq;

        for (size_t layer = 0; layer < _meta.nlayer;layer++)
        {
            _k_cache[layer] = Tensor::create({_cache_capacity, _meta.nkvh, _meta.dh},_meta
            .dtype,_device,_device_id);
            _v_cache[layer] = Tensor::create({_cache_capacity, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
        }
    }
    int64_t Qwen2Model::infer(int64_t *token_ids, size_t ntoken)
    {
        const size_t old_past_len = _past_len;
        const size_t total_len = old_past_len + ntoken;

        CHECK_ARGUMENT(_past_len <= _cache_capacity && ntoken <= _cache_capacity - _past_len, "KVCACHE EXCEEDED");

        auto input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
        input_ids->load(token_ids);

        auto hidden=Tensor::create({ntoken,_meta.hs},_meta.dtype,_device,_device_id);

        ops::embedding(hidden, input_ids, _weights.in_embed);

        std::vector<int64_t> host_pos_ids(ntoken);
        for (size_t i = 0; i < ntoken;i++)
        {
            host_pos_ids[i] = static_cast<int64_t>(_past_len + i);
        }

        tensor_t tensor_pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);

        tensor_pos_ids->load(host_pos_ids.data());

        for (size_t layer = 0; layer < _meta.nlayer; layer++) {
            tensor_t residual = hidden;

            tensor_t normed = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
            ops::rms_norm(normed, hidden, _weights.attn_norm_w[layer], _meta.epsilon);

            tensor_t q_2d = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
            ops::linear(q_2d, normed, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);
            tensor_t q = q_2d->view({ntoken, _meta.nh, _meta.dh});

            tensor_t k_2d = Tensor::create({ntoken, _meta.nkvh * _meta.dh}, _meta.dtype, _device, _device_id);
            ops::linear(k_2d, normed, _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
            tensor_t k = k_2d->view({ntoken, _meta.nkvh, _meta.dh});

            tensor_t v_2d = Tensor::create({ntoken, _meta.nkvh*_meta.dh}, _meta.dtype, _device, _device_id);
            ops::linear(v_2d, normed, _weights.attn_v_w[layer], _weights.attn_v_b[layer]);
            tensor_t v = v_2d->view({ntoken, _meta.nkvh, _meta.dh});

            tensor_t q_rotated = Tensor::create({ntoken, _meta.nh,_meta.dh}, _meta.dtype, _device, _device_id);
            tensor_t k_rotated = Tensor::create({ntoken, _meta.nkvh , _meta.dh}, _meta.dtype, _device, _device_id);

            ops::rope(q_rotated, q, tensor_pos_ids, _meta.theta);
            ops::rope(k_rotated, k, tensor_pos_ids, _meta.theta);

            auto k_destination = _k_cache[layer]->slice(0, old_past_len, total_len);
            auto v_destination = _v_cache[layer]->slice(0, old_past_len, total_len);

            k_destination->load(k_rotated->data());
            v_destination->load(v->data());

            auto all_k = _k_cache[layer]->slice(0, 0, total_len);
            auto all_v = _v_cache[layer]->slice(0, 0, total_len);

            float scales = 1.0f / std::sqrt(static_cast<float>(_meta.dh));

            tensor_t attn_value = Tensor::create({ntoken, _meta.nh,_meta.dh}, _meta.dtype, _device, _device_id);

            ops::self_attention(attn_value, q_rotated, all_k,all_v, scales);

            tensor_t attn_2d = attn_value->view({ntoken, _meta.nh * _meta.dh});

            tensor_t attn_output = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);

            ops::linear(attn_output, attn_2d, _weights.attn_o_w[layer], nullptr);

            tensor_t hidden_after_attn = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);

            ops::add(hidden_after_attn, attn_output, residual);

            hidden=hidden_after_attn;
            residual = hidden;

            ops::rms_norm(normed, hidden, _weights.mlp_norm_w[layer], _meta.epsilon);

            tensor_t gate = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
            ops::linear(gate, normed, _weights.mlp_gate_w[layer], nullptr);

            tensor_t up = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
            ops::linear(up, normed, _weights.mlp_up_w[layer], nullptr);

            tensor_t activated = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
            ops::swiglu(activated, gate, up);

            tensor_t mlp_output = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
            ops::linear(mlp_output, activated, _weights.mlp_down_w[layer], nullptr);

            tensor_t hidden_after_mlp = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);

            ops::add(hidden_after_mlp, residual, mlp_output);
            hidden = hidden_after_mlp;
        }
        _past_len = total_len;

        tensor_t final_hidden = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
        ops::rms_norm(final_hidden, hidden, _weights.out_norm_w, _meta.epsilon);

        auto last_hidden=final_hidden->slice(0,ntoken-1,ntoken);
        auto logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device, _device_id);

        ops::linear(logits, last_hidden, _weights.out_embed, nullptr);

        auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
        auto max_val = Tensor::create({1}, _meta.dtype, _device, _device_id);

        ops::argmax(max_idx, max_val, logits);

        return *(reinterpret_cast<int64_t*>(max_idx->data()));
    }

    void Qwen2Model::reset()
    {
        _past_len = 0;
    }

} // namespace llaisys::model
