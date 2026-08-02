
#include <memory>
#include "llaisys/models/qwen2.h"
#include "../model/qwen2.hpp"
#include "llaisys_tensor.hpp"



__C {

    struct LlaisysQwen2Model {
        std::unique_ptr<llaisys::model::Qwen2Model> model;

        LlaisysQwen2Weights weights_view;

        std::vector<std::unique_ptr<LlaisysTensor>> tensor_handles;

        using vector_tensor_t = std::vector<llaisysTensor_t>;

        vector_tensor_t attn_norm_w_handles;
        vector_tensor_t attn_q_w_handles;
        vector_tensor_t attn_q_b_handles;
        vector_tensor_t attn_k_w_handles;
        vector_tensor_t attn_k_b_handles;
        vector_tensor_t attn_v_w_handles;
        vector_tensor_t attn_v_b_handles;
        vector_tensor_t attn_o_w_handles;

        vector_tensor_t mlp_norm_w_handles;
        vector_tensor_t mlp_gate_w_handles;
        vector_tensor_t mlp_up_w_handles;
        vector_tensor_t mlp_down_w_handles;
    };

    static llaisysTensor_t wrapTensor(LlaisysQwen2Model *model,const llaisys::tensor_t &tensor)
    {
        auto wrapper = std::make_unique<LlaisysTensor>();
        wrapper->tensor = tensor;

        llaisysTensor_t handle = wrapper.get();
        model->tensor_handles.push_back(std::move(wrapper));

        return handle;
    }

    

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
    {
        llaisys::model::Qwen2Meta internal_meta{
            meta->dtype,
            meta->nlayer,
            meta->hs,
            meta->nh,
            meta->nkvh,
            meta->dh,
            meta->di,
            meta->maxseq,
            meta->voc,
            meta->epsilon,
            meta->theta,
            meta->end_token,

        };
        LlaisysQwen2Model* result=new LlaisysQwen2Model{std::make_unique<llaisys::model::Qwen2Model>(internal_meta, device, device_ids[0], ndevice)};

        const auto &internal_weights = result->model->weights();

        result->weights_view.in_embed = wrapTensor(result, internal_weights.in_embed);
        result->weights_view.out_embed = wrapTensor(result, internal_weights.out_embed);
        result->weights_view.out_norm_w = wrapTensor(result, internal_weights.out_norm_w);

        result->attn_norm_w_handles.resize(meta->nlayer);
        result->attn_q_w_handles.resize(meta->nlayer);
        result->attn_q_b_handles.resize(meta->nlayer);
        result->attn_k_w_handles.resize(meta->nlayer);
        result->attn_k_b_handles.resize(meta->nlayer);
        result->attn_v_w_handles.resize(meta->nlayer);
        result->attn_v_b_handles.resize(meta->nlayer);
        result->attn_o_w_handles.resize(meta->nlayer);
        result->mlp_norm_w_handles.resize(meta->nlayer);
        result->mlp_gate_w_handles.resize(meta->nlayer);
        result->mlp_up_w_handles.resize(meta->nlayer);
        result->mlp_down_w_handles.resize(meta->nlayer);

        for (size_t layer = 0; layer < meta->nlayer;layer++)
        {
            result->attn_norm_w_handles[layer] = wrapTensor(result, internal_weights.attn_norm_w[layer]);
            result->attn_q_w_handles[layer] = wrapTensor(result, internal_weights.attn_q_w[layer]);
            result->attn_q_b_handles[layer] = wrapTensor(result, internal_weights.attn_q_b[layer]);
            result->attn_k_w_handles[layer] = wrapTensor(result, internal_weights.attn_k_w[layer]);
            result->attn_k_b_handles[layer] = wrapTensor(result, internal_weights.attn_k_b[layer]);
            result->attn_v_w_handles[layer] = wrapTensor(result, internal_weights.attn_v_w[layer]);
            result->attn_v_b_handles[layer] = wrapTensor(result, internal_weights.attn_v_b[layer]);
            result->attn_o_w_handles[layer] = wrapTensor(result, internal_weights.attn_o_w[layer]);
            result->mlp_norm_w_handles[layer] = wrapTensor(result, internal_weights.mlp_norm_w[layer]);
            result->mlp_gate_w_handles[layer] = wrapTensor(result, internal_weights.mlp_gate_w[layer]);
            result->mlp_up_w_handles[layer] = wrapTensor(result, internal_weights.mlp_up_w[layer]);
            result->mlp_down_w_handles[layer] = wrapTensor(result, internal_weights.mlp_down_w[layer]);
        }
        result->weights_view.attn_norm_w = result->attn_norm_w_handles.data();
        result->weights_view.attn_q_w = result->attn_q_w_handles.data();
        result->weights_view.attn_q_b = result->attn_q_b_handles.data();
        result->weights_view.attn_k_w = result->attn_k_w_handles.data();
        result->weights_view.attn_k_b =result->attn_k_b_handles.data();
        result->weights_view.attn_v_w = result->attn_v_w_handles.data();
        result->weights_view.attn_v_b = result->attn_v_b_handles.data();
        result->weights_view.attn_o_w = result->attn_o_w_handles.data();
        result->weights_view.mlp_norm_w = result->mlp_norm_w_handles.data();
        result->weights_view.mlp_gate_w = result->mlp_gate_w_handles.data();
        result->weights_view.mlp_up_w = result->mlp_up_w_handles.data();
        result->weights_view.mlp_down_w = result->mlp_down_w_handles.data();

        return result;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model)
    {
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model)
    {
        return &model->weights_view;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken)
    {
        return model->model->infer(token_ids, ntoken);
    }

    void llaisysQwen2ModelReset(struct LlaisysQwen2Model * model)
    {
        model->model->reset();
    }
}
