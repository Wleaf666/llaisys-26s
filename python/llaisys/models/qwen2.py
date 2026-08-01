import ctypes
import json
import re
import warnings
from pathlib import Path
from typing import Sequence

import safetensors

from ..libllaisys import (
    LIB_LLAISYS,
    DataType,
    DeviceType,
    LlaisysQwen2Meta,
)


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self._model = None
        self._weights = None
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        dtype_name = config["torch_dtype"]
        dtype_map = {
            "float32": DataType.F32,
            "float16": DataType.F16,
            "bfloat16": DataType.BF16,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported model dtype: {dtype_name}")

        nlayer = config["num_hidden_layers"]
        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        nkvh = config["num_key_value_heads"]
        if hs % nh != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if nh % nkvh != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        meta = LlaisysQwen2Meta(
            dtype=int(dtype_map[dtype_name]),
            nlayer=nlayer,
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=hs // nh,
            di=config["intermediate_size"],
            maxseq=config["max_position_embeddings"],
            voc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config.get("rope_theta", 10000.0),
            end_token=config["eos_token_id"],
        )

        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            int(device),
            device_ids,
            1,
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        self._meta = meta
        self._config = config
        self._model_path = model_path
        self._source_dtype = {
            "float32": "torch.float32",
            "float16": "torch.float16",
            "bfloat16": "torch.bfloat16",
        }[dtype_name]

        try:
            self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
            if not self._weights:
                raise RuntimeError("Failed to get Qwen2 weight handles")
            self._load_weights(model_path)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _tensor_shape(tensor_handle):
        ndim = int(LIB_LLAISYS.tensorGetNdim(tensor_handle))
        shape_buffer = (ctypes.c_size_t * ndim)()
        LIB_LLAISYS.tensorGetShape(tensor_handle, shape_buffer)
        return tuple(shape_buffer)

    def _load_weights(self, model_path: Path):
        weights = self._weights.contents
        global_weights = {
            "model.embed_tokens.weight": weights.in_embed,
            "model.norm.weight": weights.out_norm_w,
            "lm_head.weight": weights.out_embed,
        }
        layer_weights = {
            "input_layernorm.weight": "attn_norm_w",
            "self_attn.q_proj.weight": "attn_q_w",
            "self_attn.q_proj.bias": "attn_q_b",
            "self_attn.k_proj.weight": "attn_k_w",
            "self_attn.k_proj.bias": "attn_k_b",
            "self_attn.v_proj.weight": "attn_v_w",
            "self_attn.v_proj.bias": "attn_v_b",
            "self_attn.o_proj.weight": "attn_o_w",
            "post_attention_layernorm.weight": "mlp_norm_w",
            "mlp.gate_proj.weight": "mlp_gate_w",
            "mlp.up_proj.weight": "mlp_up_w",
            "mlp.down_proj.weight": "mlp_down_w",
        }

        expected = set(global_weights)
        for layer in range(self._meta.nlayer):
            expected.update(
                f"model.layers.{layer}.{suffix}" for suffix in layer_weights
            )

        weight_files = sorted(model_path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        loaded = set()
        unknown = set()
        layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.+)")

        for weight_file in weight_files:
            with safetensors.safe_open(
                str(weight_file), framework="pt", device="cpu"
            ) as tensors:
                for name in tensors.keys():
                    if name in loaded:
                        raise RuntimeError(f"Duplicate weight: {name}")

                    handle = global_weights.get(name)
                    if handle is None:
                        match = layer_pattern.fullmatch(name)
                        if match is None:
                            unknown.add(name)
                            continue

                        layer = int(match.group(1))
                        suffix = match.group(2)
                        field_name = layer_weights.get(suffix)
                        if field_name is None:
                            unknown.add(name)
                            continue
                        if layer >= self._meta.nlayer:
                            raise IndexError(
                                f"Weight layer {layer} is outside [0, {self._meta.nlayer})"
                            )
                        handle = getattr(weights, field_name)[layer]

                    if not handle:
                        raise RuntimeError(f"Backend returned a null handle for {name}")

                    source = tensors.get_tensor(name).contiguous()
                    expected_shape = self._tensor_shape(handle)
                    if tuple(source.shape) != expected_shape:
                        raise ValueError(
                            f"Shape mismatch for {name}: source={tuple(source.shape)}, "
                            f"destination={expected_shape}"
                        )
                    if str(source.dtype) != self._source_dtype:
                        raise ValueError(
                            f"Dtype mismatch for {name}: source={source.dtype}, "
                            f"expected={self._source_dtype}"
                        )
                    if int(LIB_LLAISYS.tensorGetDataType(handle)) != self._meta.dtype:
                        raise ValueError(f"Backend dtype mismatch for {name}")

                    LIB_LLAISYS.tensorLoad(
                        handle, ctypes.c_void_p(source.data_ptr())
                    )
                    loaded.add(name)

        missing = expected - loaded
        if missing:
            preview = ", ".join(sorted(missing)[:10])
            raise RuntimeError(
                f"Missing {len(missing)} required weights; first entries: {preview}"
            )
        if unknown:
            warnings.warn(
                "Ignored unknown weights: " + ", ".join(sorted(unknown)),
                RuntimeWarning,
            )

    def close(self):
        if self._model:
            self._weights = None
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def __del__(self):
        self.close()

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if not self._model:
          raise RuntimeError("Qwen2 model has been closed")
        outputs = [int(token) for token in inputs]
        current_inputs=outputs

        if not outputs:
          raise ValueError("inputs must not be empty")

        if max_new_tokens is None:
          max_new_tokens = 128

        if max_new_tokens < 0:
          raise ValueError("max_new_tokens must be non-negative")

        if top_k != 1:
          raise NotImplementedError(
              "Only top_k=1 greedy generation is currently supported"
          )

        if len(outputs) + max_new_tokens > self._meta.maxseq:
          raise ValueError(
              "Generated sequence would exceed max_position_embeddings"
          )

        for token in outputs:
          if token < 0 or token >= self._meta.voc:
              raise ValueError(f"Token ID out of range: {token}")

        for _ in range(max_new_tokens):
          token_buffer = (
              ctypes.c_int64 * len(current_inputs)
          )(*current_inputs)

          next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
              self._model,
              token_buffer,
              ctypes.c_size_t(len(current_inputs)),
          )
          next_token = int(next_token)

          outputs.append(next_token)

          if next_token == self._meta.end_token:
              break
          current_inputs=[next_token]

        return outputs
