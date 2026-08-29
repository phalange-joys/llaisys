from typing import Sequence, List, Optional
from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    llaisysDeviceType_t,
    llaisysQwen2Model_t,
    llaisysQwen2Weights_t,
    llaisysQwen2Meta,
    load_qwen2,
)
from pathlib import Path
import safetensors
import json
import ctypes
import numpy as np


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device

        # Load model configuration
        with open(self.model_path / "config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)

        meta = self._prepare_meta()

        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            llaisysDeviceType_t(device.value),
            None,
            0,
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model.")

        self._load_weights()

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> List[int]:
        tokens = list(inputs)
        max_seqlen = self.config["max_position_embeddings"]
        max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else (max_seqlen - len(tokens))
        )
        eos_token_id = self.config["eos_token_id"]
        for i in range(max_new_tokens):
            c_inputs = (ctypes.c_int64 * len(tokens))(*tokens)
            token_id = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, c_inputs, len(tokens), top_k, top_p, temperature
            )
            tokens.append(token_id)
            if token_id == eos_token_id:
                break

        return tokens

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)

    def _prepare_meta(self):
        meta = llaisysQwen2Meta()

        dtype_map = {
            "float32": DataType.F32,
            "float16": DataType.F16,
            "bfloat16": DataType.BF16,
        }
        torch_dtype = self.config.get("torch_dtype", "bfloat16")
        meta.meta.dtype = dtype_map.get(torch_dtype, DataType.BF16).value

        meta.meta.nlayer = self.config.get("num_hidden_layers", 0)
        meta.meta.hidden_size = self.config.get("hidden_size", 0)
        meta.meta.nhead = self.config.get("num_attention_heads", 0)
        meta.meta.nkvhead = self.config.get("num_key_value_heads", meta.meta.nhead)
        meta.meta.d_intermediate = self.config.get("intermediate_size", 0)
        meta.meta.maxseq = self.config.get("max_position_embeddings", 0)
        meta.meta.vocab_size = self.config.get("vocab_size", 0)
        meta.meta.rms_epsilon = self.config.get("rms_norm_eps", 1e-6)
        meta.meta.rope_theta = self.config.get("rope_theta", 10000.0)
        meta.meta.end_token = self.config.get("eos_token_id", 151643)
        meta.meta.use_sliding_window = self.config.get("use_sliding_window", False)
        meta.meta.use_cache = self.config.get("use_cache", True)

        return meta

    def _load_weights(self):
        safetensors_files = sorted(self.model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self.model_path}")

        n_tensors = 0
        for file_path in safetensors_files:
            print(f"Loading: {file_path.name}")

            with open(file_path, "rb") as f:
                # safetensors bytes API: parse header
                header_size = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(header_size).decode("utf-8"))

                for name, info in header.items():
                    if name == "__metadata__":
                        continue

                    offset_start, offset_end = info["data_offsets"]
                    data_start = 8 + header_size + offset_start
                    data_size = offset_end - offset_start

                    f.seek(data_start)
                    raw_bytes = f.read(data_size)

                    # compute the number of elements
                    shape = info["shape"]
                    numel = 1
                    for dim in shape:
                        numel *= dim

                    c_numel = ctypes.c_size_t(numel)

                    buf = (ctypes.c_char * data_size).from_buffer_copy(raw_bytes)

                    result = LIB_LLAISYS.llaisysQwen2ModelLoadWeights(
                        self._model,
                        name.encode("utf-8"),
                        buf,
                        c_numel,
                    )
                    if result != 0:
                        print(f"  Warning: Failed to load {name}")
                        break

                    n_tensors += 1

        print(f"Loaded {n_tensors} tensors from {len(safetensors_files)} file(s)")
