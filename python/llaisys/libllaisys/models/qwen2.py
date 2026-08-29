import ctypes
import json
from pathlib import Path

from ctypes import (
    Structure,
    POINTER,
    c_void_p,
    c_char_p,
    c_int,
    c_int64,
    c_size_t,
    c_float,
    c_bool,
)

from ..llaisys_types import (
    llaisysDeviceType_t,
    llaisysDataType_t,
    DeviceType,
)


# -------------------------
# 1) mirror C struct layout
# -------------------------
class Qwen2ModelMeta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hidden_size", c_size_t),
        ("nhead", c_size_t),
        ("nkvhead", c_size_t),
        ("d_intermediate", c_size_t),
        ("maxseq", c_size_t),
        ("vocab_size", c_size_t),
        ("rms_epsilon", c_float),
        ("rope_theta", c_float),
        ("end_token", c_int64),
        ("use_sliding_window", c_bool),
        ("use_cache", c_bool),
    ]


class llaisysQwen2Meta(Structure):
    _fields_ = [
        ("meta", Qwen2ModelMeta),
    ]


llaisysQwen2Model_t = c_void_p
llaisysQwen2Weights_t = c_void_p


# -------------------------
# 2) bind C functions
# -------------------------
def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(llaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysQwen2ModelCreate.restype = llaisysQwen2Model_t

    lib.llaisysQwen2ModelDestroy.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelLoadWeights.argtypes = [
        llaisysQwen2Model_t,
        c_char_p,
        c_void_p,
        c_size_t,
    ]
    lib.llaisysQwen2ModelLoadWeights.restype = (
        c_int  # Return an int to indicate success or failure
    )

    lib.llaisysQwen2ModelInfer.argtypes = [
        llaisysQwen2Model_t,
        POINTER(c_int64),
        c_size_t,
        c_int64,
        c_float,
        c_float,
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    # optional, keep if you want to expose it
    lib.llaisysQwen2ModelWeights.argtypes = [llaisysQwen2Model_t]
    lib.llaisysQwen2ModelWeights.restype = llaisysQwen2Weights_t
