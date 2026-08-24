from typing import Sequence, Tuple

import struct

from .libllaisys import (
    LIB_LLAISYS,
    llaisysTensor_t,
    llaisysDeviceType_t,
    DeviceType,
    llaisysDataType_t,
    DataType,
)
from ctypes import c_size_t, c_int, c_ssize_t, c_void_p, c_byte

# dtype -> struct format for scalar unpacking (little-endian)
_SCALAR_FMT = {
    DataType.BYTE: "<b",
    DataType.BOOL: "<?",
    DataType.I8: "<b",
    DataType.I16: "<h",
    DataType.I32: "<i",
    DataType.I64: "<q",
    DataType.U8: "<B",
    DataType.U16: "<H",
    DataType.U32: "<I",
    DataType.U64: "<Q",
    DataType.F16: "<e",
    DataType.F32: "<f",
    DataType.F64: "<d",
    DataType.BF16: "<H",  # raw bits; converted to float in to_scalar
}


class Tensor:
    def __init__(
        self,
        shape: Sequence[int] = None,
        dtype: DataType = DataType.F32,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
        tensor: llaisysTensor_t = None,
    ):
        if tensor:
            self._tensor = tensor
        else:
            _ndim = 0 if shape is None else len(shape)
            _shape = None if shape is None else (c_size_t * len(shape))(*shape)
            self._tensor: llaisysTensor_t = LIB_LLAISYS.tensorCreate(
                _shape,
                c_size_t(_ndim),
                llaisysDataType_t(dtype),
                llaisysDeviceType_t(device),
                c_int(device_id),
            )

    def __del__(self):
        if hasattr(self, "_tensor") and self._tensor is not None:
            LIB_LLAISYS.tensorDestroy(self._tensor)
            self._tensor = None

    def shape(self) -> Tuple[int]:
        buf = (c_size_t * self.ndim())()
        LIB_LLAISYS.tensorGetShape(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def strides(self) -> Tuple[int]:
        buf = (c_ssize_t * self.ndim())()
        LIB_LLAISYS.tensorGetStrides(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def ndim(self) -> int:
        return int(LIB_LLAISYS.tensorGetNdim(self._tensor))

    def dtype(self) -> DataType:
        return DataType(LIB_LLAISYS.tensorGetDataType(self._tensor))

    def device_type(self) -> DeviceType:
        return DeviceType(LIB_LLAISYS.tensorGetDeviceType(self._tensor))

    def device_id(self) -> int:
        return int(LIB_LLAISYS.tensorGetDeviceId(self._tensor))

    def data_ptr(self) -> c_void_p:
        return LIB_LLAISYS.tensorGetData(self._tensor)

    def lib_tensor(self) -> llaisysTensor_t:
        return self._tensor

    def debug(self):
        LIB_LLAISYS.tensorDebug(self._tensor)

    def __repr__(self):
        return f"<Tensor shape={self.shape}, dtype={self.dtype}, device={self.device_type}:{self.device_id}>"

    def load(self, data: c_void_p):
        LIB_LLAISYS.tensorLoad(self._tensor, data)

    def is_contiguous(self) -> bool:
        return bool(LIB_LLAISYS.tensorIsContiguous(self._tensor))

    def view(self, *shape: int) -> llaisysTensor_t:
        _shape = (c_size_t * len(shape))(*shape)
        return Tensor(
            tensor=LIB_LLAISYS.tensorView(self._tensor, _shape, c_size_t(len(shape)))
        )

    def permute(self, *perm: int) -> llaisysTensor_t:
        assert len(perm) == self.ndim()
        _perm = (c_size_t * len(perm))(*perm)
        return Tensor(tensor=LIB_LLAISYS.tensorPermute(self._tensor, _perm))

    def slice(self, dim: int, start: int, end: int):
        return Tensor(
            tensor=LIB_LLAISYS.tensorSlice(
                self._tensor, c_size_t(dim), c_size_t(start), c_size_t(end)
            )
        )

    def contiguous(self):
        return Tensor(tensor=LIB_LLAISYS.tensorContiguous(self._tensor))

    def reshape(self, *shape: int):
        _shape = (c_size_t * len(shape))(*shape)
        return Tensor(
            tensor=LIB_LLAISYS.tensorReshape(self._tensor, _shape, c_size_t(len(shape)))
        )

    def to(self, device: DeviceType, device_id: int = 0):
        return Tensor(
            tensor=LIB_LLAISYS.tensorTo(
                self._tensor, llaisysDeviceType_t(device), c_int(device_id)
            )
        )

    def copy_from(self, src: "Tensor"):
        LIB_LLAISYS.tensorCopyFrom(self._tensor, src.lib_tensor())

    def to_scalar(self):
        dtype = self.dtype()
        fmt = _SCALAR_FMT.get(dtype)
        if fmt is None:
            raise NotImplementedError(f"to_scalar is not supported for dtype {dtype}")
        buf = (c_byte * struct.calcsize(fmt))()
        LIB_LLAISYS.tensorToScalar(self._tensor, buf)
        value = struct.unpack(fmt, bytes(buf))[0]
        if dtype == DataType.BF16:
            # reinterpret bf16 bits as f32 bits (shift left by 16)
            value = struct.unpack("<f", struct.pack("<I", value << 16))[0]
        return value
