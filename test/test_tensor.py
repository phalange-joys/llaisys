import llaisys

import torch
from test_utils import *
import argparse


def test_tensor():
    torch_tensor = torch.arange(60, dtype=torch_dtype("i64")).reshape(3, 4, 5)
    llaisys_tensor = llaisys.Tensor(
        (3, 4, 5), dtype=llaisys_dtype("i64"), device=llaisys_device("cpu")
    )

    # Test load
    print("===Test load===")
    llaisys_tensor.load(torch_tensor.data_ptr())
    llaisys_tensor.debug()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor, torch_tensor)

    # Test view
    print("===Test view===")
    torch_tensor_view = torch_tensor.view(6, 10)
    llaisys_tensor_view = llaisys_tensor.view(6, 10)
    llaisys_tensor_view.debug()
    assert llaisys_tensor_view.shape() == torch_tensor_view.shape
    assert llaisys_tensor_view.strides() == torch_tensor_view.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_view, torch_tensor_view)

    # Test permute
    print("===Test permute===")
    torch_tensor_perm = torch_tensor.permute(2, 0, 1)
    llaisys_tensor_perm = llaisys_tensor.permute(2, 0, 1)
    llaisys_tensor_perm.debug()
    assert llaisys_tensor_perm.shape() == torch_tensor_perm.shape
    assert llaisys_tensor_perm.strides() == torch_tensor_perm.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_perm, torch_tensor_perm)

    # Test slice
    print("===Test slice===")
    torch_tensor_slice = torch_tensor[:, :, 1:4]
    llaisys_tensor_slice = llaisys_tensor.slice(2, 1, 4)
    llaisys_tensor_slice.debug()
    assert llaisys_tensor_slice.shape() == torch_tensor_slice.shape
    assert llaisys_tensor_slice.strides() == torch_tensor_slice.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_slice, torch_tensor_slice)

    # Test contiguous
    print("===Test contiguous===")
    torch_tensor_contiguous = torch_tensor_perm.contiguous()
    llaisys_tensor_contiguous = llaisys_tensor_perm.contiguous()
    llaisys_tensor_contiguous.debug()
    assert llaisys_tensor_contiguous.is_contiguous()
    assert llaisys_tensor_contiguous.shape() == torch_tensor_contiguous.shape
    assert llaisys_tensor_contiguous.strides() == torch_tensor_contiguous.stride()
    assert check_equal(llaisys_tensor_contiguous, torch_tensor_contiguous)

    # Test reshape on a non-contiguous tensor
    print("===Test reshape===")
    torch_tensor_reshape = torch_tensor_perm.reshape(2, 30)
    llaisys_tensor_reshape = llaisys_tensor_perm.reshape(2, 30)
    llaisys_tensor_reshape.debug()
    assert llaisys_tensor_reshape.shape() == torch_tensor_reshape.shape
    assert llaisys_tensor_reshape.strides() == torch_tensor_reshape.stride()
    assert llaisys_tensor_reshape.is_contiguous()
    assert check_equal(llaisys_tensor_reshape, torch_tensor_reshape)

    # Test CPU to CPU copy
    print("===Test to===")
    torch_tensor_cpu = torch_tensor.to("cpu")
    llaisys_tensor_cpu = llaisys_tensor.to(llaisys_device("cpu"))
    llaisys_tensor_cpu.debug()
    assert llaisys_tensor_cpu.device_type() == llaisys_device("cpu")
    assert llaisys_tensor_cpu.shape() == torch_tensor_cpu.shape
    assert llaisys_tensor_cpu.strides() == torch_tensor_cpu.stride()
    assert check_equal(llaisys_tensor_cpu, torch_tensor_cpu)

    # Test copy_from
    print("===Test copy_from===")
    torch_src = torch.arange(6, dtype=torch_dtype("i64")).reshape(2, 3)
    llaisys_src = llaisys.Tensor(
        (2, 3), dtype=llaisys_dtype("i64"), device=llaisys_device("cpu")
    )
    llaisys_src.load(torch_src.data_ptr())
    llaisys_dst = llaisys.Tensor(
        (2, 3), dtype=llaisys_dtype("i64"), device=llaisys_device("cpu")
    )
    llaisys_dst.copy_from(llaisys_src)
    llaisys_dst.debug()
    assert check_equal(llaisys_dst, torch_src)

    # Test to_scalar
    print("===Test to_scalar===")
    torch_scalar = torch.tensor(42, dtype=torch_dtype("i64"))
    llaisys_scalar = llaisys.Tensor(
        (1,), dtype=llaisys_dtype("i64"), device=llaisys_device("cpu")
    )
    llaisys_scalar.load(torch_scalar.data_ptr())
    assert llaisys_scalar.to_scalar() == 42

    torch_scalar_f = torch.tensor(3.14, dtype=torch_dtype("f32"))
    llaisys_scalar_f = llaisys.Tensor(
        (1,), dtype=llaisys_dtype("f32"), device=llaisys_device("cpu")
    )
    llaisys_scalar_f.load(torch_scalar_f.data_ptr())
    assert abs(llaisys_scalar_f.to_scalar() - 3.14) < 1e-6


if __name__ == "__main__":
    test_tensor()

    print("\n\033[92mTest passed!\033[0m\n")
