# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation API compatibility and explicit extension of the rank boundary."""

import importlib

import pytest
import torch
import torch.nn.functional as F

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" or not hasattr(torch.library, "triton_op"),
    reason="PT2 adapters currently target NVIDIA with torch.library.triton_op",
)


@pytest.fixture
def pointwise():
    return importlib.import_module("flag_gems.pt2.pointwise_dynamic")


def test_legacy_layout_alias_and_diagnostics(pointwise):
    alias = pointwise.materialize_silu_and_mul_plan(layout_class="split_last_dim_c")
    canonical = pointwise.materialize_silu_and_mul_plan(layout_class="strided")
    assert alias is canonical
    assert alias.tensor_stride_order == (1, 0)
    assert alias.scalar_stride_order == (0, 1)


def test_default_preparation_has_no_duplicate_layouts(pointwise):
    plans = pointwise.materialize_pointwise_family_plans(
        ("silu_and_mul",), dtypes=(torch.float32,)
    )
    assert len(plans) == 2
    assert {p.layout_class for p in plans} == {"contiguous_c", "strided"}
    assert len({p.token for p in plans}) == len(plans)


def test_preparation_accepts_one_shot_iterables(pointwise):
    families = ("silu_and_mul", "gelu_and_mul.none")
    ranks = (1, 2)
    dtypes = (torch.float16, torch.float32)
    plans = pointwise.materialize_pointwise_family_plans(
        iter(families),
        ranks=iter(ranks),
        dtypes=iter(dtypes),
        layout_classes=iter(("strided",)),
    )
    assert {(p.op_name, p.ndim, p.dtype) for p in plans} == {
        (name, rank, dtype) for name in families for rank in ranks for dtype in dtypes
    }


@pytest.mark.parametrize(
    "dtype,shape", [(torch.float64, (2, 7)), (torch.float64, ()), (torch.int32, (2, 7))]
)
def test_unsupported_input_dtype_has_specific_error(pointwise, dtype, shape):
    x = torch.ones(2, 7)
    y = torch.ones(shape, dtype=dtype)
    with pytest.raises(
        ValueError, match="supported dtypes are float16, bfloat16 and float32"
    ):
        pointwise._resolve_plan_token("silu_and_mul", (x, y))


def test_missing_plan_requests_preparation(pointwise):
    x = torch.ones((1,) * 8)
    with pytest.raises(RuntimeError, match="Materialize it outside Dynamo"):
        pointwise._resolve_plan_token("silu_and_mul", (x, x))


def test_explicit_preparation_extends_rank_for_forward_and_backward(pointwise):
    pointwise.materialize_pointwise_family_plans(
        ("silu_and_mul", "silu_and_mul.backward"),
        ranks=(6,),
        dtypes=(torch.float32,),
    )
    x = torch.randn(2, 1, 3, 1, 1, 7, device=flag_gems.device, requires_grad=True)
    y = torch.randn(7, device=flag_gems.device, requires_grad=True)
    rx, ry = (t.detach().double().requires_grad_() for t in (x, y))
    torch._dynamo.reset()
    try:
        out = torch.compile(flag_gems.silu_and_mul, fullgraph=True, dynamic=True)(x, y)
        expected = F.silu(rx) * ry
        torch.testing.assert_close(out.double(), expected, rtol=1e-4, atol=1e-5)
        actual_grads = torch.autograd.grad(out.sum(), (x, y))
        expected_grads = torch.autograd.grad(expected.sum(), (rx, ry))
        for actual, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual.double(), reference, rtol=1e-4, atol=1e-5)
    finally:
        torch._dynamo.reset()
