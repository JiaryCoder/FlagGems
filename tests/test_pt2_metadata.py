# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile metadata regression checks; tensor work in this file runs on CPU."""

import pytest
import torch
from flag_gems.utils.shape_utils import is_non_overlapping_and_dense, stride_order
from flag_gems.utils.type_utils import type_promotion
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("kind", list(ELEMENTWISE_TYPE_PROMOTION_KIND))
def test_compiled_promotion_preserves_metadata_guards(kind):
    def fn(a, b):
        computation, result = type_promotion(a, b, type_promotion=kind)
        return torch.empty_like(a, dtype=computation), torch.empty_like(a, dtype=result)

    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    cases = [
        (torch.ones(3, dtype=torch.float16), torch.ones(3, dtype=torch.float32)),
        (torch.ones(3, dtype=torch.bfloat16), torch.ones(3, dtype=torch.float16)),
        (torch.ones(3, dtype=torch.float16), torch.ones((), dtype=torch.float32)),
        (torch.ones(3, dtype=torch.float32), torch.ones((), dtype=torch.float64)),
        (torch.ones(3, dtype=torch.int32), 0.5),
        (torch.ones(3, dtype=torch.complex64), torch.ones(3, dtype=torch.float64)),
    ]
    for a, b in cases:
        expected = elementwise_dtypes(a, b, type_promotion_kind=kind)
        actual = compiled(a, b)
        assert tuple(t.dtype for t in actual) == expected


def test_compiled_promotion_observes_default_dtype_change():
    def fn(a, b):
        _, dtype = type_promotion(
            a, b, type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        )
        return torch.empty_like(a, dtype=dtype)

    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    original = torch.get_default_dtype()
    try:
        for dtype in (torch.float32, torch.float64, torch.float32):
            torch.set_default_dtype(dtype)
            a = torch.ones(3, dtype=torch.int32)
            assert compiled(a, 0.5).dtype == (a + 0.5).dtype
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize(
    "shape,strides",
    [
        ((), ()),
        ((4, 7), (7, 1)),
        ((7, 4), (1, 7)),
        ((4, 4), (7, 2)),
        ((4, 7), (1, 0)),
        ((0, 7), (100, 5)),
        ((1, 1), (100, 5)),
        ((9, 4, 7), (1, 63, 9)),
    ],
)
def test_compiled_density_and_stride_order(shape, strides):
    def fn(x):
        return x + 1, is_non_overlapping_and_dense(x), tuple(stride_order(x.stride()))

    x = torch.empty_strided(shape, strides).zero_()
    compiled = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
    output, dense, order = compiled(x)
    torch.testing.assert_close(output, x + 1)
    assert dense == torch.ops.aten.is_non_overlapping_and_dense(x)
    assert order == tuple(sorted(range(x.ndim), key=lambda i: abs(x.stride(i))))
