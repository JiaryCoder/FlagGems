# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Public API training and inference contracts for the NVIDIA PT2 adapters."""

import flag_gems
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" or not hasattr(torch.library, "triton_op"),
    reason="PT2 adapters currently target NVIDIA with torch.library.triton_op",
)


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def activation(name, x, y):
    if name == "silu":
        return flag_gems.silu_and_mul(x, y)
    if name == "clamp":
        return flag_gems.silu_and_mul_with_clamp(x, y, 1.5)
    return flag_gems.gelu_and_mul(x, y, approximate=name)


def activation_reference(name, x, y):
    if name == "clamp":
        return F.silu(x.clamp(max=1.5)) * y.clamp(min=-1.5, max=1.5)
    if name == "silu":
        return F.silu(x) * y
    return F.gelu(x, approximate=name) * y


def inputs(n, width, dtype, layout):
    options = {"device": flag_gems.device, "dtype": dtype}
    if layout == "split":
        base = torch.randn(n, width * 2, **options)
        x, y = base[:, :width], base[:, width:]
    elif layout == "transpose":
        x, y = (torch.randn(width, n, **options).t() for _ in range(2))
    else:
        x = torch.randn(n, width, **options)
        if layout == "broadcast":
            y = torch.randn(width, **options)
        elif layout == "mixed":
            y = torch.randn(n, width * 2, **options)[:, :width]
        else:
            y = torch.randn_like(x)
    return x.detach().requires_grad_(), y.detach().requires_grad_()


def close(actual, expected, dtype):
    rtol, atol = {
        torch.float32: (5e-4, 5e-5),
        torch.float16: (5e-3, 2e-3),
        torch.bfloat16: (4e-2, 2e-2),
    }[dtype]
    torch.testing.assert_close(actual.double(), expected.double(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("name", ["silu", "none", "tanh", "clamp"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "layout", ["contiguous", "split", "broadcast", "mixed", "transpose"]
)
def test_public_activation_training(name, dtype, layout):
    def fn(x, y):
        return activation(name, x, y)

    compiled = torch.compile(fn, fullgraph=True, dynamic=True)
    # One compiled callable sees different token counts and fresh allocations.
    for n in (2, 5):
        x, y = inputs(n, 37, dtype, layout)
        rx = x.detach().double().requires_grad_()
        ry = y.detach().double().requires_grad_()
        expected = activation_reference(name, rx, ry)
        actual = compiled(x, y)
        close(actual, expected, dtype)
        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (x, y), grad)
        expected_grads = torch.autograd.grad(expected, (rx, ry), grad.double())
        for got, want in zip(actual_grads, expected_grads):
            close(got, want, dtype)


@pytest.mark.parametrize("name", ["silu", "none", "tanh", "clamp"])
@pytest.mark.parametrize("shape", [(), (0, 37), (2, 1, 3, 1, 7)])
def test_activation_scalar_empty_and_zero_stride_grad(name, shape):
    x = torch.randn(shape, device=flag_gems.device, requires_grad=True)
    y = torch.randn(shape, device=flag_gems.device, requires_grad=True)
    rx, ry = (arg.detach().double().requires_grad_() for arg in (x, y))
    compiled = torch.compile(lambda a, b: activation(name, a, b), fullgraph=True)
    actual = compiled(x, y)
    expected = activation_reference(name, rx, ry)
    # sum() exercises the expanded/zero-stride upstream gradient.
    got = torch.autograd.grad(actual.sum(), (x, y))
    want = torch.autograd.grad(expected.sum(), (rx, ry))
    for a, b in zip(got, want):
        close(a, b, x.dtype)


@pytest.mark.parametrize("grad_x,grad_y", [(True, False), (False, True)])
def test_activation_mixed_dtype_and_partial_grad(grad_x, grad_y):
    x = torch.randn(
        3, 37, device=flag_gems.device, dtype=torch.float16, requires_grad=grad_x
    )
    y = torch.randn(
        37, device=flag_gems.device, dtype=torch.float32, requires_grad=grad_y
    )
    rx = x.detach().double().requires_grad_(grad_x)
    ry = y.detach().double().requires_grad_(grad_y)
    out = torch.compile(flag_gems.silu_and_mul, fullgraph=True)(x, y)
    ref = F.silu(rx) * ry
    assert out.dtype == torch.float32
    target, reference = (x, rx) if grad_x else (y, ry)
    (got,) = torch.autograd.grad(out.sum(), (target,))
    (want,) = torch.autograd.grad(ref.sum(), (reference,))
    assert got.dtype == target.dtype
    close(got, want, target.dtype)


@pytest.mark.parametrize("width", [37, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("grad_x,grad_w", [(True, True), (True, False), (False, True)])
def test_public_rms_norm_training(width, dtype, grad_x, grad_w):
    def fn(x, w):
        return flag_gems.rms_norm(x, [width], w, 1e-6)

    compiled = torch.compile(fn, fullgraph=True, dynamic=True)
    for n in (2, 5):
        x = torch.randn(n, width * 2, device=flag_gems.device, dtype=dtype)[:, :width]
        x = x.detach().requires_grad_(grad_x)
        w = torch.randn(
            width, device=flag_gems.device, dtype=dtype, requires_grad=grad_w
        )
        rx = x.detach().double().requires_grad_(grad_x)
        rw = w.detach().double().requires_grad_(grad_w)
        ref = rx * torch.rsqrt(rx.square().mean(-1, keepdim=True) + 1e-6) * rw
        out = compiled(x, w)
        close(out, ref, dtype)
        grad = torch.randn_like(out)
        targets = tuple(t for t in (x, w) if t.requires_grad)
        references = tuple(t for t in (rx, rw) if t.requires_grad)
        got = torch.autograd.grad(out, targets, grad)
        want = torch.autograd.grad(ref, references, grad.double())
        for a, b in zip(got, want):
            close(a, b, dtype)


def test_rms_norm_empty_backward_and_no_grad():
    def fn(x, w):
        return flag_gems.rms_norm(x, [37], w, 1e-6)

    x = torch.empty(0, 37, device=flag_gems.device, requires_grad=True)
    w = torch.ones(37, device=flag_gems.device, requires_grad=True)
    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(x, w)
    dx, dw = torch.autograd.grad(out.sum(), (x, w))
    assert dx.shape == x.shape
    torch.testing.assert_close(dw, torch.zeros_like(w))
    with torch.no_grad():
        out = compiled(torch.randn(2, 37, device=flag_gems.device), w)
        assert not out.requires_grad


def test_explicit_pt2_eager_keeps_activation_autograd():
    from flag_gems.pt2 import silu_and_mul_pointwise

    x, y = inputs(2, 37, torch.float32, "broadcast")
    output = silu_and_mul_pointwise(x, y)
    dx, dy = torch.autograd.grad(output.sum(), (x, y))
    assert dx.shape == x.shape and dy.shape == y.shape
