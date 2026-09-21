# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Transparent PT2 contracts for the existing FlagGems RMSNorm kernels.

The public FlagGems launchers mix tensor work with logging, contiguous
conversion, output allocation, hidden-size dispatch, and ``LibEntry`` cache
control.  Dynamo cannot trace those host-side Python responsibilities in a
fullgraph, and the pt2 contracts are the compile-mode path used under
torch.compile for these ops; eager callers keep the original launchers.

There is no generated or compile-only mathematical kernel here: eager and
compiled execution run the same kernel objects.

* the small standalone path wraps ``rms_norm_kernel.jit_function``;
* the large standalone path wraps the *same* Autotuner nested in
  ``rms_norm_loop_kernel`` so all configs and its ``key=[\"N\"]`` survive;
* the fused paths wrap the JITFunctions nested in the two original LibEntry
  objects and preserve their in-place writes to ``x`` and ``residual``.

The vLLM integration contract has one normalized dimension.  The standalone
path matches the existing eager launcher by materializing strided inputs
before the contiguous Triton ABI.  Hidden size chooses the small/loop family
under a Dynamo guard, whereas leading token dimensions remain symbolic.  The
fused path keeps its stricter contiguous contract so its mutation and alias
semantics cannot be changed by an implicit copy.
"""

from __future__ import annotations

import torch
import triton

from flag_gems.fused.fused_add_rms_norm import (
    fused_add_rms_norm as _eager_fused_add_rms_norm,
)
from flag_gems.fused.fused_add_rms_norm import (
    fused_add_rms_norm_kernel as _fused_small_entry,
)
from flag_gems.fused.fused_add_rms_norm import (
    fused_add_rms_norm_loop_kernel as _fused_loop_entry,
)
from flag_gems.ops.rms_norm import rms_norm as _eager_rms_norm
from flag_gems.ops.rms_norm import rms_norm_grad_dw_kernel as _grad_dw_entry
from flag_gems.ops.rms_norm import rms_norm_grad_dx_kernel as _grad_dx_entry
from flag_gems.ops.rms_norm import rms_norm_grad_dx_loop_kernel as _grad_dx_loop_entry
from flag_gems.ops.rms_norm import rms_norm_kernel as _rms_small_entry
from flag_gems.ops.rms_norm import rms_norm_loop_kernel as _rms_loop_entry
from flag_gems.pt2.manifest import CompileKind, CompileOpSpec, register_compile_spec

_HAS_TRITON_OP = hasattr(torch.library, "triton_op") and hasattr(
    torch.library, "wrap_triton"
)
_SMALL_HIDDEN_LIMIT = 4096
_FUSED_LOOP_BLOCK_SIZE = 1024

# Preserve object identity with the eager launch chain.  LibEntry itself is a
# Python dispatch/cache layer and is not understood by Dynamo.  For a direct
# LibEntry(JITFunction) chain we expose that exact nested JITFunction.  The
# standalone loop is LibEntry(Autotuner(JITFunction)), so the whole original
# Autotuner is wrapped rather than silently choosing one configuration.
RMS_NORM_SMALL_JIT = _rms_small_entry.jit_function
RMS_NORM_LOOP_AUTOTUNER = _rms_loop_entry.fn
FUSED_ADD_RMS_NORM_SMALL_JIT = _fused_small_entry.jit_function
FUSED_ADD_RMS_NORM_LOOP_JIT = _fused_loop_entry.jit_function
RMS_NORM_GRAD_DX_JIT = _grad_dx_entry.jit_function
RMS_NORM_GRAD_DX_LOOP_JIT = _grad_dx_loop_entry.jit_function
RMS_NORM_GRAD_DW_JIT = _grad_dw_entry.jit_function


def supports_pt2_rms_norm() -> bool:
    """Return whether transparent Triton operator APIs are available."""

    return _HAS_TRITON_OP


def _check_standard_contract(x: torch.Tensor, weight: torch.Tensor) -> tuple[int, int]:
    """Validate the one-dimensional, contiguous kernel ABI."""

    torch._check(x.ndim >= 1)
    torch._check(weight.ndim == 1)
    torch._check(x.dtype == weight.dtype)
    torch._check(x.device == weight.device)
    torch._check(x.is_contiguous())
    torch._check(weight.is_contiguous())
    hidden_size = weight.numel()
    torch._check(hidden_size > 0)
    torch._check(x.shape[-1] == hidden_size)
    num_rows = x.numel() // hidden_size
    return num_rows, hidden_size


def _check_fused_contract(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[int, int]:
    num_rows, hidden_size = _check_standard_contract(x, weight)
    torch._check(residual.ndim == x.ndim)
    for axis in range(x.ndim):
        torch._check(residual.shape[axis] == x.shape[axis])
    torch._check(residual.dtype == x.dtype)
    torch._check(residual.device == x.device)
    torch._check(residual.is_contiguous())
    return num_rows, hidden_size


if _HAS_TRITON_OP:

    def _rms_norm_forward(x, weight, eps):
        num_rows, hidden_size = _check_standard_contract(x, weight)
        out = torch.empty_like(x)
        inv_rms = torch.empty((num_rows,), device=x.device, dtype=torch.float32)
        if num_rows == 0:
            return out, inv_rms
        if hidden_size <= _SMALL_HIDDEN_LIMIT:
            block_size = triton.next_power_of_2(hidden_size)
            torch.library.wrap_triton(RMS_NORM_SMALL_JIT)[(num_rows,)](
                out,
                inv_rms,
                x,
                weight,
                hidden_size,
                1,
                hidden_size,
                1,
                hidden_size,
                eps,
                block_size,
            )
        else:
            torch.library.wrap_triton(RMS_NORM_LOOP_AUTOTUNER)[(num_rows,)](
                out, inv_rms, x, weight, hidden_size, eps
            )
        return out, inv_rms

    @torch.library.triton_op("flag_gems_pt2::rms_norm", mutates_args={})
    def _rms_norm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return _rms_norm_forward(x, weight, eps)[0]

    @torch.library.triton_op("flag_gems_pt2::rms_norm_training", mutates_args={})
    def _rms_norm_training_op(
        x: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Auxiliary output is private to the adapter. It is saved by autograd,
        # while the public FlagGems API continues to return only the result.
        return _rms_norm_forward(x, weight, eps)

    @torch.library.triton_op("flag_gems_pt2::rms_norm_backward", mutates_args={})
    def _rms_norm_backward_op(
        dy: torch.Tensor,
        x: torch.Tensor,
        inv_rms: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_rows, hidden_size = _check_standard_contract(x, weight)
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        if num_rows == 0:
            return dx, torch.zeros_like(weight)
        if hidden_size <= _SMALL_HIDDEN_LIMIT:
            block_size = triton.next_power_of_2(hidden_size)
            kernel = RMS_NORM_GRAD_DX_JIT
        else:
            block_size = _FUSED_LOOP_BLOCK_SIZE
            kernel = RMS_NORM_GRAD_DX_LOOP_JIT
        torch.library.wrap_triton(kernel)[(num_rows,)](
            x,
            dy,
            inv_rms,
            dx,
            weight,
            hidden_size,
            1,
            hidden_size,
            1,
            hidden_size,
            eps,
            block_size,
        )
        row_block_size = 16
        col_block_size = 256
        row_blocks = triton.cdiv(num_rows, row_block_size)
        col_blocks = triton.cdiv(hidden_size, col_block_size)
        partial = torch.empty(
            (row_blocks, hidden_size), device=x.device, dtype=torch.float32
        )
        torch.library.wrap_triton(RMS_NORM_GRAD_DW_JIT)[(row_blocks, col_blocks)](
            x,
            dy,
            inv_rms,
            partial,
            hidden_size,
            1,
            hidden_size,
            1,
            num_rows,
            hidden_size,
            row_block_size,
            col_block_size,
        )
        dw = partial.sum(dim=0, dtype=torch.float32).to(weight.dtype)
        return dx, dw

    def _setup_rms_context(ctx, inputs, output):
        x, weight, eps = inputs
        _, inv_rms = output
        ctx.save_for_backward(x, weight, inv_rms)
        ctx.eps = eps
        ctx.mark_non_differentiable(inv_rms)

    def _rms_backward(ctx, grad_output, grad_inv_rms):
        x, weight, inv_rms = ctx.saved_tensors
        dx, dw = _rms_norm_backward_op(grad_output, x, inv_rms, weight, ctx.eps)
        return dx, dw, None

    _rms_norm_training_op.register_autograd(
        _rms_backward, setup_context=_setup_rms_context
    )

    @torch.library.triton_op(
        "flag_gems_pt2::fused_add_rms_norm",
        mutates_args={"x", "residual"},
    )
    def _fused_add_rms_norm_op(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> None:
        num_rows, hidden_size = _check_fused_contract(x, residual, weight)
        if num_rows == 0:
            return
        if hidden_size <= _SMALL_HIDDEN_LIMIT:
            block_size = triton.next_power_of_2(hidden_size)
            torch.library.wrap_triton(FUSED_ADD_RMS_NORM_SMALL_JIT)[(num_rows,)](
                x,
                residual,
                weight,
                hidden_size,
                1,
                hidden_size,
                1,
                hidden_size,
                eps,
                block_size,
            )
        else:
            torch.library.wrap_triton(FUSED_ADD_RMS_NORM_LOOP_JIT)[(num_rows,)](
                x,
                residual,
                weight,
                hidden_size,
                eps,
                _FUSED_LOOP_BLOCK_SIZE,
            )

else:
    _rms_norm_op = None
    _rms_norm_training_op = None
    _rms_norm_backward_op = None
    _fused_add_rms_norm_op = None


_RMS_REQUIRES = (
    "torch.library.triton_op",
    "torch.library.wrap_triton",
    "standalone inputs materialized to a contiguous one-dimensional shape",
)

_FUSED_RMS_REQUIRES = (
    "torch.library.triton_op",
    "torch.library.wrap_triton",
    "contiguous one-dimensional normalized shape for mutation safety",
)

RMS_NORM_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::rms_norm",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.ops.rms_norm.{rms_norm_kernel.jit_function,"
            "rms_norm_loop_kernel.fn}"
        ),
        dynamic_dims=("leading_token_dims",),
        requires=_RMS_REQUIRES,
    )
)

RMS_NORM_TRAINING_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::rms_norm_training",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=RMS_NORM_SPEC.source_kernel,
        dynamic_dims=("leading_token_dims",),
        requires=(*_RMS_REQUIRES, "torch.library.register_autograd"),
    )
)

RMS_NORM_BACKWARD_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::rms_norm_backward",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.ops.rms_norm.{rms_norm_grad_dx_kernel,"
            "rms_norm_grad_dx_loop_kernel,rms_norm_grad_dw_kernel}.jit_function"
        ),
        dynamic_dims=("leading_token_dims",),
        requires=_RMS_REQUIRES,
    )
)

FUSED_ADD_RMS_NORM_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::fused_add_rms_norm",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.fused.fused_add_rms_norm."
            "{fused_add_rms_norm_kernel,fused_add_rms_norm_loop_kernel}."
            "jit_function"
        ),
        mutates_args=("x", "residual"),
        dynamic_dims=("leading_token_dims",),
        requires=_FUSED_RMS_REQUIRES,
    )
)


def _missing_triton_op() -> RuntimeError:
    return RuntimeError(
        "This Torch build lacks triton_op/wrap_triton; the transparent "
        "FlagGems RMSNorm PT2 contract is unavailable"
    )


def rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the original FlagGems RMSNorm family in eager or compiled mode."""

    if torch.compiler.is_compiling():
        if _rms_norm_op is None or _fused_add_rms_norm_op is None:
            raise _missing_triton_op()
        if residual is None:
            # Match the original eager launcher.  Qwen-style Q/K projections
            # are split views whose last dimension is dense but whose row
            # stride still spans the complete QKV allocation.  Record the
            # materialization in the outer FX graph, then keep the triton_op's
            # contiguous checks as the fail-closed kernel ABI.
            x = x.contiguous()
            weight = weight.contiguous()
            if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
                return _rms_norm_training_op(x, weight, eps)[0]
            return _rms_norm_op(x, weight, eps)
        _fused_add_rms_norm_op(x, residual, weight, eps)
        return x, residual

    normalized_shape = list(weight.size())
    if residual is None:
        return _eager_rms_norm(x, normalized_shape, weight, eps)
    return _eager_fused_add_rms_norm(x, residual, normalized_shape, weight, eps)


__all__ = [
    "FUSED_ADD_RMS_NORM_LOOP_JIT",
    "FUSED_ADD_RMS_NORM_SMALL_JIT",
    "FUSED_ADD_RMS_NORM_SPEC",
    "rms_norm",
    "RMS_NORM_BACKWARD_SPEC",
    "RMS_NORM_LOOP_AUTOTUNER",
    "RMS_NORM_SMALL_JIT",
    "RMS_NORM_SPEC",
    "RMS_NORM_TRAINING_SPEC",
    "supports_pt2_rms_norm",
]
