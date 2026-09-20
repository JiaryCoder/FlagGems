# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Transparent PT2 adapters for generated ``pointwise_dynamic`` kernels.

The Python control plane and tensor execution plane are deliberately split:

* a :class:`PointwiseFamilySpec` binds one existing FlagGems scalar/kernel
  generator to structural input metadata;
* :func:`materialize_pointwise_plan` runs rank code generation and imports the
  generated module before Dynamo starts;
* eager execution keeps the generated wrapper/``LibEntry`` path;
* compiled execution launches the exact same generated ``JITFunction`` through
  ``torch.library.triton_op`` and ``torch.library.wrap_triton``.

Plans contain no Tensor, data pointer, output allocation, token count, grid, or
shape value. Rank, dtype, layout family, and guarded GELU approximation select
an immutable plan; concrete shapes and launch parameters remain symbolic or
runtime-specialized after the Dynamo boundary.

Forward and first-order backward reuse the original pointwise kernels and
their shared metadata preparation. Broadcasting, dtype promotion and physical
dimension collapse are inferred for each invocation, not saved from a warmup
Tensor. Only immutable code-generation state is retained by a plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import triton
from flag_gems.fused.gelu_and_mul import gelu_and_mul as _eager_gelu
from flag_gems.fused.gelu_and_mul import (
    gelu_none_and_mul_grad_kernel as _gelu_none_grad_source,
)
from flag_gems.fused.gelu_and_mul import gelu_none_and_mul_kernel as _gelu_none_source
from flag_gems.fused.gelu_and_mul import (
    gelu_tanh_and_mul_grad_kernel as _gelu_tanh_grad_source,
)
from flag_gems.fused.gelu_and_mul import gelu_tanh_and_mul_kernel as _gelu_tanh_source
from flag_gems.fused.silu_and_mul import silu_and_mul as _eager_silu
from flag_gems.fused.silu_and_mul import silu_and_mul_grad_kernel as _silu_grad_source
from flag_gems.fused.silu_and_mul import silu_and_mul_kernel as _silu_source
from flag_gems.fused.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp as _eager_silu_clamp,
)
from flag_gems.fused.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp_grad_kernel as _silu_clamp_grad_source,
)
from flag_gems.fused.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp_kernel as _silu_clamp_source,
)
from flag_gems.pt2.manifest import CompileKind, CompileOpSpec, register_compile_spec
from flag_gems.utils.codegen_config_utils import get_heuristics_for_num_warps_fn
from flag_gems.utils.pointwise_dynamic import (
    PointwiseDynamicFunction,
    PointwiseKernelMaterialization,
    _balanced_grid_partition,
)
from flag_gems.utils.shape_utils import (
    broadcasted_stride,
    heuristics_for_tile_size,
    stride_order,
)

_HAS_TRITON_OP = hasattr(torch.library, "triton_op") and hasattr(
    torch.library, "wrap_triton"
)
_SUPPORTED_LAYOUTS = ("contiguous_c", "strided")
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class PointwiseFamilySpec:
    """Graph-independent description of one existing generated kernel family."""

    name: str
    source_pointwise: PointwiseDynamicFunction
    num_inputs: int
    scalar_input_indices: tuple[int, ...] = ()
    num_outputs: int = 1

    @property
    def primary_input_indices(self) -> tuple[int, ...]:
        return tuple(
            i for i in range(self.num_inputs) if i not in self.scalar_input_indices
        )


@dataclass(frozen=True)
class PointwisePlan:
    """Tensor-free specialization plan for one generated kernel family."""

    token: int
    op_name: str
    ndim: int
    kernel_ndim: int
    dtype: torch.dtype
    layout_class: str
    num_inputs: int
    scalar_input_indices: tuple[int, ...]
    materialization: PointwiseKernelMaterialization
    num_warps_policy: Callable[[int], int]
    num_outputs: int = 1

    @property
    def jit_function(self):
        return self.materialization.jit_function

    # Retain the old diagnostic attributes without storing a stride order in
    # each plan. Execution derives its order from the actual runtime strides.
    @property
    def tensor_stride_order(self) -> tuple[int, ...]:
        return tuple(reversed(range(self.kernel_ndim)))

    @property
    def scalar_stride_order(self) -> tuple[int, ...]:
        return tuple(range(self.kernel_ndim))

    @property
    def primary_input_indices(self) -> tuple[int, ...]:
        return tuple(
            i for i in range(self.num_inputs) if i not in self.scalar_input_indices
        )


_FAMILIES: dict[str, PointwiseFamilySpec] = {}
_PLANS_BY_KEY: dict[tuple[str, int, torch.dtype, str], PointwisePlan] = {}
_PLANS_BY_TOKEN: dict[int, PointwisePlan] = {}
_NEXT_PLAN_TOKEN = 0


def register_pointwise_family(spec: PointwiseFamilySpec) -> PointwiseFamilySpec:
    """Register structural facts without generating or launching a kernel."""

    if torch.compiler.is_compiling():
        raise RuntimeError("pointwise family registration is forbidden inside Dynamo")
    if not isinstance(spec.source_pointwise, PointwiseDynamicFunction):
        raise TypeError(f"{spec.name!r} is not a PointwiseDynamicFunction")
    if spec.num_inputs < 1:
        raise ValueError("pointwise family must have at least one tensor input")

    scalar_indices = tuple(sorted(set(spec.scalar_input_indices)))
    if scalar_indices != spec.scalar_input_indices:
        raise ValueError("scalar_input_indices must be sorted and unique")
    if any(i < 0 or i >= spec.num_inputs for i in scalar_indices):
        raise ValueError(f"invalid scalar input index for {spec.name!r}")
    if len(scalar_indices) == spec.num_inputs:
        raise ValueError("pointwise family requires a non-scalar primary tensor")

    schema = spec.source_pointwise.fx
    if (
        schema.num_input_tensors() != spec.num_inputs
        or schema.num_non_tensor_args() != 0
        or schema.num_output_tensors() != spec.num_outputs
        or spec.num_outputs < 1
    ):
        raise RuntimeError(
            f"Unsupported PT2 pointwise schema for {spec.name!r}: {schema}. "
            "This adapter requires tensor-only inputs and matching output arity."
        )

    previous = _FAMILIES.get(spec.name)
    if previous is not None and previous != spec:
        raise RuntimeError(f"Conflicting pointwise family {spec.name!r}")
    _FAMILIES[spec.name] = spec
    return spec


SILU_AND_MUL_FAMILY = register_pointwise_family(
    PointwiseFamilySpec("silu_and_mul", _silu_source, num_inputs=2)
)
GELU_NONE_AND_MUL_FAMILY = register_pointwise_family(
    PointwiseFamilySpec("gelu_and_mul.none", _gelu_none_source, num_inputs=2)
)
GELU_TANH_AND_MUL_FAMILY = register_pointwise_family(
    PointwiseFamilySpec("gelu_and_mul.tanh", _gelu_tanh_source, num_inputs=2)
)
SILU_AND_MUL_WITH_CLAMP_FAMILY = register_pointwise_family(
    PointwiseFamilySpec(
        "silu_and_mul_with_clamp",
        _silu_clamp_source,
        num_inputs=3,
        scalar_input_indices=(2,),
    )
)

ACTIVATION_POINTWISE_FAMILIES = (
    SILU_AND_MUL_FAMILY.name,
    GELU_NONE_AND_MUL_FAMILY.name,
    GELU_TANH_AND_MUL_FAMILY.name,
    SILU_AND_MUL_WITH_CLAMP_FAMILY.name,
)

for _forward_name, _source, _num_inputs, _scalars in (
    (SILU_AND_MUL_FAMILY.name, _silu_grad_source, 3, ()),
    (GELU_NONE_AND_MUL_FAMILY.name, _gelu_none_grad_source, 3, ()),
    (GELU_TANH_AND_MUL_FAMILY.name, _gelu_tanh_grad_source, 3, ()),
    (SILU_AND_MUL_WITH_CLAMP_FAMILY.name, _silu_clamp_grad_source, 4, (3,)),
):
    register_pointwise_family(
        PointwiseFamilySpec(
            _forward_name + ".backward",
            _source,
            num_inputs=_num_inputs,
            scalar_input_indices=_scalars,
            num_outputs=2,
        )
    )

ACTIVATION_BACKWARD_POINTWISE_FAMILIES = tuple(
    name + ".backward" for name in ACTIVATION_POINTWISE_FAMILIES
)


def _family(name: str) -> PointwiseFamilySpec:
    try:
        return _FAMILIES[name]
    except KeyError as exc:
        raise KeyError(f"unknown pointwise family: {name!r}") from exc


def materialize_pointwise_plan(
    op_name: str,
    *,
    ndim: int = 2,
    dtype: torch.dtype = torch.bfloat16,
    layout_class: str = "split_last_dim_c",
) -> PointwisePlan:
    """Materialize structural codegen state outside a Dynamo graph."""

    global _NEXT_PLAN_TOKEN

    if torch.compiler.is_compiling():
        raise RuntimeError("pointwise plan materialization is forbidden inside Dynamo")
    if not isinstance(ndim, int) or ndim < 0:
        raise ValueError(f"expected a nonnegative rank, got {ndim!r}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported pointwise dtype: {dtype}")
    # The old split-view plan already uses the general strided kernel ABI.
    # Preserve explicit preparation through that public spelling as well.
    if layout_class == "split_last_dim_c":
        layout_class = "strided"
    if layout_class not in _SUPPORTED_LAYOUTS:
        raise ValueError(f"unsupported pointwise layout class: {layout_class!r}")

    family = _family(op_name)
    key = (family.name, ndim, dtype, layout_class)
    cached = _PLANS_BY_KEY.get(key)
    if cached is not None:
        return cached

    # The call-site metadata decides whether physical dimension collapse is
    # legal. A scalar broadcast selects a strided plan, even if the main input
    # is contiguous. The generated kernel is still the original eager kernel.
    kernel_ndim = 1 if layout_class == "contiguous_c" else ndim
    generated = family.source_pointwise.materialize(kernel_ndim)
    if generated.runtime_chain != ("LibEntry", "JITFunction"):
        raise RuntimeError(
            "Transparent pointwise PT2 only supports the generated "
            "LibEntry(JITFunction) chain; refusing to drop tuner/heuristic "
            f"semantics from {generated.runtime_chain!r}"
        )

    plan = PointwisePlan(
        token=_NEXT_PLAN_TOKEN,
        op_name=family.name,
        ndim=ndim,
        kernel_ndim=kernel_ndim,
        dtype=dtype,
        layout_class=layout_class,
        num_inputs=family.num_inputs,
        scalar_input_indices=family.scalar_input_indices,
        materialization=generated,
        num_warps_policy=get_heuristics_for_num_warps_fn(),
        num_outputs=family.num_outputs,
    )
    _NEXT_PLAN_TOKEN += 1
    _PLANS_BY_KEY[key] = plan
    _PLANS_BY_TOKEN[plan.token] = plan
    return plan


def materialize_pointwise_family_plans(
    op_names: Iterable[str],
    *,
    ranks: Iterable[int] = (2,),
    dtypes: Iterable[torch.dtype] = _SUPPORTED_DTYPES,
    layout_classes: Iterable[str] = _SUPPORTED_LAYOUTS,
) -> tuple[PointwisePlan, ...]:
    """Freeze a Cartesian product of structural plans before Dynamo capture."""

    # Each inner iterable must be reusable across families and ranks.
    ranks = tuple(ranks)
    dtypes = tuple(dtypes)
    layout_classes = tuple(layout_classes)
    plans = []
    for op_name in op_names:
        for ndim in ranks:
            for dtype in dtypes:
                for layout_class in layout_classes:
                    plans.append(
                        materialize_pointwise_plan(
                            op_name,
                            ndim=ndim,
                            dtype=dtype,
                            layout_class=layout_class,
                        )
                    )
    return tuple(plans)


def materialize_silu_and_mul_plan(**kwargs) -> PointwisePlan:
    return materialize_pointwise_plan(SILU_AND_MUL_FAMILY.name, **kwargs)


def _gelu_family(approximate: str) -> PointwiseFamilySpec:
    if approximate == "none":
        return GELU_NONE_AND_MUL_FAMILY
    if approximate == "tanh":
        return GELU_TANH_AND_MUL_FAMILY
    raise ValueError(f"Invalid approximate value: {approximate}")


def materialize_gelu_and_mul_plan(
    *, approximate: str = "none", **kwargs
) -> PointwisePlan:
    return materialize_pointwise_plan(_gelu_family(approximate).name, **kwargs)


def materialize_silu_and_mul_with_clamp_plan(**kwargs) -> PointwisePlan:
    return materialize_pointwise_plan(SILU_AND_MUL_WITH_CLAMP_FAMILY.name, **kwargs)


def materialized_pointwise_plans(
    op_name: str | None = None,
) -> tuple[PointwisePlan, ...]:
    """Return a stable diagnostic snapshot without exposing mutable caches."""

    plans = tuple(_PLANS_BY_TOKEN[token] for token in sorted(_PLANS_BY_TOKEN))
    if op_name is None:
        return plans
    return tuple(plan for plan in plans if plan.op_name == op_name)


def _resolve_plan_token(op_name: str, inputs: tuple[torch.Tensor, ...]) -> int:
    """Resolve already-materialized metadata; never codegen on a cache miss."""

    family = _family(op_name)
    if len(inputs) != family.num_inputs:
        raise RuntimeError(
            f"{family.name!r} requires {family.num_inputs} inputs, got {len(inputs)}"
        )
    for tensor in inputs:
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported PT2 pointwise dtype {tensor.dtype} for {family.name!r}; "
                "supported dtypes are float16, bfloat16 and float32"
            )
    _, shape, _, dtypes, collapsed = family.source_pointwise.prepare_metadata(*inputs)
    key = (
        family.name,
        len(shape),
        dtypes[0],
        "contiguous_c" if collapsed else "strided",
    )
    plan = _PLANS_BY_KEY.get(key)
    if plan is None:
        raise RuntimeError(
            "No materialized pointwise plan for "
            f"op={key[0]!r}, rank={key[1]}, dtype={key[2]}, layout={key[3]!r}. "
            "Materialize it outside Dynamo before compiling this specialization."
        )
    return plan.token


def _check_contract(plan: PointwisePlan, inputs: tuple[torch.Tensor, ...]) -> None:
    torch._check(len(inputs) == plan.num_inputs)
    reference = inputs[plan.primary_input_indices[0]]
    for index, tensor in enumerate(inputs):
        torch._check(tensor.dtype in _SUPPORTED_DTYPES)
        torch._check(tensor.device == reference.device)
        if index in plan.scalar_input_indices:
            torch._check(tensor.numel() == 1)


def _task_shape(plan: PointwisePlan, out: torch.Tensor):
    if plan.kernel_ndim == plan.ndim:
        return out.shape
    torch._check(plan.kernel_ndim == 1)
    return (out.numel(),)


def _runtime_strides(plan: PointwisePlan, tensor: torch.Tensor, task_shape):
    if plan.layout_class == "contiguous_c":
        return (1,)
    return broadcasted_stride(tensor.shape, tensor.stride(), task_shape)


def _partition(plan: PointwisePlan, out: torch.Tensor):
    """Use the eager wrapper's tile and balanced-grid helpers."""

    shape = _task_shape(plan, out)
    num_tasks = out.numel()
    if num_tasks == 0:
        return None
    if plan.materialization.prefer_block_pointer:
        # Eager prepare_args selects a non-block-pointer ABI for larger tensors.
        # That ABI must be materialized outside Dynamo as a different plan.
        torch._check(num_tasks <= 2_147_483_647)

    tile_shape = (num_tasks,) if plan.materialization.prefer_1d_tile else shape
    tile_sizes = heuristics_for_tile_size(
        plan.materialization.max_tile_size, *tile_shape
    )
    num_tiles = math.prod(
        triton.cdiv(size, tile_size) for size, tile_size in zip(tile_shape, tile_sizes)
    )

    if plan.materialization.balance_grid:
        num_ctas, tiles_per_cta = _balanced_grid_partition(
            num_tiles, plan.materialization.max_grid_size[0]
        )
    else:
        num_ctas = min(plan.materialization.max_grid_size[0], num_tiles)
        tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    num_warps = plan.num_warps_policy(math.prod(tile_sizes))
    return (
        (num_ctas, 1, 1),
        num_tasks,
        tiles_per_cta,
        tile_sizes,
        tiles_per_cta == 1,
        num_warps,
    )


def _launch_plan(
    plan: PointwisePlan,
    inputs: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
    launch,
) -> None:
    grid, num_tasks, tiles_per_cta, tiles, one_tile, num_warps = launch
    wrapped = torch.library.wrap_triton(plan.jit_function)
    args = [*inputs, *outputs]
    task_shape = _task_shape(plan, outputs[0])
    use_block_pointer = (
        plan.materialization.prefer_block_pointer
        and not plan.materialization.prefer_1d_tile
    )
    for tensor in (*inputs, *outputs):
        strides = _runtime_strides(plan, tensor, task_shape)
        args.extend(strides)
        if use_block_pointer:
            args.extend(stride_order(strides))

    args.extend(task_shape)
    args.append(num_tasks)

    kwargs = {
        "tiles_per_cta": tiles_per_cta,
        "one_tile_per_cta": one_tile,
        "num_warps": num_warps,
    }
    if plan.materialization.prefer_1d_tile:
        kwargs["tile_size"] = tiles[0]
    else:
        for axis, tile_size in enumerate(tiles):
            kwargs[f"tile_size{axis}"] = tile_size
    wrapped[grid](*args, **kwargs)


def _execute_plan(
    inputs: tuple[torch.Tensor, ...], plan_token: int
) -> tuple[torch.Tensor, ...]:
    plan = _PLANS_BY_TOKEN[plan_token]
    _check_contract(plan, inputs)
    task_shape, shape, reference, dtypes, collapsed = _family(
        plan.op_name
    ).source_pointwise.prepare_metadata(*inputs)
    torch._check(len(shape) == plan.ndim)
    torch._check(len(task_shape) == plan.kernel_ndim)
    torch._check(collapsed == (plan.layout_class == "contiguous_c"))
    torch._check(len(dtypes) == plan.num_outputs)
    torch._check(dtypes[0] == plan.dtype)
    outputs = tuple(
        (
            torch.empty_like(reference, dtype=dtype)
            if reference is not None
            else torch.empty(shape, dtype=dtype, device=inputs[0].device)
        )
        for dtype in dtypes
    )
    launch = _partition(plan, outputs[0])
    if launch is not None:
        _launch_plan(plan, inputs, outputs, launch)
    return outputs


if _HAS_TRITON_OP:

    @torch.library.triton_op("flag_gems_pt2::silu_and_mul_pointwise", mutates_args={})
    def _silu_and_mul_pointwise_op(
        gate: torch.Tensor, up: torch.Tensor, plan_token: int
    ) -> torch.Tensor:
        return _execute_plan((gate, up), plan_token)[0]

    @torch.library.triton_op("flag_gems_pt2::gelu_and_mul_pointwise", mutates_args={})
    def _gelu_and_mul_pointwise_op(
        gate: torch.Tensor, up: torch.Tensor, plan_token: int
    ) -> torch.Tensor:
        return _execute_plan((gate, up), plan_token)[0]

    @torch.library.triton_op(
        "flag_gems_pt2::silu_and_mul_with_clamp_pointwise", mutates_args={}
    )
    def _silu_and_mul_with_clamp_pointwise_op(
        gate: torch.Tensor,
        up: torch.Tensor,
        limit: torch.Tensor,
        plan_token: int,
    ) -> torch.Tensor:
        return _execute_plan((gate, up, limit), plan_token)[0]

    @torch.library.triton_op("flag_gems_pt2::activation_backward", mutates_args={})
    def _activation_backward_op(
        inputs: list[torch.Tensor], plan_token: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dx, dy = _execute_plan(tuple(inputs), plan_token)
        return dx, dy

    def _setup_activation_context(ctx, inputs, output):
        *tensors, plan_token = inputs
        ctx.save_for_backward(*tensors)
        ctx.backward_family = _PLANS_BY_TOKEN[plan_token].op_name + ".backward"

    def _activation_backward(ctx, grad_output):
        x, y, *extra = ctx.saved_tensors
        inputs = (x, y, grad_output, *extra)
        token = _resolve_plan_token(ctx.backward_family, inputs)
        dx, dy = _activation_backward_op(list(inputs), token)
        # The source kernels are elementwise. Broadcast axes belong to the
        # caller's input tensors and must be reduced before returning gradients.
        dx = dx.sum_to_size(x.shape).to(x.dtype)
        dy = dy.sum_to_size(y.shape).to(y.dtype)
        return (dx, dy, *((None,) * len(extra)), None)

    for _op in (
        _silu_and_mul_pointwise_op,
        _gelu_and_mul_pointwise_op,
        _silu_and_mul_with_clamp_pointwise_op,
    ):
        _op.register_autograd(
            _activation_backward, setup_context=_setup_activation_context
        )

else:
    _silu_and_mul_pointwise_op = None
    _gelu_and_mul_pointwise_op = None
    _silu_and_mul_with_clamp_pointwise_op = None
    _activation_backward_op = None


_POINTWISE_REQUIRES = (
    "PointwiseDynamicFunction.materialize",
    "torch.library.triton_op",
    "torch.library.wrap_triton",
)

SILU_AND_MUL_POINTWISE_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::silu_and_mul_pointwise",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.fused.silu_and_mul.silu_and_mul_kernel"
            ".materialize(ndim).jit_function"
        ),
        dynamic_dims=("n_tokens",),
        requires=_POINTWISE_REQUIRES,
    )
)

GELU_AND_MUL_POINTWISE_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::gelu_and_mul_pointwise",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "guarded approximate selects flag_gems.fused.gelu_and_mul."
            "gelu_{none,tanh}_and_mul_kernel.materialize(ndim).jit_function"
        ),
        dynamic_dims=("n_tokens",),
        requires=_POINTWISE_REQUIRES,
    )
)

SILU_AND_MUL_WITH_CLAMP_POINTWISE_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::silu_and_mul_with_clamp_pointwise",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "flag_gems.fused.silu_and_mul_with_clamp."
            "silu_and_mul_with_clamp_kernel.materialize(ndim).jit_function"
        ),
        dynamic_dims=("n_tokens",),
        requires=_POINTWISE_REQUIRES,
    )
)

ACTIVATION_BACKWARD_POINTWISE_SPEC = register_compile_spec(
    CompileOpSpec(
        op_name="flag_gems_pt2::activation_backward",
        kind=CompileKind.TRITON_TRACEABLE,
        source_kernel=(
            "original silu_and_mul, gelu_{none,tanh}_and_mul and "
            "silu_and_mul_with_clamp gradient kernels.materialize(ndim).jit_function"
        ),
        dynamic_dims=("n_tokens",),
        requires=_POINTWISE_REQUIRES,
    )
)


def _missing_triton_op() -> RuntimeError:
    return RuntimeError(
        "This Torch build lacks triton_op/wrap_triton; the transparent "
        "pointwise PT2 contract is unavailable"
    )


def silu_and_mul_pointwise(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Run the original generated SiLU-and-multiply kernel."""

    if torch.compiler.is_compiling():
        if _silu_and_mul_pointwise_op is None:
            raise _missing_triton_op()
        plan_token = _resolve_plan_token(SILU_AND_MUL_FAMILY.name, (gate, up))
        return _silu_and_mul_pointwise_op(gate, up, plan_token)
    return _eager_silu(gate, up)


def gelu_and_mul_pointwise(
    gate: torch.Tensor, up: torch.Tensor, approximate: str = "none"
) -> torch.Tensor:
    """Run the guarded original GELU-none or GELU-tanh generated kernel."""

    family = _gelu_family(approximate)
    if torch.compiler.is_compiling():
        if _gelu_and_mul_pointwise_op is None:
            raise _missing_triton_op()
        plan_token = _resolve_plan_token(family.name, (gate, up))
        return _gelu_and_mul_pointwise_op(gate, up, plan_token)
    return _eager_gelu(gate, up, approximate)


def silu_and_mul_with_clamp_pointwise(
    gate: torch.Tensor, up: torch.Tensor, limit: torch.Tensor
) -> torch.Tensor:
    """Run the original generated clamped SiLU-and-multiply kernel."""

    inputs = (gate, up, limit)
    if torch.compiler.is_compiling():
        if _silu_and_mul_with_clamp_pointwise_op is None:
            raise _missing_triton_op()
        plan_token = _resolve_plan_token(SILU_AND_MUL_WITH_CLAMP_FAMILY.name, inputs)
        return _silu_and_mul_with_clamp_pointwise_op(gate, up, limit, plan_token)
    return _eager_silu_clamp(gate, up, limit)


__all__ = [
    "ACTIVATION_BACKWARD_POINTWISE_FAMILIES",
    "ACTIVATION_BACKWARD_POINTWISE_SPEC",
    "ACTIVATION_POINTWISE_FAMILIES",
    "gelu_and_mul_pointwise",
    "GELU_AND_MUL_POINTWISE_SPEC",
    "GELU_NONE_AND_MUL_FAMILY",
    "GELU_TANH_AND_MUL_FAMILY",
    "materialize_gelu_and_mul_plan",
    "materialize_pointwise_family_plans",
    "materialize_pointwise_plan",
    "materialize_silu_and_mul_plan",
    "materialize_silu_and_mul_with_clamp_plan",
    "materialized_pointwise_plans",
    "PointwiseFamilySpec",
    "PointwisePlan",
    "register_pointwise_family",
    "SILU_AND_MUL_FAMILY",
    "silu_and_mul_pointwise",
    "SILU_AND_MUL_POINTWISE_SPEC",
    "SILU_AND_MUL_WITH_CLAMP_FAMILY",
    "silu_and_mul_with_clamp_pointwise",
    "SILU_AND_MUL_WITH_CLAMP_POINTWISE_SPEC",
]
