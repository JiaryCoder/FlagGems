---
title: torch.compile and PT2
weight: 82
---

# torch.compile and PT2

The covered public FlagGems functions keep their names, arguments and return
values. On NVIDIA, their public entry points select PT2 adapters while
`torch.compiler.is_compiling()` is true; ordinary calls keep the eager path.
Applications do not need a separate set of operator calls for compilation.

The adapters have been tested with PyTorch 2.11 and Triton 3.6 on NVIDIA.
They require `torch.library.triton_op` and `torch.library.wrap_triton`.
The presence of these APIs alone does not establish compatibility with every
Torch, Triton or vendor build.

## Coverage and input boundaries

| Public functions | Compiled behavior and boundaries | Autograd added by PT2 |
| --- | --- | --- |
| `silu_and_mul`, `gelu_and_mul` (`none` / `tanh`), `silu_and_mul_with_clamp` | FP16/BF16/FP32 tensor inputs; broadcast, mixed strides and dtype promotion use shared pointwise metadata. Output ranks 0–5 are prepared by default. | First-order x/y gradients; clamp limit remains non-differentiable. |
| `rms_norm` | One normalized dimension, matching x/weight dtype and device; strided inputs are materialized as in eager. FP16/BF16/FP32 are tested. | First-order x/weight gradients. |
| `fused_add_rms_norm` | One normalized dimension; preserves the public function's output and mutation behavior. | No new training support. |
| `apply_rotary_pos_emb` | In-place path with explicit `position_ids`; preserves query/key writes. | No new training support. |
| `topk_softmax`, `topk_softplus_sqrt`, `grouped_topk`, `moe_sum` | Existing routing/reduction adapters, including softplus/hash routing and output-buffer writes. | No new training support. |
| `mhc_pre`, `mhc_post`, `hc_head_fused_kernel` | Existing MHC inference adapters and their original kernels. | No new full MHC backward. |

This is not blanket compilation support for every FlagGems operator, every
`out` variant or every input accepted by eager. Second-order gradients and
vmap/JVP are outside this coverage. Unsupported dtypes and unprepared
pointwise specializations raise errors; the adapter does not automatically
switch them to eager inside a full graph.

```python
import torch
import flag_gems

x = torch.randn(8, 128, device="cuda", requires_grad=True)
y = torch.randn(128, device="cuda", requires_grad=True)
compiled = torch.compile(flag_gems.silu_and_mul, fullgraph=True, dynamic=True)
compiled(x, y).sum().backward()
```

## Preparation and dynamic inputs

On supported NVIDIA installations, importing FlagGems registers the PT2
operators and prepares the configured activation plans before graph capture.
Preparation generates/imports Python launch code and obtains the existing
Triton JIT objects. It does not run the activation kernels or cache an example
tensor. The default 288 metadata plans reuse 48 kernel materializations.

Broadcast shapes, strides, output allocation and launch sizes are inferred on
each call. Token counts and tensor addresses are not frozen by preparation.
Preparation has CPU startup and memory costs, including for eager-only users;
registration/materialization failures currently propagate from package import.

For a higher output rank, prepare the forward and backward families before
calling `torch.compile`. For example, six-dimensional SiLU inputs:

```python
from flag_gems import pt2

pt2.materialize_pointwise_family_plans(
    ("silu_and_mul", "silu_and_mul.backward"),
    ranks=(6,),
    dtypes=(torch.float32,),
)
```

`dtypes` selects the promoted output dtype of each family. The canonical
layout names are `contiguous_c` (physical dimension collapse) and `strided`;
`split_last_dim_c` remains an alias for `strided`. Use the preparation
factories to obtain immutable plans. Their stride-order diagnostic attributes
are retained for compatibility; actual launches use runtime strides.

## Adding or updating an operator

A new eager operator is not enrolled in PT2 automatically. It can compile
without an adapter if its implementation is already traceable. Otherwise,
provide an adapter using the original kernel and add public-entry routing
before non-traceable logging, caches or pointer wrappers.

For generated pointwise operators, register the family, verify tensor-only
input/output arity, and prepare the required structural plans before capture.
The current pointwise bridge accepts `LibEntry(JITFunction)`; a different
decorator chain needs an adapter that preserves its heuristics/autotuning.
Also check any operator-specific launch policy in the eager generator.
Register an autograd formula when exposing existing backward support, and
prepare its kernel family too. Updating an existing kernel's signature,
outputs or decorator chain requires reviewing the corresponding adapter.

Check eager parity, fullgraph execution, dynamic shapes, supported dtypes and
layouts, empty inputs, mutation/alias semantics, and any advertised gradients.
Deployment integrations should also check fresh-process cache loading and
CUDA Graph replay. FL's CachedOp prewarm selects a backend; it does not make
an unadapted kernel traceable, and CachedOps must exist before that prewarm.
