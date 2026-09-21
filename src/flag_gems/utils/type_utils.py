# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes


@torch.compiler.assume_constant_result
def _metadata_elementwise_dtypes(metadata, promotion_kind, default_dtype):
    """Evaluate PyTorch's promotion rules on immutable, guarded metadata.

    No runtime Tensor, symbolic size or data pointer may enter this function.
    Rank only distinguishes scalar tensors from tensors with dimensions.
    Reading the default dtype at the call site keeps it visible to Dynamo.
    """
    assert torch.get_default_dtype() == default_dtype
    operands = []
    for dtype, scalar_type, is_scalar_tensor in metadata:
        if dtype is not None:
            shape = () if is_scalar_tensor else (0,)
            operands.append(torch.empty(shape, dtype=dtype, device="meta"))
        elif scalar_type is not None:
            operands.append(scalar_type(0))
    return elementwise_dtypes(*operands, type_promotion_kind=promotion_kind)


def type_promotion(*args, type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND):
    if torch.compiler.is_compiling():
        metadata = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                metadata.append((arg.dtype, None, arg.ndim == 0))
            elif arg is None:
                metadata.append((None, None, False))
            elif isinstance(arg, bool):
                metadata.append((None, bool, False))
            elif isinstance(arg, int):
                metadata.append((None, int, False))
            elif isinstance(arg, float):
                metadata.append((None, float, False))
            elif isinstance(arg, complex):
                metadata.append((None, complex, False))
            else:
                raise TypeError("Unsupported scalar type for compiled type promotion")
        return _metadata_elementwise_dtypes(
            tuple(metadata), type_promotion, torch.get_default_dtype()
        )
    computation_dtype, result_dtype = elementwise_dtypes(
        *args,
        type_promotion_kind=type_promotion,
    )
    return computation_dtype, result_dtype


_accumulator_dtype_map = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.complex32: torch.complex64,
}


def get_accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    return _accumulator_dtype_map.get(dtype, dtype)
