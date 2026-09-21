# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pure-Python PyTorch compiler integration for existing FlagGems kernels."""

from flag_gems.pt2.fused_moe import (
    MOE_SUM_SPEC,
    TOPK_SOFTMAX_SPEC,
    moe_sum,
    supports_pt2_fused_moe,
    topk_softmax,
)
from flag_gems.pt2.manifest import (
    CompileKind,
    CompileOpSpec,
    get_compile_manifest,
    get_compile_spec,
)
from flag_gems.pt2.mhc import (
    HC_HEAD_FUSED_SPEC,
    MHC_POST_SPEC,
    MHC_PRE_SPEC,
    hc_head_fused_kernel,
    mhc_post,
    mhc_pre,
)
from flag_gems.pt2.moe_routing import (
    GROUPED_TOPK_SPEC,
    TOPK_HASH_SOFTPLUS_SQRT_SPEC,
    TOPK_SOFTPLUS_SQRT_SPEC,
    grouped_topk,
    supports_pt2_moe_routing,
    topk_softplus_sqrt,
    uses_common_moe_routing_kernels,
)
from flag_gems.pt2.pointwise_dynamic import (
    ACTIVATION_BACKWARD_POINTWISE_FAMILIES,
    ACTIVATION_BACKWARD_POINTWISE_SPEC,
    ACTIVATION_POINTWISE_FAMILIES,
    GELU_AND_MUL_POINTWISE_SPEC,
    SILU_AND_MUL_POINTWISE_SPEC,
    SILU_AND_MUL_WITH_CLAMP_POINTWISE_SPEC,
    PointwiseFamilySpec,
    PointwisePlan,
    gelu_and_mul_pointwise,
    materialize_gelu_and_mul_plan,
    materialize_pointwise_family_plans,
    materialize_pointwise_plan,
    materialize_silu_and_mul_plan,
    materialize_silu_and_mul_with_clamp_plan,
    materialized_pointwise_plans,
    silu_and_mul_pointwise,
    silu_and_mul_with_clamp_pointwise,
)
from flag_gems.pt2.rms_norm import (
    FUSED_ADD_RMS_NORM_SPEC,
    RMS_NORM_BACKWARD_SPEC,
    RMS_NORM_SPEC,
    RMS_NORM_TRAINING_SPEC,
    rms_norm,
    supports_pt2_rms_norm,
)
from flag_gems.pt2.rotary_embedding import (
    ROTARY_EMBEDDING_INPLACE_SPEC,
    rotary_embedding_inplace,
    supports_pt2_triton,
)

__all__ = [
    "ACTIVATION_BACKWARD_POINTWISE_FAMILIES",
    "ACTIVATION_BACKWARD_POINTWISE_SPEC",
    "ACTIVATION_POINTWISE_FAMILIES",
    "CompileKind",
    "CompileOpSpec",
    "FUSED_ADD_RMS_NORM_SPEC",
    "gelu_and_mul_pointwise",
    "GELU_AND_MUL_POINTWISE_SPEC",
    "get_compile_manifest",
    "get_compile_spec",
    "grouped_topk",
    "GROUPED_TOPK_SPEC",
    "hc_head_fused_kernel",
    "HC_HEAD_FUSED_SPEC",
    "materialize_gelu_and_mul_plan",
    "materialize_pointwise_family_plans",
    "materialize_pointwise_plan",
    "materialize_silu_and_mul_plan",
    "materialize_silu_and_mul_with_clamp_plan",
    "materialized_pointwise_plans",
    "mhc_post",
    "MHC_POST_SPEC",
    "mhc_pre",
    "MHC_PRE_SPEC",
    "moe_sum",
    "MOE_SUM_SPEC",
    "PointwiseFamilySpec",
    "PointwisePlan",
    "rms_norm",
    "RMS_NORM_BACKWARD_SPEC",
    "RMS_NORM_SPEC",
    "RMS_NORM_TRAINING_SPEC",
    "rotary_embedding_inplace",
    "ROTARY_EMBEDDING_INPLACE_SPEC",
    "silu_and_mul_pointwise",
    "SILU_AND_MUL_POINTWISE_SPEC",
    "silu_and_mul_with_clamp_pointwise",
    "SILU_AND_MUL_WITH_CLAMP_POINTWISE_SPEC",
    "supports_pt2_fused_moe",
    "supports_pt2_moe_routing",
    "supports_pt2_rms_norm",
    "supports_pt2_triton",
    "TOPK_HASH_SOFTPLUS_SQRT_SPEC",
    "topk_softmax",
    "TOPK_SOFTMAX_SPEC",
    "topk_softplus_sqrt",
    "TOPK_SOFTPLUS_SQRT_SPEC",
    "uses_common_moe_routing_kernels",
]
