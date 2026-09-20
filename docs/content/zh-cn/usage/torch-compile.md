---
title: torch.compile 与 PT2
weight: 82
---

# torch.compile 与 PT2

已覆盖的 FlagGems 公共函数保持原来的名字、参数和返回值。在 NVIDIA 上，
这些入口在 `torch.compiler.is_compiling()` 为真时选择 PT2 适配器，普通调用继续走 eager。
应用无需为编译模式改用另一组算子接口。

目前在 NVIDIA、PyTorch 2.11、Triton 3.6 上完成了验证。
适配器需要 `torch.library.triton_op` 和 `torch.library.wrap_triton`。
仅检测到这两个接口存在，不代表所有 Torch、Triton 或设备版本都已兼容。

## 覆盖范围与输入边界

| 公共函数 | 编译路径与边界 | PT2 接通的反向能力 |
| --- | --- | --- |
| `silu_and_mul`、`gelu_and_mul`（`none` / `tanh`）、`silu_and_mul_with_clamp` | Tensor 输入支持 FP16/BF16/FP32；广播、混合 stride、dtype promotion 共用 pointwise 元数据推导。默认准备输出 rank 0–5。 | x/y 一阶梯度；clamp limit 仍不可微。 |
| `rms_norm` | 一个 normalized dimension；x/weight 同 dtype、同设备；与 eager 一样将跨 stride 输入变为连续布局。已测 FP16/BF16/FP32。 | x/weight 一阶梯度。 |
| `fused_add_rms_norm` | 一个 normalized dimension；保持公共函数的返回和修改行为。 | 未新增训练支持。 |
| `apply_rotary_pos_emb` | 原地路径，并显式传入 `position_ids`；保持 query/key 写入行为。 | 未新增训练支持。 |
| `topk_softmax`、`topk_softplus_sqrt`、`grouped_topk`、`moe_sum` | 既有路由/归约适配，包括 softplus/hash 路由与输出缓冲写入。 | 未新增训练支持。 |
| `mhc_pre`、`mhc_post`、`hc_head_fused_kernel` | 既有 MHC 推理适配及原 kernel。 | 未新增完整 MHC 反向。 |

这不等于整个 FlagGems、所有 `out` 变体或所有 eager 合法输入都支持编译。
二阶梯度、vmap/JVP 不在本次覆盖范围内。不支持的 dtype 或尚未准备的 pointwise
结构组合会报错，适配器不会在 fullgraph 内自动切回 eager。

```python
import torch
import flag_gems

x = torch.randn(8, 128, device="cuda", requires_grad=True)
y = torch.randn(128, device="cuda", requires_grad=True)
compiled = torch.compile(flag_gems.silu_and_mul, fullgraph=True, dynamic=True)
compiled(x, y).sum().backward()
```

## 编译前准备与动态输入

在具备所需接口的 NVIDIA 环境中，导入 FlagGems 时会注册 PT2 算子，并在图捕获前
准备已配置的激活算子计划。准备过程生成/导入 Python 发射代码，取得已有的 Triton JIT 对象；
它不会执行激活 kernel，也不保存示例 Tensor。默认的 288 个元数据计划复用 48 份 kernel 物化结果。

每次调用仍推导广播 shape、stride、输出分配和发射规模。准备过程不冻结 token 数或张量地址。
它有 CPU 启动和内存开销，纯 eager 用户也会承担；目前注册/物化失败会从包导入处向外报错。

更高输出 rank 可以在调用 `torch.compile` 之前显式准备前向和反向算子族。
例如使用六维 SiLU 输入时：

```python
from flag_gems import pt2

pt2.materialize_pointwise_family_plans(
    ("silu_and_mul", "silu_and_mul.backward"),
    ranks=(6,),
    dtypes=(torch.float32,),
)
```

`dtypes` 指每个算子族经过类型提升后的输出 dtype。规范布局名为
`contiguous_c`（物理维度折叠）和 `strided`；旧名 `split_last_dim_c` 仍作为
`strided` 的别名接受。通过准备函数取得不可变的 plan；保留的 stride 顺序诊断属性
用于兼容既有读取，真正发射时使用当次调用的 stride。

## 新增或更新算子

新增 eager 算子不会自动注册到 PT2。如果实现本身已经可追踪，可以直接编译；
否则，需要复用原 kernel 编写适配器，并在公共入口进入不可追踪的日志、缓存或指针包装之前分流。

对于生成式 pointwise 算子，需要注册算子族、核对纯 Tensor 输入和输出数量，
并在捕获前准备所需的结构计划。目前的 pointwise 桥接支持 `LibEntry(JITFunction)`；
其他装饰链需要保留其 heuristics/autotune 语义的适配方式。
同时应核对 eager 生成器中该算子专用的发射策略。
接通已有反向时，还需注册 autograd 公式并准备反向算子族。
已有 kernel 的签名、输出数量或装饰链变化时，也应同步检查适配器。

验证应覆盖 eager 对照、fullgraph、动态 shape、支持的 dtype/布局、空输入、
修改与别名语义，以及承诺支持的梯度。部署集成还应验证新进程加载缓存和 CUDA Graph 重放。
FL 的 CachedOp 预热负责选择 backend，不会让未经适配的 kernel 自动变得可追踪；
相应 CachedOp 需要在预热前创建。
