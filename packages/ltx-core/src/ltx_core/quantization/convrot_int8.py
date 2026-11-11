"""Load pre-quantized ConvRot INT8 weights and execute them with the ROCm HIP backend."""

from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from hashlib import sha256
import json

import torch
from torch import nn

from ltx_core.loader.checkpoint import CheckpointReader
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.loader.units import FuseRule


_HIP_CALLS: ContextVar[Counter[str] | None] = ContextVar("convrot_int8_hip_calls", default=None)


@contextmanager
def measure_hip_calls() -> Iterator[Counter[str]]:
    """Count native kernel submissions in the calling inference context."""
    calls: Counter[str] = Counter()
    token = _HIP_CALLS.set(calls)
    try:
        yield calls
    finally:
        _HIP_CALLS.reset(token)


class ConvRotInt8Linear(nn.Module):
    """Inference linear with INT8 weights and dynamic INT8 activation quantization."""

    def __init__(self, linear: nn.Linear, groupsize: int, scale_shape: torch.Size, component: str) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.groupsize = groupsize
        self.component = component
        self.weight = nn.Parameter(torch.empty_like(linear.weight, dtype=torch.int8), requires_grad=False)
        # Store the FP32 bits in an integer tensor so the loader's BF16 cast preserves them.
        self.register_buffer("weight_scale_bits", torch.empty(scale_shape, device=linear.weight.device, dtype=torch.int32))
        self.bias = linear.bias

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        if not torch.version.hip or activation.device.type != "cuda":
            raise RuntimeError("ConvRot INT8 inference requires a ROCm GPU")
        from comfy_kitchen.backends import hip

        if not hip._convrot_supported(
            self.in_features, self.groupsize, activation.device, activation.dtype, int8_global_spill=True
        ):
            raise RuntimeError("ConvRot INT8 has no native HIP kernel for this activation")
        output = hip.int8_linear(
            activation,
            self.weight,
            self.weight_scale_bits.view(torch.float32),
            self.bias,
            out_dtype=activation.dtype,
            convrot=True,
            convrot_groupsize=self.groupsize,
        )
        if (calls := _HIP_CALLS.get()) is not None:
            calls[self.component] += 1
        return output


def _preserve_scale(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    scale = inputs[key]
    if not scale.is_meta and not torch.all(torch.isfinite(scale) & (scale > 0)).item():
        raise ValueError(f"Invalid ConvRot INT8 weight scale for {key}")
    return {f"{key}_bits": scale.view(torch.int32)}


def _reject_lora(*_args: object) -> dict[str, torch.Tensor]:
    raise ValueError("LoRA fusion into pre-quantized ConvRot INT8 checkpoints is unsupported")


_CONVROT_NO_FUSION = FuseRule("convrot-int8-no-lora", torch.bfloat16, _reject_lora)
_CONVROT_STATE_OPS = SDOps("convrot-int8-scales-v1").with_kv_operation(_preserve_scale, key_suffix=".weight_scale")


def with_convrot_int8(builder: SimpleBuilder, *, component: str, selection: SDOps | None = None) -> SimpleBuilder:
    """Bind a split LTX ConvRot checkpoint using the builder's existing key mapping."""
    if builder.loras:
        raise ValueError("LoRA fusion into pre-quantized ConvRot INT8 checkpoints is unsupported")
    stages = builder.sd_ops + ((selection,) if selection is not None else ())
    linears = {}
    with CheckpointReader.open(builder.paths) as reader:
        sources, metadata = reader.scan()
        for key, (shape, dtype) in metadata.items():
            if dtype != torch.int8 or len(shape) != 2 or not key.endswith(".weight"):
                continue
            mapped = key
            for stage in stages:
                mapped = stage.apply_to_key(mapped)
                if mapped is None:
                    break
            if mapped is None:
                continue
            layer = key.removesuffix(".weight")
            quant_key, scale_key = f"{layer}.comfy_quant", f"{layer}.weight_scale"
            if metadata.get(quant_key, (None, None))[1] != torch.uint8:
                raise ValueError(f"Missing ConvRot INT8 metadata for {key}")
            info = json.loads(bytes(reader.read(sources[quant_key], quant_key).tolist()))
            groupsize = info.get("convrot_groupsize")
            if (
                info.get("format") != "int8_tensorwise"
                or info.get("convrot") is not True
                or groupsize not in (16, 64, 256)
                or shape[1] % groupsize != 0
            ):
                raise ValueError(f"Unsupported ConvRot INT8 metadata for {key}")
            scale_shape, scale_dtype = metadata.get(scale_key, (None, None))
            if scale_dtype != torch.float32 or scale_shape not in ((), (1,), (shape[0],), (shape[0], 1)):
                raise ValueError(f"Invalid ConvRot INT8 weight scale shape or dtype for {key}")
            linears[mapped.removesuffix(".weight")] = (shape, groupsize, scale_shape)
    if not linears:
        raise ValueError("No ConvRot INT8 weights mapped into the model")

    def replace_linears(model: nn.Module) -> nn.Module:
        for name, (shape, groupsize, scale_shape) in linears.items():
            linear = model.get_submodule(name)
            if not isinstance(linear, nn.Linear) or linear.weight.shape != shape:
                raise ValueError(f"ConvRot INT8 checkpoint does not match the linear at {name}")
            model.set_submodule(name, ConvRotInt8Linear(linear, groupsize, scale_shape, component))
        return model

    # Retain every non-quantization input key after the component-specific mapping.
    mapped_quant_keys = set()
    for key in metadata:
        if key.endswith(".comfy_quant"):
            mapped = key
            for stage in stages:
                mapped = stage.apply_to_key(mapped)
                if mapped is None:
                    break
            if mapped is not None:
                mapped_quant_keys.add(mapped)
    fingerprint = sha256(repr((component, sorted(linears.items()))).encode()).hexdigest()[:16]
    drop_quant_keys = SDOps(f"convrot-int8-keep:{fingerprint}", excluded_prefixes=tuple(sorted(mapped_quant_keys)))
    return replace(
        builder,
        sd_ops=(*stages, drop_quant_keys, _CONVROT_STATE_OPS),
        module_ops=(*builder.module_ops, ModuleOps(f"convrot-int8:{fingerprint}", lambda _model: True, replace_linears)),
        fuse_rule=_CONVROT_NO_FUSION,
    )
