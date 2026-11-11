import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.quantization.convrot_int8 import ConvRotInt8Linear, measure_hip_calls, with_convrot_int8


class TinyModel:
    @staticmethod
    def from_metadata(_metadata):
        return nn.Sequential(nn.Linear(16, 4, bias=False), nn.Linear(4, 2))


def checkpoint(tmp_path, *, groupsize=16, include_scale=True, scale_value=0.1234567):
    path = tmp_path / "model.safetensors"
    scale = torch.full((4, 1), scale_value, dtype=torch.float32)
    weight = torch.arange(64, dtype=torch.int8).reshape(4, 16)
    quant = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize})
    tensors = {
        "prefix.0.weight": weight,
        "prefix.0.comfy_quant": torch.tensor(list(quant.encode()), dtype=torch.uint8),
        "prefix.1.weight": torch.ones(2, 4, dtype=torch.bfloat16),
        "prefix.1.bias": torch.zeros(2, dtype=torch.bfloat16),
    }
    if include_scale:
        tensors["prefix.0.weight_scale"] = scale
    save_file(tensors, str(path))
    keys = SDOps("tiny-prefix").with_matching(prefix="prefix.").with_replacement("prefix.", "")
    return SimpleBuilder(TinyModel, (path,), (keys,)), weight, scale


def test_load_convrot_without_upcasting_weights_or_rounding_scales(tmp_path):
    builder, weight, scale = checkpoint(tmp_path)
    model = with_convrot_int8(builder, component="test").build(torch.device("cpu"), dtype=torch.bfloat16)
    assert isinstance(model[0], ConvRotInt8Linear)
    assert model[0].weight.dtype == torch.int8
    assert torch.equal(model[0].weight, weight)
    assert torch.equal(model[0].weight_scale_bits.view(torch.float32), scale)
    assert isinstance(model[1], nn.Linear)
    assert model[1].weight.dtype == torch.bfloat16

    with measure_hip_calls() as calls, pytest.raises(RuntimeError, match="requires a ROCm GPU"):
        model(torch.ones(1, 16, dtype=torch.bfloat16))
    assert not calls


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"groupsize": 256}, "Unsupported ConvRot INT8 metadata"),
        ({"include_scale": False}, "weight scale shape or dtype"),
        ({"scale_value": float("nan")}, "Invalid ConvRot INT8 weight scale"),
    ],
)
def test_reject_invalid_checkpoint(tmp_path, options, message):
    builder, _, _ = checkpoint(tmp_path, **options)
    with pytest.raises(ValueError, match=message):
        with_convrot_int8(builder, component="test").build(torch.device("cpu"), dtype=torch.bfloat16)


@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="ROCm GPU required")
def test_hip_kernel_matches_convrot_reference(tmp_path):
    from comfy_kitchen.backends.eager.quantization import int8_linear as reference

    builder, weight, scale = checkpoint(tmp_path)
    device = torch.device("cuda")
    model = with_convrot_int8(builder, component="test").build(device, dtype=torch.bfloat16)
    activation = torch.randn(16, 16, generator=torch.Generator().manual_seed(42)).to(device, torch.bfloat16)
    with measure_hip_calls() as calls:
        actual = model[0](activation)
    torch.cuda.synchronize(device)
    expected = reference(
        activation, weight.to(device), scale.to(device), convrot=True, convrot_groupsize=16
    )
    assert calls == {"test": 1}
    torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.02)


def test_vocoder_materializes_fp32_weights_without_unsupported_autocast():
    from ltx_core.model.audio_vae.vocoder import VocoderWithBWE

    class TinyVocoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(2, 2, 1).to(torch.bfloat16)

        def forward(self, value):
            return self.conv(value.mean(dim=-1))

    model = VocoderWithBWE.__new__(VocoderWithBWE)
    nn.Module.__init__(model)
    model.vocoder = TinyVocoder()
    model.bwe_generator = TinyVocoder()
    model.resampler = nn.Identity()
    model.input_sampling_rate = model.output_sampling_rate = 48000
    model.hop_length = 1
    model._compute_mel = lambda value: value.unsqueeze(2)
    with torch.inference_mode():
        output = model(torch.randn(1, 2, 16, 4, dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16
    assert output.shape == (1, 2, 16)
    assert torch.isfinite(output).all()
    assert all(value.dtype == torch.bfloat16 for value in model.parameters())
