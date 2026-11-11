# LTX-2 Pipelines

High-level pipeline implementations for generating audio-video content with Lightricks' **LTX-2** model. This package provides ready-to-use pipelines for text-to-video, image-to-video, video-to-video, audio-to-video, keyframe interpolation, and retake tasks.

Pipelines are built using building blocks from [`ltx-core`](../ltx-core/) (schedulers, guiders, noisers, patchifiers) and handle the complete inference flow including model loading, encoding, decoding, and file I/O.

**Key Features:**

- 🎬 **Multiple Pipeline Types**: Text-to-video, image-to-video, video-to-video, audio-to-video, keyframe interpolation, detail-fidelity rendering, and retake
- 🎥 **HDR / EXR**: Native `--hdr` EXR conditioning with EXR + BT.2020/HLG output, plus dedicated HDR IC-LoRA
- ⚡ **Optimized Performance**: Support for FP8 transformers, gradient estimation, and memory optimization
- 🎯 **Production quality**: [`DFRPipeline`](docs/pipelines.md#12-dfrpipeline) for best output; DistilledPipeline for speed
- 🔧 **LoRA Support**: Easy integration with trained LoRA adapters
- 📦 **Self-Contained**: Handles model loading, encoding, decoding, and file I/O
- 🚀 **CLI Support**: All pipelines can be run as command-line scripts

## Quick Start

```bash
# From the repository root
uv sync --frozen

# Run a pipeline (example: two-stage text-to-video)
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path path/to/checkpoint.safetensors \
    --distilled-lora path/to/distilled_lora.safetensors 0.8 \
    --spatial-upsampler-path path/to/upsampler.safetensors \
    --gemma-root path/to/gemma \
    --prompt "A beautiful sunset over the ocean" \
    --output-path output.mp4
```

DFR uses the **same distilled transformer** as DistilledPipeline, plus `--detailing-lora` (no `--distilled-lora`). Copy-paste commands, 4K sizes, fps, and memory: [Running DFR](docs/pipelines.md#running-dfr).

See [Installation & Usage](docs/installation.md) for full setup, CLI modules, and shared flags. To call a pipeline from Python, wrap the call in `torch.inference_mode()`: see [Calling pipelines from Python](docs/installation.md#calling-pipelines-from-python).

## 📚 Documentation

| Topic | Description |
| ----- | ----------- |
| [Installation & Usage](docs/installation.md) | Install, requirements, running pipelines from the CLI, common flags |
| [Pipeline Selection Guide](docs/pipeline-selection.md) | Decision tree + feature comparison to pick the right pipeline |
| [Available Pipelines](docs/pipelines.md) | Full reference for all 12 pipelines, including [Running DFR](docs/pipelines.md#running-dfr) |
| [Conditioning Types](docs/conditioning.md) | Image and video conditioning methods |
| [HDR Support](docs/hdr.md) | Native EXR/`--hdr` colour spaces, I/O, and examples |
| [VideoSeal marking](docs/marking.md) | Opt-in post-hoc VideoSeal / AudioSeal (`--marker-root-path`) |
| [Multimodal Guidance](docs/multimodal-guidance.md) | CFG / STG / modality guidance parameters and tuning |
| [Optimization Tips](docs/optimization.md) | FP8 quantization, `torch.compile`, gradient estimation |
| [Multi-GPU Inference](docs/multigpu/README.md) | Run a single generation across GPUs for latency (SP, TDP, distributed VAE, distributed Gemma) |

## 🔗 Related Projects

- **[LTX-Core](../ltx-core/)** - Core model implementation and inference components (schedulers, guiders, noisers, patchifiers)
- **[LTX-Trainer](../ltx-trainer/)** - Training and fine-tuning tools

### LTX-2.5 ConvRot INT8 on ROCm

Install the ROCm PyTorch build for the target device before installing the
`ltx-core[convrot-rocm]` and `ltx-pipelines` packages from this fork. The optional
extra pins the standalone HIP kernel package. For a single-stage clip with sound,
use the split transformer, Gemma and VAE files:

```bash
python -m ltx_pipelines.distilled \
  --convrot-int8 --single-stage --offload cpu \
  --transformer-path /models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  --text-encoder-path /models/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  --video-vae-path /models/ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae-path /models/ltx-2.5-audio-vae-bf16.safetensors \
  --image /path/to/first-frame.png 0 0.85 \
  --prompt "A steady shot of a lighthouse with the sound of soft ocean waves." \
  --width 1024 --height 576 --frame-rate 24 --num-frames 121 --seed 7 \
  --output-path /path/to/result.mp4
```

The loader preserves the INT8 weights and FP32 scales. Quantized transformer,
Gemma and embeddings layers require native HIP ConvRot execution. Video and
audio VAEs use their original BF16 weights. This mode requires eager execution
and a pre-quantized split pack. LoRA fusion into the quantized weights is
unsupported.
