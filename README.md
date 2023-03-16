# Whisper Edge

Porting [OpenAI Whisper](https://github.com/openai/whisper) speech recognition to edge devices with hardware ML accelerators, enabling always-on live voice transcription. Current work includes [Jetson Nano](#jetson-nano) and [Coral Edge TPU](#coral-edge-tpu).

## Jetson Nano

Instructions for the [NVIDIA Jetson Nano Developer Kit (4GB)](https://developer.nvidia.com/embedded/jetson-nano-developer-kit). Additional [active cooling](https://noctua.at/en/nf-a4x10-flx) recommended.

### Model

The [`base.en` version](https://github.com/openai/whisper#available-models-and-languages) of Whisper seems to work best for the Jetson Nano:
 - `base` is the largest model size that fits into the 4GB of memory without modification.
 - Inference performance with `base` is ~10x real-time, which gives some room to process chunks shorter than 30 seconds (and therefore minimize the time from recording to transcription).
 - Using the english-only `.en` version further improves WER ([<5% on LibriSpeech test-clean](https://cdn.openai.com/papers/whisper.pdf)).

### Hack

Dilemma:
 - Whisper and some of its dependencies require Python 3.8.
 - The latest supported version of [JetPack](https://developer.nvidia.com/embedded/jetpack) for Jetson Nano is [4.6.3](https://developer.nvidia.com/jetpack-sdk-463), which is on Python 3.6.
 - [No easy way](https://github.com/maxbbraun/whisper-edge/issues/2) to update beyond Python 3.6 without losing CUDA support for PyTorch.

Solution:
 - Fork [whisper](https://github.com/maxbbraun/whisper) and [tiktoken](https://github.com/maxbbraun/tiktoken), downgrading them to Python 3.6.

### Setup

First, follow the [Jetson Nano Developer Kit setup instructions](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit). We will use [NVIDIA Docker containers](https://github.com/dusty-nv/jetson-inference) to run inference.

```
ssh user@jetson-nano.local
git clone --recursive https://github.com/dusty-nv/jetson-inference
exit
```

```
git clone https://github.com/maxbbraun/whisper-edge.git
scp whisper-edge/Dockerfile.jetson-nano user@jetson-nano.local:~/jetson-inference/Dockerfile
```

```
ssh user@jetson-nano.local
cd jetson-inference
docker/build.sh dustynv/jetson-inference:r32.7.1
exit
```

### Run

```
ssh user@jetson-nano.local
cd jetson-inference
docker/run.sh
```

## Coral Edge TPU

See the corresponding [issue](https://github.com/maxbbraun/whisper-edge/issues/1) about what supporting the [Google Coral Edge TPU](https://coral.ai/products/) may look like.
