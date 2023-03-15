# Whisper Edge

OpenAI Whisper running on edge devices

## NVIDIA Jetson Nano

- https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit
- https://github.com/dusty-nv/jetson-inference

On Jetson Nano:
```
git clone --recursive https://github.com/dusty-nv/jetson-inference
```

On host machine:
```
git clone https://github.com/maxbbraun/whisper-edge.git
scp whisper-edge/Dockerfile.jetson-nano user@jetson-nano.local:~/jetson-inference/Dockerfile
```

On Jetson Nano:
```
cd jetson-inference
docker/build.sh dustynv/jetson-inference:r32.7.1
docker/run.sh
```
