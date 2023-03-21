#!/usr/bin/env bash

# Exit on error.
set -e

# Make the GPU available during the build.
sudo bash -c "cat > /etc/docker/daemon.json" << EOL
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOL
sudo systemctl restart docker

# Build the docker image.
sudo docker build \
  -f whisper-edge/Dockerfile.jetson-nano \
  -t whisper-inference \
  whisper-edge/
