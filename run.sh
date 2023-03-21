#!/usr/bin/env bash

# Exit on error.
set -e

# Run the docker image with the streaming script. Pass through any arguments.
sudo docker run \
  --runtime nvidia \
  -it \
  --rm  \
  --network host \
  --device /dev/snd \
  whisper-inference \
  python stream.py \
  $@
