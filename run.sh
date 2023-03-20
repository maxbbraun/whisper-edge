#!/usr/bin/env bash

sudo docker run \
  --runtime nvidia \
  -it \
  --rm  \
  --network host \
  --device /dev/snd \
  whisper-inference \
  python stream.py
