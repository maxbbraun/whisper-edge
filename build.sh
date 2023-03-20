#!/usr/bin/env bash

sudo docker build \
  -f Dockerfile.jetson-nano \
  -t whisper-inference \
  .
