#!/bin/bash
# Sequential training queue (each step is resumable; re-run this script after a crash).
set -e
cd "$(dirname "$0")/.."
P=.venv-gpu/bin/python
[ -f weights/digit_ddpm_ema.safetensors ] || $P gpu/train.py --name ddpm --objective ddpm --steps 10000
[ -f weights/digit_classifier.safetensors ] || $P gpu/classifier.py
[ -f weights/digit_flow_ema.safetensors ] || $P gpu/train.py --name flow --objective flow --steps 10000
echo QUEUE_DONE
