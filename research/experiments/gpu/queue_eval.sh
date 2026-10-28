#!/bin/bash
# Sequential evaluation queue (every stage skips work already saved; re-run after a crash).
set -e
cd "$(dirname "$0")/.."
P=.venv-gpu/bin/python
while [ ! -f weights/digit_flow_ema.safetensors ]; do sleep 20; done
E="$P gpu/eval_digits.py"
$E gen ddpm_ddpm1000_w1 ddpm_ddim100_w1 ddpm_ddim50_w1 ddpm_ddim20_w1 ddpm_ddim10_w1 ddpm_ddim5_w1 ddpm_ddim2_w1 ddpm_ddim1_w1
$E gen flow_euler50_w1 flow_euler20_w1 flow_euler10_w1 flow_euler5_w1 flow_euler2_w1 flow_euler1_w1
$E gen ddpm_ddim50_w0 ddpm_ddim50_w2 ddpm_ddim50_w3 ddpm_ddim50_w5 ddpm_ddim50_w8
$E extras forward_process unet_activations same_noise_grids trajectories latency
$E metrics
$E extras nearest_neighbour_study
echo EVAL_A_DONE
[ -f weights/digit_dit_ema.safetensors ] || $P gpu/train.py --name dit --objective ddpm --arch dit --steps 10000
$E gen dit_ddim50_w1 dit_ddim50_w3
$E gen flow_euler20_w0 flow_euler20_w2 flow_euler20_w3 flow_euler20_w5 flow_euler20_w8
$E extras architecture loss_vs_t
$E metrics
echo EVAL_B_DONE
