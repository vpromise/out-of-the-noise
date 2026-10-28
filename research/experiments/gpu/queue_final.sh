#!/bin/bash
# After eval B: extra digit extras, then the lighthouse (GPU otherwise idle), then the video toy, then the report.
set -e
cd "$(dirname "$0")/.."
P=.venv-gpu/bin/python
until grep -q EVAL_B_DONE logs/queue_eval.log; do sleep 20; done
$P gpu/eval_digits.py extras trajectories_more cfid_uncertainty
HF_HUB_OFFLINE=1 $P gpu/lighthouse.py
$P gpu/video_toy.py --steps 2500 --batch 64
$P gpu/build_report.py > logs/build_report.log
echo FINAL_DONE
