#!/bin/bash

# Translated comment.
shopt -s expand_aliases

# Translated comment.
alias PYTHON_PATH="/home/xrj/IsaacLab/isaaclab.sh -p"

# checkpoint="results/ours_with_full_data/score_based.ckpt"
# checkpoint="data/pretrained_models/motion_tracker/smpl/last.ckpt"
# Translated comment.
# checkpoint="results/res_expert1_prior_s2_lrA_w1e-3_train175/score_based.ckpt"
# checkpoint="results/res_expert4_prior_s175_lrA_w1e-2_train15/score_based.ckpt"
checkpoint="results/res_expert2_prior_s15_lrA_w1e-1_train125/score_based.ckpt"
# checkpoint="results/res_expert4_prior_s125_lrA_w0.5_train1/score_based.ckpt"
# checkpoint="results/res_moe_sar_dea/score_based.ckpt"
# checkpoint="results/fa_res_moe_dea/score_based.ckpt"
# checkpoint="results/fa_res_moe_sar_top1/score_based.ckpt"
# checkpoint="results/fa_full_moe_sar_dea/score_based.ckpt"
# checkpoint="results/fa_no_moe/score_based.ckpt"


PYTHON_PATH protomotions/eval_agent.py +robot=smpl \
+simulator=isaaclab \
+checkpoint="$checkpoint" \
+headless=true \
+num_envs=138 \
+terrain=flat \
+motion_file=data/amass/amass_test.pt \
+speed_evaluation=1.0

PYTHON_PATH protomotions/eval_agent.py +robot=smpl \
+simulator=isaaclab \
+checkpoint="$checkpoint" \
+headless=true \
+num_envs=1320 \
+terrain=flat \
+motion_file=data/aist++/aist++_clean.pt \
+speed_evaluation=1.0

PYTHON_PATH protomotions/eval_agent.py +robot=smpl \
+simulator=isaaclab \
+checkpoint="$checkpoint" \
+headless=true \
+num_envs=1392 \
+terrain=flat \
+motion_file=data/mdm/mdm_clean.pt \
+speed_evaluation=1.0

PYTHON_PATH protomotions/eval_agent.py +robot=smpl \
+simulator=isaaclab \
+checkpoint="$checkpoint" \
+headless=true \
+num_envs=663 \
+terrain=flat \
+motion_file=data/motion_x/mesh_recovery/kungfu_g.pt \
+speed_evaluation=1.0

PYTHON_PATH protomotions/eval_agent.py +robot=smpl \
+simulator=isaaclab \
+checkpoint="$checkpoint" \
+headless=true \
+num_envs=173 \
+terrain=flat \
+motion_file=data/video_convert/video_convert_clean.pt \
+speed_evaluation=1.0

PYTHON_PATH protomotions/eval_agent.py +robot=smpl \
+simulator=isaaclab \
+checkpoint="$checkpoint" \
+headless=true \
+num_envs=45 \
+terrain=flat \
+motion_file=data/emdb/emdb.pt \
+speed_evaluation=1.0