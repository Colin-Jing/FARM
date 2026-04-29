# FARM: Frame-Accelerated Augmentation and Residual Mixture-of-Experts for Physics-Based High-Dynamic Humanoid Control

FARM improves high-dynamic physics-based motion tracking and was presented at **AAAI 2026 (Oral)**.

## Data policy (important)

The datasets used in this project come from external sources with redistribution restrictions.  
Therefore, this repository **does not distribute packaged training/evaluation data**.

- Public data access link:  
  https://drive.google.com/drive/folders/1h3naRVehFrEZ_mhJxiEZ_vzk7caJVuDZ?usp=sharing

## Environment setup

Environment setup is unchanged from ProtoMotions. Please follow:

- https://github.com/NVlabs/ProtoMotions/blob/main/README.md#installation

Example IsaacLab alias:

```bash
alias PYTHON_PATH="/path/to/isaaclab.sh -p"
```

## FARM checkpoints and configs

- Main checkpoint: `data/pretrained_models/Farm/score_based.ckpt`
- Related configs:
  - `protomotions/config/residual_moe_spare_gate_velocity_prior_125.yaml`

## Data processing

Use scripts in `data/scripts/` to build motion YAML/PT files from downloaded raw sources.

Typical AMASS-style pipeline:

```bash
python data/scripts/convert_amass_to_isaac.py <path_to_amass_root>
python data/scripts/process_hml3d_data.py data/yaml_files/amass_train.yaml <path_to_amass_root>
python data/scripts/package_motion_lib.py \
  data/yaml_files/amass_train.yaml \
  <path_to_amass_root> \
  data/amass/amass_train.pt
```

For other datasets (AIST++, MDM, VideoConvert, EMDB), run the corresponding `process_*_data.py` script first, then package with `package_motion_lib.py`.

## Training commands

FARM training in this repository follows this workflow:

### Step 1: Mine difficult motions with 1.25x evaluation

Run the pretrained motion tracker on `amass_train.pt` at 1.25x speed:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_PATH protomotions/eval_agent.py \
  +robot=smpl \
  +simulator=isaaclab \
  +checkpoint=data/pretrained_models/motion_tracker/smpl/last.ckpt \
  +motion_file=data/amass/amass_train.pt \
  +headless=true \
  +num_envs=1024 \
  +terrain=flat \
  +speed_evaluation=1.25 \
  +agent.config.fine_tune=false
```

This writes failed IDs to:

- `results_failed_motion_tracker_smpl/failed_motions_amass_train.txt`

### Step 2: Build `amass_train_difficult_125.pt`

1. Create difficult-only YAML:

```bash
python data/scripts/get_difficult_motion.py \
  --failed-ids-file results_failed_motion_tracker_smpl/failed_motions_amass_train.txt \
  --input-yaml data/yaml_files/amass_train.yaml \
  --output-yaml data/yaml_files/amass_train_difficult_125.yaml
```

2. Package to PT:

```bash
python data/scripts/package_motion_lib.py \
  data/yaml_files/amass_train_difficult_125.yaml \
  <path_to_amass_root> \
  data/amass/amass_train_difficult_125.pt
```

### Step 3: Train with FARM config (Frame-Accelerated augmentation + Residual MoE)

```bash
PYTHON_PATH protomotions/train_agent.py \
  --config-name=residual_moe_spare_gate_velocity_prior_125.yaml \
  +robot=smpl \
  +simulator=isaaclab \
  checkpoint=data/pretrained_models/motion_tracker/smpl/last.ckpt \
  motion_file=data/amass/amass_train_difficult_125.pt \
  experiment_name=farm_125
```

## Inference / evaluation commands

Single motion visualization:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_PATH protomotions/eval_agent.py \
  +robot=smpl \
  +simulator=isaaclab \
  +checkpoint=data/pretrained_models/Farm/score_based.ckpt \
  +headless=false \
  +num_envs=1 \
  +terrain=flat \
  +motion_file=data/aist++/motions-smpl/gBR_sBM_cAll_d04_mBR0_ch06.npy \
  +speed_evaluation=1.0
```

Dataset evaluation (headless):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_PATH protomotions/eval_agent.py \
  +robot=smpl \
  +simulator=isaaclab \
  +checkpoint=data/pretrained_models/Farm/score_based.ckpt \
  +headless=true \
  +num_envs=138 \
  +terrain=flat \
  +motion_file=data/amass/amass_test.pt \
  +speed_evaluation=1.0
```

See `instruct.txt` for additional single-sequence examples.

## Acknowledgement and citation

This project is built on top of ProtoMotions. Thanks to the ProtoMotions authors and contributors. Please respect all upstream licenses.

FARM paper page:

- https://ojs.aaai.org/index.php/AAAI/article/view/38924

If you use FARM, please cite:

```bibtex
@article{Tan_Chen_Li_Xu_Xu_2026,
  title={FARM: Frame-Accelerated Augmentation and Residual Mixture-of-Experts for Physics-Based High-Dynamic Humanoid Control},
  volume={40},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/38924},
  DOI={10.1609/aaai.v40i22.38924},
  number={22},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Tan, Jing and Chen, Shiting and Li, Yangfan and Xu, Weisheng and Xu, Renjing},
  year={2026},
  month={Mar.},
  pages={18575-18583}
}
```
