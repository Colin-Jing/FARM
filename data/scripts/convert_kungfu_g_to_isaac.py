# This code is adapted from https://github.com/zhengyiluo/phc/ and generalized to work with any humanoid.
# https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/convert_amass_isaac.py

import os
import uuid
from pathlib import Path
from typing import Optional

import ipdb
import yaml
import joblib
import numpy as np
import torch
import typer
from scipy.spatial.transform import Rotation as sRot
import pickle
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from pycocotools.coco import COCO  # Import COCO from pycocotools
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from tqdm import tqdm

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
import time
from datetime import timedelta

TMP_SMPL_DIR = "/tmp/smpl"


def main(
    aist_root_dir: Path,
    robot_type: str = None,
    humanoid_type: str = "smpl",
    force_remake: bool = False,
    force_neutral_body: bool = True,
    not_upright_start: bool = False,  # By default, let's start upright (for consistency across all models).
    humanoid_mjcf_path: Optional[str] = None,
    force_retarget: bool = False,
):
    if robot_type is None:
        robot_type = humanoid_type
    elif robot_type in ["h1", "g1"]:
        assert (
            force_retarget
        ), f"Data is either SMPL or SMPL-X. The {robot_type} robot must use the retargeting pipeline."
    assert humanoid_type in [
        "smpl",
        "smplx",
        "smplh",
    ], "Humanoid type must be one of smpl, smplx, smplh"
    append_name = robot_type
    if force_retarget:
        append_name += "_retargeted"
    upright_start = not not_upright_start

    if humanoid_type == "smpl":
        mujoco_joint_names = SMPL_MUJOCO_NAMES
        joint_names = SMPL_BONE_ORDER_NAMES
    elif humanoid_type == "smplx" or humanoid_type == "smplh":
        mujoco_joint_names = SMPLH_MUJOCO_NAMES
        joint_names = SMPLH_BONE_ORDER_NAMES
    else:
        raise NotImplementedError

    left_to_right_index = []
    for idx, entry in enumerate(mujoco_joint_names):
        # swap text "R_" and "L_"
        if entry.startswith("R_"):
            left_to_right_index.append(mujoco_joint_names.index("L_" + entry[2:]))
        elif entry.startswith("L_"):
            left_to_right_index.append(mujoco_joint_names.index("R_" + entry[2:]))
        else:
            left_to_right_index.append(idx)

    folder_names = [
        "kungfu"
    ]

    ignore_list = []
    # ignore_folder_path = aist_root_dir / "kungfu_ignore"
    # if ignore_folder_path.exists():
    #     for video in ignore_folder_path.glob("*.mp4"):
    #         ignore_list.append(video.name.split(".")[0])
    
    ignore_list_txt = aist_root_dir / "ignore_kungfu_g_file_name.txt"
    if ignore_list_txt.exists():
        with open(ignore_list_txt, "r") as f:
            for line in f:
                ignore_list.append(line.strip())

    print(f"Ignore list: {ignore_list}")

    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": upright_start,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": humanoid_type,
        "sim": "isaacgym",
    }

    smpl_local_robot = SMPL_Robot(
        robot_cfg,
        data_dir="data/smpl",
    )

    if humanoid_mjcf_path is not None:
        skeleton_tree = SkeletonTree.from_mjcf(humanoid_mjcf_path)
    else:
        skeleton_tree = None

    uuid_str = uuid.uuid4()

    # Count total number of files that need processing
    start_time = time.time()
    total_files = 0
    total_files_to_process = 0
    processed_files = 0


    for folder_name in folder_names:
        if "retarget" in folder_name or "smpl" in folder_name or "h1" in folder_name:
            # Ignore folders where we store motions retargeted to AMP
            continue

        data_dir = aist_root_dir / folder_name
        output_dir = aist_root_dir / f"{folder_name}-{append_name}"

        print(f"Processing subset {folder_name}")
        os.makedirs(output_dir, exist_ok=True)

        files = [
            f
            for f in Path(data_dir).glob("**/*.json")         # input {aist_root_dir}/kungfu/*.json
            if (f.name != "shape.npz" and "stagei.npz" not in f.name)
        ]
        print(f"Processing {len(files)} files")

        filter_files = []
        for f in files:
            file_name = f.name.split(".")[0]
            if file_name in ignore_list:
                print(f"Skipping {f} as it is in the ignore list.")
                continue
            filter_files.append(f)
        
        print(f"Processing {len(filter_files)} files after ignore list")

        filter_files.sort()
        total_files_to_process += len(filter_files)
        total_files += len(files)

        print(f"Total files to process: {total_files_to_process}/{total_files}")

        for filename in tqdm(filter_files):
            try:
                relative_path_dir = filename.relative_to(data_dir).parent
                outpath = (
                    output_dir
                    / relative_path_dir
                    / filename.name.replace(".json", ".npy")   #output {aist_root_dir}/kungfu-{robot_type}/*.npy
                )

                # Check if the output file already exists
                if not force_remake and outpath.exists():
                    print(f"Skipping {filename} as it already exists.")
                    continue

                # Create the output directory if it doesn't exist
                os.makedirs(output_dir / relative_path_dir, exist_ok=True)

                print(f"Processing {filename}")
            
                if filename.suffix == ".json":
                    # Translated comment.
                    db = COCO(filename)
                    
                    # Translated comment.
                    pose_aa_list = []
                    trans_list = []
                    
                    # Translated comment.
                    ann_ids = sorted(db.anns.keys(), key=lambda x: int(x))
                    
                    for aid in ann_ids:
                        ann = db.anns[aid]
                        smplx_params = ann["smplx_params"]
                        
                        # Translated comment.
                        root_orient = smplx_params["root_orient"]  # [3]
                        pose_body = smplx_params["pose_body"]      # [63]
                        pose_aa_ = root_orient + pose_body          # [66]
                        
                        # Translated comment.
                        trans_ = smplx_params["trans"]              # [3]
                        
                        pose_aa_list.append(pose_aa_)
                        trans_list.append(trans_)
                    
                    # Translated comment.
                    pose_aa_array = np.array(pose_aa_list)
                    trans_array = np.array(trans_list)

                    transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False)
                    new_root = (transform * sRot.from_rotvec(pose_aa_array[:, :3])).as_rotvec()
                    pose_aa_array[:, :3] = new_root

                    trans_array = trans_array.dot(transform.as_matrix().T)
                    

                    # transform = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False)
                    # new_root = (transform * sRot.from_rotvec(aist_pose[:, :3])).as_rotvec()
                    # aist_pose[:, :3] = new_root

                    # aist_trans = aist_trans.dot(transform.as_matrix().T)
                    # aist_trans[:, 2] = aist_trans[:, 2] - (aist_trans[0, 2] - 0.92)
                    betas = np.zeros(10)
                    mocap_fr = 30
                    

                    # betas = motion_data["shape_est_betas"][:10]
                    gender = "neutral"  # motion_data["gender"]
                    # amass_pose = motion_data["pose_est_fullposes"]
                    # amass_trans = motion_data["pose_est_trans"]
                    # mocap_fr = motion_data["mocap_framerate"]
                else:
                    print(f"Skipping {filename} as it is not a valid file")
                    continue

                pose_aa = torch.tensor(pose_aa_array)
                aist_trans = torch.tensor(trans_array)
                betas = torch.from_numpy(betas)

                if force_neutral_body:
                    betas[:] = 0
                    gender = "neutral"

                motion_data = {
                    "pose_aa": pose_aa.numpy(),
                    "trans": aist_trans.numpy(),
                    "beta": betas.numpy(),
                    "gender": gender,
                }

                smpl_2_mujoco = [
                    joint_names.index(q) for q in mujoco_joint_names if q in joint_names
                ]
                batch_size = motion_data["pose_aa"].shape[0]

                if humanoid_type == "smpl":
                    pose_aa = np.concatenate(
                        [motion_data["pose_aa"][:, :66], np.zeros((batch_size, 6))],
                        axis=1,
                    )  # TODO: need to extract correct handle rotations instead of zero
                    pose_aa_mj = pose_aa.reshape(batch_size, 24, 3)[:, smpl_2_mujoco]
                    pose_quat = (
                        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
                        .as_quat()
                        .reshape(batch_size, 24, 4)
                    )

                if isinstance(gender, np.ndarray):
                    gender = gender.item()

                if isinstance(gender, bytes):
                    gender = gender.decode("utf-8")
                if gender == "neutral":
                    gender_number = [0]
                elif gender == "male":
                    gender_number = [1]
                elif gender == "female":
                    gender_number = [2]
                else:
                    ipdb.set_trace()
                    raise Exception("Gender Not Supported!!")

                if skeleton_tree is None:
                    smpl_local_robot.load_from_skeleton(
                        betas=betas[None,], gender=gender_number, objs_info=None
                    )
                    smpl_local_robot.write_xml(
                        f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml"
                    )
                    skeleton_tree = SkeletonTree.from_mjcf(
                        f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml"
                    )

                root_trans_offset = (
                    torch.from_numpy(motion_data["trans"])
                    + skeleton_tree.local_translation[0]
                )

                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
                    torch.from_numpy(pose_quat),
                    root_trans_offset,
                    is_local=True,
                )

                formats = ["regular"]

                for format in formats:
                    if robot_cfg["upright_start"]:
                        B = pose_aa.shape[0]
                        pose_quat_global = (
                            (
                                sRot.from_quat(
                                    sk_state.global_rotation.reshape(-1, 4).numpy()
                                )
                                * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                            )
                            .as_quat()
                            .reshape(B, -1, 4)
                        )
                    else:
                        pose_quat_global = sk_state.global_rotation.numpy()

                    trans = root_trans_offset.clone()

                    new_sk_state = SkeletonState.from_rotation_and_root_translation(
                        skeleton_tree,
                        torch.from_numpy(pose_quat_global),
                        trans,
                        is_local=False,
                    )

                    new_sk_motion = SkeletonMotion.from_skeleton_state(
                        new_sk_state, fps=mocap_fr
                    )


                    print(f"Saving to {outpath}")

                    new_sk_motion.to_file(str(outpath))

                    processed_files += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_file = elapsed_time / processed_files
                    remaining_files = total_files_to_process - processed_files
                    estimated_time_remaining = avg_time_per_file * remaining_files

                    print(
                        f"\nProgress: {processed_files}/{total_files_to_process} files"
                    )
                    print(
                        f"Average time per file: {timedelta(seconds=int(avg_time_per_file))}"
                    )
                    print(
                        f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}"
                    )
                    print(
                        f"Estimated completion time: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_time_remaining))}\n"
                    )
            except Exception as e:
                print(f"Error processing {filename}")
                print(f"Error: {e}")
                print(f"Line: {e.__traceback__.tb_lineno}")
                continue


if __name__ == "__main__":
    with torch.no_grad():
        typer.run(main)
