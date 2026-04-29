# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import typer
import yaml
from tqdm import tqdm

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState


@dataclass
class ProcessingOptions:
    ignore_occlusions: bool
    occlusion_bound: int = 0
    occlusion: int = 0


def fix_motion_fps(motion, dur):
    true_fps = motion.local_rotation.shape[0] / dur

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree,
        motion.local_rotation,
        motion.root_translation,
        is_local=True,
    )
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=true_fps)

    return new_motion


def is_valid_motion(
    occlusion_data: dict,
    motion_name: str,
    options: ProcessingOptions,
):
    if options.ignore_occlusions and len(occlusion_data) > 0:
        issue = occlusion_data["issue"]
        if (issue == "sitting" or issue == "airborne") and "idxes" in occlusion_data:
            bound = occlusion_data["idxes"][
                0
            ]  # This bounded is calculated assuming 30 FPS.....
            if bound < 10:
                options.occlusion_bound += 1
                print("bound too small", motion_name, bound)
                return False, 0
            else:
                return True, bound
        else:
            options.occlusion += 1
            print("issue irrecoverable", motion_name, issue)
            return False, 0

    return True, None


def main(
    outfile: Path = Path("data/yaml_files/amass_test.yaml"),
    amass_data_path: Path = Path("data/amass/amass"),
    motion_fps_path: Path = Path("data/yaml_files/motion_fps_smpl_test.yaml"),
    occlusion_data_path: Path = Path("data/amass/amass_copycat_occlusion_v3.pkl"),
    humanoid_type: str = "smpl",
    ignore_occlusions: bool = True,
):
    """
    We need the babel file to get the duration of the clip
    to adjust the fps.
    """
    total_motions = 0

    occlusion_data = joblib.load(occlusion_data_path)
    # print("occlusion_data = ", occlusion_data)

    motion_fps_dict = yaml.load(open(motion_fps_path, "r"), Loader=yaml.FullLoader)

    output_motions = {}

    options = ProcessingOptions(
        ignore_occlusions=ignore_occlusions,
    )

    for path, motion_fps in tqdm(motion_fps_dict.items()):
        path_parts = path.split(os.path.sep)
        path_parts[0] = path_parts[0] + "-" + humanoid_type
        key = os.path.join(*(path_parts))

        occlusion_key = "-".join(["0"] + ["_".join(path.split("/"))])[:-4]

        if not os.path.exists(f"{amass_data_path}/{key}"):
            continue

        if occlusion_key in occlusion_data:
            this_motion_occlusion = occlusion_data[occlusion_key]
        else:
            this_motion_occlusion = []


        is_valid, fps_30_bound_frame = is_valid_motion(
            this_motion_occlusion, occlusion_key, options
        )
        if not is_valid:
            continue

        if key not in output_motions:
            output_motions[key] = []
            total_motions += 1

        output_motions[key].append(
            {
                "fps": motion_fps,
            }
        )

    yaml_dict_format = {"motions": []}
    num_motions = 0
    for key, value in output_motions.items():
        item_dict = {
            "file": key,
            "fps": value[0]["fps"],
            "idx": num_motions,
            "weight": 1.0,
        }
        num_motions += 1

        yaml_dict_format["motions"].append(item_dict)

    print(f"Saving {len(output_motions)} motions to {outfile}")

    print(
        f"Num occluded: {options.occlusion}, occluded_bound: {options.occlusion_bound}"
    )

    with open(outfile, "w") as file:
        yaml.dump(yaml_dict_format, file)


if __name__ == "__main__":
    typer.run(main)
