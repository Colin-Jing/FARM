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
    outfile: Path,
    aist_data_path: Path,
):
    """
    We need the babel file to get the duration of the clip
    to adjust the fps.
    """

    num_too_long = 0
    num_too_short = 0
    total_motions = 0


    output_motions = {}

    motions_path = os.path.join(aist_data_path, "motions-smpl")
    print(f"Processing {motions_path}")
    files = os.listdir(motions_path)
    motions_path = [
        os.path.join("motions-smpl", path) for path in files
    ]

    for path in motions_path:
        key = path

        if not os.path.exists(f"{aist_data_path}/{key}"):
            continue

        if key not in output_motions:
            output_motions[key] = []
            total_motions += 1

        output_motions[key].append(
            {
                "fps": 30,
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
    print(f"Num too long: {num_too_long}")
    print(f"Num too short: {num_too_short}")


    with open(outfile, "w") as file:
        yaml.dump(yaml_dict_format, file)


if __name__ == "__main__":
    typer.run(main)
