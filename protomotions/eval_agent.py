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
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

has_robot_arg = False
simulator = None
for arg in sys.argv:
    # This hack ensures that isaacgym is imported before any torch modules.
    # The reason it is here (and not in the main func) is due to pytorch lightning multi-gpu behavior.
    if "robot" in arg:
        has_robot_arg = True
    if "simulator" in arg:
        if not has_robot_arg:
            raise ValueError("+robot argument should be provided before +simulator")
        if "isaacgym" in arg.split("=")[-1]:
            import isaacgym  # noqa: F401

            simulator = "isaacgym"
        elif "isaaclab" in arg.split("=")[-1]:
            from isaaclab.app import AppLauncher

            simulator = "isaaclab"

        elif "genesis" in arg.split("=")[-1]:
            simulator = "genesis"

from lightning.fabric import Fabric  # noqa: E402
from utils.config_utils import *  # noqa: E402, F403

from protomotions.agents.ppo.agent import PPO  # noqa: E402


@hydra.main(config_path="config") # translated comment
def main(override_config: OmegaConf):

    os.chdir(hydra.utils.get_original_cwd())
    if override_config.checkpoint is not None:
        has_config = True

        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                print(f"Could not find config path: {config_path}")

        if has_config:
            print(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            # Translated comment.
            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )
            # Translated comment.
            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config

    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()

    if simulator == "isaaclab":
        app_launcher = AppLauncher({"headless": config.headless})
        simulation_app = app_launcher.app
        env = instantiate(
            config.env, device=fabric.device, simulation_app=simulation_app
        )
    else:
        env = instantiate(config.env, device=fabric.device)

    agent: PPO = instantiate(config.agent, env=env, fabric=fabric)#Config
    agent.setup()
    agent.eval_load(config.checkpoint)
    print("config.speed_evaluation = ", config.speed_evaluation)

    if config.headless:
        dataset_name = config.motion_file.split("/")[-1].split(".")[0]
        experiment_name = config.experiment_name

        # import torch
        # import numpy as np
        # gvs = agent.motion_lib.gvs
        # ave_gvs = torch.mean(torch.abs(gvs), dim=(1, 2))
        # Translated comment.
        # print("gvs_numpy.shape = ", gvs_numpy.shape)
        # folder_name = f"results_velocity"
        # Translated comment.
        # if not os.path.exists(folder_name):
        #     os.makedirs(folder_name)
        # folder_path = Path(folder_name)
        # print(f"Saving to: {folder_path / f'{dataset_name}.npy'}")
        # Translated comment.
        # np.save(folder_path / f"{dataset_name}.npy", gvs_numpy)

        result = agent.evaluate_tracking_performance(dataset_name, experiment_name)
        # speed = str(config.agent.config.speed_progressive[1])
        # expert_num = str(config.agent.config.multi_branch.branch_num)
        folder_name = f"results_eval_{experiment_name}"
        # Translated comment.
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open(f"{folder_name}/{dataset_name}_eval_result.txt", "w") as f:
            for key, value in result.items():
                print(f"{key}: {value}")
                f.write(f"{key}: {value}\n")
    else:
        agent.evaluate_policy()
        

if __name__ == "__main__":
    main()
