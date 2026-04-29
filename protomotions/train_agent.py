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

# Translated comment.
os.environ["WANDB_DISABLE_SENTRY"] = "true"  # Must be first environment variable
# Translated comment.
os.environ["WANDB_SILENT"] = "true"
# Translated comment.
os.environ["WANDB_DISABLE_CODE"] = "true"

os.environ["WANDB_API_KEY"] = "d873738e8f6f6966d06e6d0241f923b363cb1f5c"

# os.environ["WANDB_MODE"] = "offline"

import sys
from pathlib import Path
import logging
import hydra
from hydra.utils import instantiate

# Translated comment.
has_robot_arg = False
simulator = None
# Translated comment.
for arg in sys.argv:
    # This hack ensures that isaacgym is imported before any torch modules.
    # The reason it is here (and not in the main func) is due to pytorch lightning multi-gpu behavior.
    # Translated comment.
    if "robot" in arg:
        has_robot_arg = True
    # Translated comment.
    if "simulator" in arg:
        if not has_robot_arg:
            raise ValueError("+robot argument should be provided before +simulator")
        if "isaacgym" in arg.split("=")[-1]:
            import isaacgym  # noqa: F401

            simulator = "isaacgym"
        elif "isaaclab" in arg.split("=")[-1]:
            from isaaclab.app import AppLauncher

            simulator = "isaaclab"

import wandb  # noqa: E402
wandb.login(key=os.environ["WANDB_API_KEY"])

# Translated comment.
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from utils.config_utils import *  # noqa: E402, F403
from utils.common import seeding  # noqa: E402, F403

from protomotions.agents.ppo.agent import PPO  # noqa: E402

log = logging.getLogger(__name__)

# Translated comment.
@hydra.main(config_path="config", config_name="base")
def main(config: OmegaConf):
    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation
    # Translated comment.
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    # Translated comment.
    os.chdir(hydra.utils.get_original_cwd())

    torch.set_float32_matmul_precision("high") # translated comment
    # Translated comment.
    # Translated comment.
    save_dir = Path(config.save_dir)
    pre_existing_checkpoint = save_dir / "last.ckpt"
    checkpoint_config_path = save_dir / "config.yaml"
    if pre_existing_checkpoint.exists():
        log.info(f"Found latest checkpoint at {pre_existing_checkpoint}")
        # Load config from checkpoint folder
        if checkpoint_config_path.exists():
            log.info(f"Loading config from {checkpoint_config_path}")
            config = OmegaConf.load(checkpoint_config_path)

        # Set the checkpoint path in the config
        config.checkpoint = pre_existing_checkpoint
    # Translated comment.
    # Fabric should launch AFTER loading the config. This ensures that wandb parameters are loaded correctly for proper experiment resuming.
    # Translated comment.
    # Translated comment.
    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()

    # Translated comment.
    # Translated comment.
    if config.seed is not None:
        rank = fabric.global_rank
        if rank is None:
            rank = 0
        fabric.seed_everything(config.seed + rank)
        seeding(config.seed + rank, torch_deterministic=config.torch_deterministic)

    if simulator == "isaaclab":
        app_launcher_flags = {
            "headless": config.headless,
        }
        if fabric.world_size > 1:
            # This is needed when running with SLURM.
            # When launching multi-GPU/node jobs without SLURM, or differently, maybe this needs to be adapted accordingly.
            app_launcher_flags["distributed"] = True
            os.environ["LOCAL_RANK"] = str(fabric.local_rank)
            os.environ["RANK"] = str(fabric.global_rank)

        app_launcher = AppLauncher(app_launcher_flags)

        simulation_app = app_launcher.app
        env = instantiate(
            config.env, device=fabric.device, simulation_app=simulation_app
        )
    else:
        env = instantiate(config.env, device=fabric.device)

    agent: PPO = instantiate(config.agent, env=env, fabric=fabric)
    agent.setup()                       # translated comment
    agent.fabric.strategy.barrier()     # translated comment
    agent.load(config.checkpoint)       # translated comment


    # find out wandb id and save to config.yaml if 1st run:
    # wandb on rank 0
    # Translated comment.
    # Translated comment.
    if fabric.global_rank == 0 and not checkpoint_config_path.exists():
        if "wandb" in config:
            for logger in fabric.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

            # saving config with wandb id for next resumed run
            wandb_id = wandb.run.id
            log.info(f"wandb_id found {wandb_id}")
            unresolved_conf["wandb"]["wandb_id"] = wandb_id

        # only save before 1st run.
        # note, we save unresolved config for easier inference time logic
        log.info(f"Saving config file to {save_dir}")
        with open(checkpoint_config_path, "w") as file:
            OmegaConf.save(unresolved_conf, file)
    # Translated comment.
    agent.fabric.strategy.barrier()
    # Translated comment.
    agent.fit()


if __name__ == "__main__":
    # Translated comment.
    main()
