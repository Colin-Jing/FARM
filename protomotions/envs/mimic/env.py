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

from typing import Dict, Optional

import torch
from torch import Tensor
from isaac_utils import rotations, torch_utils
from protomotions.envs.mimic.mimic_utils import (
    dof_to_local,
    exp_tracking_reward,
)
from protomotions.envs.base_env.env_utils.humanoid_utils import quat_diff_norm
from protomotions.simulator.base_simulator.config import MarkerConfig, VisualizationMarker, MarkerState

from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.mimic.components.mimic_obs import MimicObs
from protomotions.envs.mimic.components.mimic_motion_manager import MimicMotionManager
from protomotions.envs.mimic.components.masked_mimic_obs import MaskedMimicObs
# import numpy as np


class Mimic(BaseEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config, device, *args, **kwargs)
        # Tracks the internal mimic metrics.
        self.mimic_info_dict = {}

        self.mimic_obs_cb = MimicObs(self.config, self)
        if self.config.masked_mimic.enabled:
            self.masked_mimic_obs_cb = MaskedMimicObs(self.config.masked_mimic, self)

        self.failed_due_bad_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        # For dynamic sampling, we record whether the motion was respawned on a flat terrain.
        # We do not record failures on irregular terrain for prioritized sampling as there are no guarantees it should have succeeded.
        self.respawned_on_flat = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        
        
    def create_motion_manager(self):
        self.motion_manager = MimicMotionManager(self.config.motion_manager, self)
        
    def create_visualization_markers(self):
        if self.config.headless:
            return {}
        
        visualization_markers = super().create_visualization_markers()
        
        body_markers = []
        if self.config.masked_mimic.enabled:
            body_names = self.config.robot.trackable_bodies_subset
        else:
            body_names = self.config.robot.body_names

        for body_name in body_names:
            if (
                self.config.robot.mimic_small_marker_bodies is not None
                and body_name in self.config.robot.mimic_small_marker_bodies
            ):
                body_markers.append(MarkerConfig(size="small"))
            else:
                body_markers.append(MarkerConfig(size="regular"))
                
        body_markers_cfg = VisualizationMarker(
            type="sphere",
            color=(1.0, 0.0, 0.0),
            markers=body_markers
        )
        visualization_markers["body_markers"] = body_markers_cfg

        # Translated comment.
        # expert_markers = [MarkerConfig(size="regular")]
        # expert_markers_cfg = VisualizationMarker(
        #     type="sphere",
        # Translated comment.
        #     markers=expert_markers
        # )
        # visualization_markers["expert_markers"] = expert_markers_cfg
        # Translated comment.

        if self.config.masked_mimic.enabled:
            future_body_markers = []
            for body_name in self.config.robot.trackable_bodies_subset:
                if (
                    self.config.robot.mimic_small_marker_bodies is not None
                    and body_name in self.config.robot.mimic_small_marker_bodies
                ):
                    future_body_markers.append(MarkerConfig(size="small"))
                else:
                    future_body_markers.append(MarkerConfig(size="regular"))
            future_body_markers_cfg = VisualizationMarker(
                type="sphere",
                color=(1.0, 1.0, 0.0),
                markers=future_body_markers
            )
            visualization_markers["future_body_markers"] = future_body_markers_cfg
        
        return visualization_markers
        
    def get_markers_state(self):
        if self.config.headless:
            return {}

        markers_state = super().get_markers_state()
        
        # Update mimic markers
        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids, self.motion_manager.motion_times
        )

        target_pos = ref_state.rigid_body_pos
        target_pos += self.respawn_offset_relative_to_data.clone().view(
            self.num_envs, 1, 3
        )

        target_pos[..., -1:] += self.terrain.get_ground_heights(
            target_pos[:, 0]
        ).view(self.num_envs, 1, 1)

        if self.config.masked_mimic.enabled:
            num_conditionable_bodies = len(
                self.masked_mimic_obs_cb.conditionable_body_ids
            )
            target_pos = target_pos[
                :, self.masked_mimic_obs_cb.conditionable_body_ids, :
            ]

            inactive_markers = torch.ones(
                self.num_envs,
                num_conditionable_bodies,
                dtype=torch.bool,
                device=self.device,
            )

            mask_time_len = (
                self.config.masked_mimic.masked_mimic_target_pose.num_future_steps
            )

            translation_view = (
                self.masked_mimic_obs_cb.masked_mimic_target_bodies_masks.view(
                    self.num_envs, mask_time_len, num_conditionable_bodies + 1, 2
                )[:, 0, :-1, 0]
            )  # ignore the last entry, that is for speed/heading
            active_translations = translation_view == 1

            inactive_markers[active_translations] = False

            target_pos[inactive_markers] += 100

        target_pos = target_pos.view(self.num_envs, -1, 3)
        markers_state["body_markers"] = MarkerState(
            translation=target_pos,
            orientation=torch.zeros(self.num_envs, target_pos.shape[1], 4, device=self.device),
        )

        # Inbetweening markers
        if self.config.masked_mimic.enabled:
            ref_state = self.motion_lib.get_motion_state(
                self.motion_manager.motion_ids,
                self.masked_mimic_obs_cb.target_pose_time,
            )
            target_pos = ref_state.rigid_body_pos
            target_pos += self.respawn_offset_relative_to_data.clone().view(
                self.num_envs, 1, 3
            )
            target_pos[..., -1:] += self.terrain.get_ground_heights(
                target_pos[:, 0]
            ).view(self.num_envs, 1, 1)

            target_pos = target_pos[
                :, self.masked_mimic_obs_cb.conditionable_body_ids, :
            ]

            translation_view = self.masked_mimic_obs_cb.target_pose_joints_mask.view(
                self.num_envs, num_conditionable_bodies + 1, 2
            )[
                :, :-1, 0
            ]  # ignore the last entry, that is for speed/heading
            active_translations = translation_view == 1

            inactive_markers[active_translations] = False

            target_pos[inactive_markers] += 100

            target_pos[
                torch.logical_not(
                    self.masked_mimic_obs_cb.target_pose_visible_mask.view(-1)
                )
            ] += 100
            target_pos = target_pos.view(self.num_envs, -1, 3)

            markers_state["future_body_markers"] = MarkerState(
                translation=target_pos,
                orientation=torch.zeros(self.num_envs, target_pos.shape[1], 4, device=self.device),
            )
        
        # Translated comment.
        # expert_pos = torch.tensor([[2.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1, 1)
        # markers_state["expert_markers"] = MarkerState(
        #     translation=expert_pos,
        #     orientation=torch.zeros(self.num_envs, 1, 4, device=self.device),
        # )
        # Translated comment.        
        return markers_state

    def get_obs(self):
        obs = super().get_obs()
        mimic_obs = self.mimic_obs_cb.get_obs()
        obs.update(mimic_obs)
        if self.config.masked_mimic.enabled:
            masked_mimic_obs = self.masked_mimic_obs_cb.get_obs()
            obs.update(masked_mimic_obs)
        return obs

    def get_envs_respawn_position(
        self,
        env_ids,
        offset=0,
        rigid_body_pos: torch.tensor = None,
        requires_scene: torch.tensor = None,
    ):
        """
        Get the offset of the respawn position relative to the current position.
        Also updates the respawned_on_flat flag.
        """
        respawn_position = super().get_envs_respawn_position(
            env_ids, offset=offset, rigid_body_pos=rigid_body_pos, requires_scene=requires_scene
        )

        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids[env_ids],
            self.motion_manager.motion_times[env_ids],
        )
        target_cur_gt = ref_state.rigid_body_pos
        target_cur_root_pos = target_cur_gt[:, 0, :]

        self.respawn_offset_relative_to_data[env_ids, :2] = (
            respawn_position[:, :2] - target_cur_root_pos[:, :2]
        )

        # Check if spawned on flat, for prioritized sampling
        new_root_pos = respawn_position[..., :2].clone().reshape(env_ids.shape[0], 1, 2)
        new_root_pos = (new_root_pos / self.terrain.horizontal_scale).long()
        px = new_root_pos[:, :, 0].view(-1)
        py = new_root_pos[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.terrain.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.terrain.height_samples.shape[1] - 2)

        self.respawned_on_flat[env_ids] = self.terrain.flat_field_raw[px, py] == 0
        # if scene interaction motion -- also consider as "flat" for dynamic sampling measurements
        if requires_scene is not None and torch.any(requires_scene):
            self.respawned_on_flat[env_ids[requires_scene]] = True

        return respawn_position
    
    def get_motion_requires_scene(self, motion_ids):
        requires_scene = (
            torch.zeros_like(motion_ids, dtype=torch.bool, device=self.device)
        )
        # TODO
        # if (
        #     self.motion_lib.motion_to_scene_ids.shape[0] > 0
        #     and self.num_objects_per_scene > 0
        # ):
        #     motions_lacking_a_scene = self.motion_lib.scenes_per_motion[motion_ids] == 0
        #     assert not torch.any(motions_lacking_a_scene), "Motions lacking a scene are not supported."
        #
        #     motions_with_scenes = self.motion_lib.scenes_per_motion[motion_ids] > 0
        #     requires_scene[motions_with_scenes] = True

        return requires_scene

    def compute_reset(self):
        super().compute_reset()

        # self.flat_termination_buf = torch.zeros(
        #     self.num_envs, device=self.device, dtype=torch.long
        # )
        # flat_termination = torch.zeros_like(self.reset_buf, dtype=torch.bool)
        if self.config.mimic_early_termination:
            reward_too_bad = torch.zeros_like(self.reset_buf, dtype=torch.bool)
            for entry in self.config.mimic_early_termination:
                key = entry.mimic_early_termination_key
                thresh = entry.mimic_early_termination_thresh
                thresh_on_flat = entry.mimic_early_termination_thresh_on_flat
                value = self.mimic_info_dict[key]

                if entry.less_than:
                    entry_too_bad = value < thresh
                    entry_on_flat_too_bad = value < thresh_on_flat
                else:
                    entry_too_bad = value > thresh
                    entry_on_flat_too_bad = value > thresh_on_flat

                no_scene_interaction = ~self.agent_in_scene
                tight_tracking_threshold = no_scene_interaction & self.respawned_on_flat

                # Translated comment.
                # flat_termination_this_entry = tight_tracking_threshold & entry_on_flat_too_bad
                # Translated comment.
                entry_too_bad[tight_tracking_threshold] = entry_on_flat_too_bad[tight_tracking_threshold]
                reward_too_bad |= entry_too_bad

            has_reset_grace = self.motion_manager.get_has_reset_grace()
            reward_too_bad &= ~has_reset_grace

            self.reset_buf[reward_too_bad] = 1
            self.terminate_buf[reward_too_bad] = 1
            # self.flat_termination_buf[flat_termination] = 1
            self.log_dict["reward_too_bad"] = reward_too_bad.float().mean()

        done_clip = self.motion_manager.get_done_tracks()
        self.reset_buf[done_clip] = 1

    def process_kb(self, gt: Tensor, gr: Tensor):
        kb = gt[:, self.key_body_ids]

        if self.config.mimic_reward_config.relative_kb_pos:
            rt = gt[:, 0]
            rr = gr[:, 0]
            kb = kb - rt.unsqueeze(1)

            heading_rot = torch_utils.calc_heading_quat_inv(rr, True)
            rr_expand = heading_rot.unsqueeze(1).expand(rr.shape[0], kb.shape[1], 4)
            kb = rotations.quat_rotate(
                rr_expand.reshape(-1, 4), kb.view(-1, 3), True
            ).view(kb.shape)

        return kb

    def rotate_pos_to_local(self, pos: Tensor, heading: Optional[Tensor] = None):
        if heading is None:
            raise NotImplementedError("Heading is required for local rotation")
            # root_rot = self.rigid_body_rot[:, 0]
            root_rot = self.get_bodies_state().body_rot[:, 0]
            heading = torch_utils.calc_heading_quat_inv(root_rot, True)

        pos_num_dims = len(pos.shape)
        expanded_heading = heading.view(
            [heading.shape[0]] + [1] * (pos_num_dims - 2) + [heading.shape[1]]
        ).expand(pos.shape[:-1] + (4,))

        rotated = rotations.quat_rotate(
            expanded_heading.reshape(-1, 4), pos.reshape(-1, 3), True
        ).view(pos.shape)
        return rotated

    def compute_reward(self):
        """
        Abbreviations:

        gt = global translation
        gr = global rotation
        rt = root translation
        rr = root rotation
        kb = key bodies
        dv = dof (degrees of freedom velocity)
        """
        speeds = self.motion_manager.motion_speeds
        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids, self.motion_manager.motion_times
        )
        ref_gt = ref_state.rigid_body_pos
        ref_gr = ref_state.rigid_body_rot
        ref_lr = ref_state.local_rot
        # ref_gv = ref_state.rigid_body_vel * speeds.view(-1, 1, 1)
        last_ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids,
            self.motion_manager.motion_times - self.dt * speeds,
        )
        last_ref_gt = last_ref_state.rigid_body_pos
        ref_gv = (ref_gt - last_ref_gt) / self.dt
        ref_gav = ref_state.rigid_body_ang_vel * speeds.view(-1, 1, 1)
        ref_dv = ref_state.dof_vel

        ref_lr = ref_lr[:, self.simulator.get_dof_body_ids()]
        ref_kb = self.process_kb(ref_gt, ref_gr)

        current_state = self.simulator.get_bodies_state()
        gt, gr, gv, gav = (
            current_state.rigid_body_pos,
            current_state.rigid_body_rot,
            current_state.rigid_body_vel,
            current_state.rigid_body_ang_vel,
        )
        # first remove height based on current position
        relative_to_data_gt = gt.clone()
        relative_to_data_gt[:, :, -1:] -= self.terrain.get_ground_heights(gt[:, 0]).view(
            self.num_envs, 1, 1
        )
        # then remove offset to get back to the ground-truth data position
        relative_to_data_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[
            ..., :2
        ].view(self.num_envs, 1, 2)

        kb = self.process_kb(relative_to_data_gt, gr)

        rt = relative_to_data_gt[:, 0]
        ref_rt = ref_gt[:, 0]

        if self.config.mimic_reward_config.rt_ignore_height:
            rt = rt[..., :2]
            ref_rt = ref_rt[..., :2]

        rr = gr[:, 0]
        ref_rr = ref_gr[:, 0]

        inv_heading = torch_utils.calc_heading_quat_inv(rr, True)
        ref_inv_heading = torch_utils.calc_heading_quat_inv(ref_rr, True)

        rv = gv[:, 0]
        ref_rv = ref_gv[:, 0]

        rav = gav[:, 0]
        ref_rav = ref_gav[:, 0]

        dof_state = self.simulator.get_dof_state()
        lr = dof_to_local(dof_state.dof_pos, self.simulator.get_dof_offsets(), True)

        if self.config.mimic_reward_config.add_rr_to_lr:
            rr = gr[:, 0]
            ref_rr = ref_gr[:, 0]

            lr = torch.cat([rr.unsqueeze(1), lr], dim=1)
            ref_lr = torch.cat([ref_rr.unsqueeze(1), ref_lr], dim=1)

        rew_dict = exp_tracking_reward(
            gt=relative_to_data_gt,
            rt=rt,
            kb=kb,
            gr=gr,
            lr=lr,
            rv=rv,
            rav=rav,
            gv=gv,
            gav=gav,
            dv=dof_state.dof_vel,
            ref_gt=ref_gt,
            ref_rt=ref_rt,
            ref_kb=ref_kb,
            ref_gr=ref_gr,
            ref_lr=ref_lr,
            ref_rv=ref_rv,
            ref_rav=ref_rav,
            ref_gv=ref_gv,
            ref_gav=ref_gav,
            ref_dv=ref_dv,
            config=self.config.mimic_reward_config
        )
        dof_forces = self.simulator.get_dof_forces()
        power = torch.abs(torch.multiply(dof_forces, dof_state.dof_vel)).sum(dim=-1)
        pow_rew = -power

        has_reset_grace = self.motion_manager.get_has_reset_grace()
        pow_rew[has_reset_grace] = 0

        rew_dict["pow_rew"] = pow_rew

        local_ref_gt = self.rotate_pos_to_local(ref_gt, ref_inv_heading)
        local_gt = self.rotate_pos_to_local(relative_to_data_gt, inv_heading)
        cartesian_err = (
            ((local_ref_gt - local_ref_gt[:, 0:1]) - (local_gt - local_gt[:, 0:1]))
            .pow(2)
            .sum(-1)
            .sqrt()
            .mean(-1)
        )

        gt_per_joint_err = (ref_gt - relative_to_data_gt).pow(2).sum(-1).sqrt()
        gt_err = gt_per_joint_err.mean(-1)
        max_joint_err = gt_per_joint_err.max(-1)[0]

        # gv_err = (gv*self.dt*speeds.view(-1, 1, 1) - ref_gv*self.dt*speeds.view(-1, 1, 1)).pow(2).sum(-1).sqrt().mean(-1) # torch.norm(gv - ref_gv, p=2, dim=-1).mean(dim=-1)gv_err = (gv*self.dt*speeds.view(-1, 1, 1) - ref_gv*self.dt*speeds.view(-1, 1, 1)).pow(2).sum(-1).sqrt().mean(-1) # torch.norm(gv - ref_gv, p=2, dim=-1).mean(dim=-1)
        # gv_err = (relative_to_data_gt - )
        # if not hasattr(self, 'last_ref_gv'):
        #     self.last_ref_gv = 0
        #     self.last_gv = 0

        # ac_err = ((gv*self.dt*speeds.view(-1, 1, 1) - self.last_gv*self.dt*speeds.view(-1, 1, 1)) -
        #          (ref_gv*self.dt*speeds.view(-1, 1, 1) - self.last_ref_gv*self.dt*speeds.view(-1, 1, 1)))\
        #          .pow(2).sum(-1).sqrt().mean(-1)

        # self.last_ref_gv = ref_gv.clone()
        # self.last_gv = gv.clone()
        if not hasattr(self, 'last_relative_to_data_gt'):
            self.last_relative_to_data_gt = torch.zeros_like(relative_to_data_gt)
            self.last_ref_gt = torch.zeros_like(ref_gt)
            gv_err = torch.zeros_like(gt_err)
            ac_err = torch.zeros_like(gt_err)
        else:
            gv_err = ((ref_gt - self.last_ref_gt) - (relative_to_data_gt - self.last_relative_to_data_gt)).pow(2).sum(-1).sqrt().mean(-1)
            if not hasattr(self, 'last_last_relative_to_data_gt'):
                self.last_last_relative_to_data_gt = torch.zeros_like(relative_to_data_gt)
                self.last_last_ref_gt = torch.zeros_like(ref_gt)
                ac_err = torch.zeros_like(gt_err)
            else:
                ac_err = ((ref_gt - 2*self.last_ref_gt + self.last_last_ref_gt) - \
                          (relative_to_data_gt - 2*self.last_relative_to_data_gt + self.last_last_relative_to_data_gt))\
                          .pow(2).sum(-1).sqrt().mean(-1)
                self.last_last_relative_to_data_gt = self.last_relative_to_data_gt.clone()
                self.last_last_ref_gt = self.last_ref_gt.clone()
            self.last_relative_to_data_gt = relative_to_data_gt.clone()
            self.last_ref_gt = ref_gt.clone()
        
        rh_err = (ref_gt - relative_to_data_gt)[:, 0, -1].abs()

        gr_diff = quat_diff_norm(gr, ref_gr, True)
        gr_err = gr_diff.mean(-1)
        gr_err_degrees = gr_err * 180 / torch.pi

        max_gr_err = gr_diff.max(-1)[0]
        max_gr_err_degrees = max_gr_err * 180 / torch.pi

        lr_diff = quat_diff_norm(lr, ref_lr, True)
        lr_err = lr_diff.mean(-1)
        lr_err_degrees = lr_err * 180 / torch.pi
        max_lr_err = lr_diff.max(-1)[0]
        max_lr_err_degrees = max_lr_err * 180 / torch.pi

        # # cst
        # ################################################################################################
        # Translated comment.
        # Translated comment.
        # ref_gt, pred_gt = self.get_phc_metrics_data()
        
        # Translated comment.
        # mpjpe_g = torch.zeros(self.num_envs, device=self.device)
        # mpjpe_l = torch.zeros(self.num_envs, device=self.device)
        # accel_dist = torch.zeros(self.num_envs, device=self.device)
        # vel_dist = torch.zeros(self.num_envs, device=self.device)
        # # success_rate = torch.zeros(self.num_envs, device=self.device)
        
        # Translated comment.
        # if not hasattr(self, '_phc_frame_data'):
        #     self._phc_frame_data = {
        # Translated comment.
        # Translated comment.
        # Translated comment.
        # Translated comment.
        # Translated comment.
        # Translated comment.
        # Translated comment.        # Translated comment.        # Translated comment.        # Translated comment.        #     }
        
        # Translated comment.
        # if ref_gt is not None and pred_gt is not None:
        # Translated comment.
        #     mpjpe = (pred_gt - ref_gt).norm(dim=-1).mean(dim=-1)  # [num_envs]
            
        # Translated comment.
        #     self._phc_frame_data['mpjpe'].append(mpjpe.cpu().numpy())
        #     self._phc_frame_data['gt_pos'].append(ref_gt.cpu().numpy())
        #     self._phc_frame_data['pred_pos'].append(pred_gt.cpu().numpy())
        #     self._phc_frame_data['curr_steps'] += 1
            

            
        # Translated comment.
        #     motion_lengths = self.motion_lib.get_motion_length(self.motion_manager.motion_ids)
        # Translated comment.
        #     motion_num_steps_tensor = motion_num_steps.to(self.device)
            
        # Translated comment.
        #     if hasattr(self, 'reset_buf'):
        #         termination_state = torch.logical_and(
        #             self._phc_frame_data['curr_steps'] <= motion_num_steps_tensor - 1, 
        #             self.reset_buf
        #         )
        #         self._phc_frame_data['terminate_state'] = torch.logical_or(
        #             self._phc_frame_data['terminate_state'], 
        #             termination_state
        #         )
            
        # Translated comment.
        #     if (~self._phc_frame_data['terminate_state']).sum() > 0:
        #         max_possible_id = self.motion_lib.num_motions() - 1
        #         curr_ids = self.motion_manager.motion_ids
                
        # Translated comment.
        #         if (max_possible_id == curr_ids).sum() > 0:
        # Translated comment.
        #             bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                    
        # Translated comment.
        #             if (~self._phc_frame_data['terminate_state'][:bound]).sum() > 0:
        #                 curr_max = motion_num_steps_tensor[:bound][
        #                     ~self._phc_frame_data['terminate_state'][:bound]
        #                 ].max()
        #             else:
        # Translated comment.
        #                 curr_max = (self._phc_frame_data['curr_steps'] - 1)
        #         else:
        # Translated comment.
        #             curr_max = motion_num_steps_tensor[~self._phc_frame_data['terminate_state']].max()
                
        # Translated comment.
        # Translated comment.
        #         # if self._phc_frame_data['curr_steps'] >= curr_max: 
        #         #     curr_max = self._phc_frame_data['curr_steps'] + 1
        #     else:
        # Translated comment.
        #         curr_max = motion_num_steps_tensor.max()
            

            
        #     if (self._phc_frame_data['curr_steps'] >= curr_max or 
        #         self._phc_frame_data['terminate_state'].sum() == self.num_envs):
                
        # Translated comment.
        #         self._phc_frame_data['curr_steps'] = 0
                
        # Translated comment.
        #         self._phc_frame_data['terminate_memory'].append(
        #             self._phc_frame_data['terminate_state'].cpu().numpy()
        #         )
                
        # Translated comment.
        #         terminate_hist = np.concatenate(self._phc_frame_data['terminate_memory'])
        #         self._phc_frame_data['success_rate'] = (
        #             1 - terminate_hist[:self.motion_lib.num_motions()].mean()
        #         )
                
        # Translated comment.
        #         all_mpjpe = np.stack(self._phc_frame_data['mpjpe'])
                
        # Translated comment.
        #         curr_motion_ids = self.motion_manager.motion_ids.cpu().numpy()
                
        # Translated comment.
        #         all_mpjpe_per_motion = []
        #         pred_pos_per_motion = []
        #         gt_pos_per_motion = []
                
        #         for idx in range(len(curr_motion_ids)):
        #             motion_id = curr_motion_ids[idx]
        #             motion_steps = motion_num_steps_tensor[idx].item()
                    
        # Translated comment.
        #             actual_steps = min(motion_steps, len(all_mpjpe))
                    
        # Translated comment.
        #             motion_mpjpe = all_mpjpe[:actual_steps, idx].mean()
        #             all_mpjpe_per_motion.append(motion_mpjpe)
                    
        # Translated comment.
        #             all_pred_pos = np.stack(self._phc_frame_data['pred_pos'])
        #             all_gt_pos = np.stack(self._phc_frame_data['gt_pos'])
                    
        #             pred_pos_per_motion.append(all_pred_pos[:actual_steps, idx])
        #             gt_pos_per_motion.append(all_gt_pos[:actual_steps, idx])
                
        # Translated comment.
        #         self._phc_frame_data['mpjpe_all'].append(all_mpjpe_per_motion)
        #         self._phc_frame_data['pred_pos_all'].extend(pred_pos_per_motion)
        #         self._phc_frame_data['gt_pos_all'].extend(gt_pos_per_motion)
                
        #         if hasattr(self.motion_manager, 'start_idx'):
        #             if (self.motion_manager.start_idx + self.num_envs >= self.motion_lib.num_motions()):
        # Translated comment.
        #                 final_metrics = self.compute_final_phc_metrics()
        #                 if final_metrics:
        # Translated comment.
        #                     mpjpe_g = torch.full((self.num_envs,), final_metrics.get('mpjpe_g', 0.0), device=self.device)
        #                     mpjpe_l = torch.full((self.num_envs,), final_metrics.get('mpjpe_l', 0.0), device=self.device)
        #                     accel_dist = torch.full((self.num_envs,), final_metrics.get('accel_dist', 0.0), device=self.device)
        #                     vel_dist = torch.full((self.num_envs,), final_metrics.get('vel_dist', 0.0), device=self.device)
        #                     # success_rate = torch.full((self.num_envs,), final_metrics.get('success_rate', 0.0), device=self.device)
        #         else:
        # Translated comment.
        #             total_evaluated = len(self._phc_frame_data['pred_pos_all'])
        #             if total_evaluated >= self.motion_lib.num_motions():
        # Translated comment.
        #                 final_metrics = self.compute_final_phc_metrics()
        #                 if final_metrics:
        # Translated comment.
        #                     mpjpe_g = torch.full((self.num_envs,), final_metrics.get('mpjpe_g', 0.0), device=self.device)
        #                     mpjpe_l = torch.full((self.num_envs,), final_metrics.get('mpjpe_l', 0.0), device=self.device)
        #                     accel_dist = torch.full((self.num_envs,), final_metrics.get('accel_dist', 0.0), device=self.device)
        #                     vel_dist = torch.full((self.num_envs,), final_metrics.get('vel_dist', 0.0), device=self.device)
        #                     # success_rate = torch.full((self.num_envs,), final_metrics.get('success_rate', 0.0), device=self.device)
                
        # Translated comment.
        #         self._phc_frame_data['terminate_state'] = torch.zeros(self.num_envs, device=self.device)
                
        # Translated comment.
        #         self._phc_frame_data['mpjpe'] = []
        #         self._phc_frame_data['gt_pos'] = []
        #         self._phc_frame_data['pred_pos'] = []
        
        # Translated comment.
        # if hasattr(self, '_phc_frame_data') and len(self._phc_frame_data.get('pred_pos_all', [])) > 0:
        #     sequence_metrics = self.compute_phc_metrics_sequence()
        #     if sequence_metrics:
        # Translated comment.
        #         for metric_name, metric_value in sequence_metrics.items():
        #             print(f"{metric_name}: {metric_value:.3f}")
                
        # Translated comment.
        #         mpjpe_g = torch.full((self.num_envs,), sequence_metrics.get('mpjpe_g', 0.0), device=self.device)
        #         mpjpe_l = torch.full((self.num_envs,), sequence_metrics.get('mpjpe_l', 0.0), device=self.device)
        #         accel_dist = torch.full((self.num_envs,), sequence_metrics.get('accel_dist', 0.0), device=self.device)
        #         vel_dist = torch.full((self.num_envs,), sequence_metrics.get('vel_dist', 0.0), device=self.device)
        #         # success_rate = torch.full((self.num_envs,), sequence_metrics.get('success_rate', 0.0), device=self.device)
                
        # Translated comment.
        #         self.log_dict["eval/mpjpe_g"] = mpjpe_g.mean().item()
        #         self.log_dict["eval/mpjpe_l"] = mpjpe_l.mean().item()
        #         self.log_dict["eval/accel_dist"] = accel_dist.mean().item()
        #         self.log_dict["eval/vel_dist"] = vel_dist.mean().item()
        #         # self.log_dict["eval/success_rate"] = success_rate.mean().item()
        # ################################################################################################
        scaled_rewards: Dict[str, Tensor] = {
            k: v * getattr(self.config.mimic_reward_config.component_weights, f"{k}_w")
            for k, v in rew_dict.items()
        }

        tracking_rew = sum(scaled_rewards.values())

        self.rew_buf = tracking_rew + self.config.mimic_reward_config.positive_constant

        for rew_name, rew in rew_dict.items():
            self.log_dict[f"raw/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"raw/{rew_name}_std"] = rew.std()

        for rew_name, rew in scaled_rewards.items():
            self.log_dict[f"scaled/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"scaled/{rew_name}_std"] = rew.std()

        other_log_terms = {
            "tracking_rew": tracking_rew,
            "total_rew": self.rew_buf,
            "cartesian_err": cartesian_err,
            "gt_err": gt_err,
            "gr_err": gr_err,
            "gr_err_degrees": gr_err_degrees,
            "lr_err_degrees": lr_err_degrees,
            "max_joint_err": max_joint_err,
            "max_lr_err_degrees": max_lr_err_degrees,
            "max_gr_err_degrees": max_gr_err_degrees,
            "root_height_error": rh_err,
            "gv_err": gv_err,
            "ac_err": ac_err,
            # cst
            # "mpjpe_g": mpjpe_g,
            # "mpjpe_l": mpjpe_l,
            # "accel_dist": accel_dist,
            # "vel_dist": vel_dist,
            # "success_rate": success_rate
        }

        for rew_name, rew in other_log_terms.items():
            self.log_dict[f"mimic_other/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"mimic_other/{rew_name}_std"] = rew.std()

        self.mimic_info_dict.update(rew_dict)
        self.mimic_info_dict.update(other_log_terms)

    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device).long()

        self.mimic_obs_cb.compute_observations(env_ids)
        if self.config.masked_mimic.enabled:
            self.masked_mimic_obs_cb.compute_observations(env_ids)

    def pre_physics_step(self, actions):
        if self.config.mimic_residual_control:
            actions = self.residual_actions_to_actual(actions)

        return super().pre_physics_step(actions)

    def post_physics_step(self):
        self.motion_manager.post_physics_step()
        super().post_physics_step()
        self.motion_manager.handle_reset_track()
        
        if self.config.masked_mimic.enabled:
            self.masked_mimic_obs_cb.post_physics_step()
            
        # Translated comment.
        # if not hasattr(self, 'agent') and hasattr(self, 'simulator') and hasattr(self.simulator, 'agent'):
        #     self.agent = self.simulator.agent

    def user_reset(self):
        super().user_reset()
        self.motion_manager.motion_times[:] = 1e6

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if len(env_ids) > 0:
            if self.config.masked_mimic.enabled:
                self.masked_mimic_obs_cb.reset_track(env_ids)
        return super().reset(env_ids)

    def residual_actions_to_actual(
        self,
        residual_actions: Tensor,
        target_ids: Optional[Tensor] = None,
        target_times: Optional[Tensor] = None,
    ):
        if target_ids is None:
            target_ids = self.motion_manager.motion_ids

        if target_times is None:
            target_times = self.motion_manager.motion_times + self.dt

        ref_state = self.motion_lib.get_motion_state(target_ids, target_times)

        target_local_rot = dof_to_local(
            ref_state.dof_pos, self.simulator.get_dof_offsets(), True
        )
        residual_actions_as_quats = dof_to_local(
            residual_actions, self.simulator.get_dof_offsets(), True
        )

        actions_as_quats = rotations.quat_mul(
            residual_actions_as_quats, target_local_rot, True
        )
        actions = torch_utils.quat_to_exp_map(actions_as_quats, True).view(
            self.num_envs, -1
        )

        return actions

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict["motion_manager"] = self.motion_manager.get_state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.motion_manager.load_state_dict(state_dict["motion_manager"])

    # #cst
    # def get_phc_metrics_data(self):
    #     """
    # Translated comment.
    #     """
    # Translated comment.
    #     ref_state = self.motion_lib.get_motion_state(
    #         self.motion_manager.motion_ids, self.motion_manager.motion_times
    #     )
    # Translated comment.        
    # Translated comment.
    #     current_state = self.simulator.get_bodies_state()
    # Translated comment.        
    # Translated comment.
    #     if ref_gt is None or gt is None:
    #         return None, None
        
    # Translated comment.
    #     relative_to_data_gt = gt.clone()
    #     relative_to_data_gt[:, :, -1:] -= self.terrain.get_ground_heights(gt[:, 0]).view(
    #         self.num_envs, 1, 1
    #     )
    #     relative_to_data_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[
    #         ..., :2
    #     ].view(self.num_envs, 1, 2)
        
    #     return ref_gt, relative_to_data_gt

    # #cst
    # def compute_phc_metrics_sequence(self):
    #     """
    # Translated comment.
    # Translated comment.
    #     """
    #     try:
    #         if not hasattr(self, '_phc_frame_data') or len(self._phc_frame_data.get('pred_pos_all', [])) == 0:
    #             return {}
            
    # Translated comment.
    #         pred_pos_all = self._phc_frame_data['pred_pos_all']
    #         gt_pos_all = self._phc_frame_data['gt_pos_all']
            
    #         if len(pred_pos_all) == 0:
    #             return {}
            
    # Translated comment.
    #         try:
    #             from smpl_sim.smpllib.smpl_eval import compute_metrics_lite
                
    # Translated comment.
    #             metrics = compute_metrics_lite(pred_pos_all, gt_pos_all)
                
    # Translated comment.
    #             metrics_print = {m: np.mean(v) for m, v in metrics.items()}
                
    # Translated comment.
    #             if hasattr(self, '_phc_frame_data') and 'success_rate' in self._phc_frame_data:
    #                 metrics_print['success_rate'] = self._phc_frame_data['success_rate']
    #             else:
    #                 metrics_print['success_rate'] = 0.0
                
    #             return metrics_print
                
    #         except ImportError:

    #             return {}
                
    #     except Exception as e:
    #         return {}

    # #cst
    # def compute_final_phc_metrics(self):
    #     """
    # Translated comment.
    # Translated comment.
    # Translated comment.
    #     """
    #     try:
    # Translated comment.
    #         terminate_hist = np.concatenate(self._phc_frame_data['terminate_memory'])
            
    # Translated comment.
    #         succ_idxes = np.flatnonzero(~terminate_hist[:self.motion_lib.num_motions()]).tolist()
            
    # Translated comment.
    #         pred_pos_all_succ = [self._phc_frame_data['pred_pos_all'][i] for i in succ_idxes]
    #         gt_pos_all_succ = [self._phc_frame_data['gt_pos_all'][i] for i in succ_idxes]
            
    #         pred_pos_all = self._phc_frame_data['pred_pos_all'][:self.motion_lib.num_motions()]
    #         gt_pos_all = self._phc_frame_data['gt_pos_all'][:self.motion_lib.num_motions()]
            
    # Translated comment.
    # Translated comment.
    #         num_motions = self.motion_lib.num_motions()
    #         failed_keys = terminate_hist[:num_motions]
    #         success_keys = ~terminate_hist[:num_motions]
            
    # Translated comment.
    #         try:
    #             from smpl_sim.smpllib.smpl_eval import compute_metrics_lite
                
    # Translated comment.
    #             metrics_all = compute_metrics_lite(pred_pos_all, gt_pos_all)
    #             metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ) if pred_pos_all_succ else {}
                
    # Translated comment.
    #             metrics_all_print = {m: np.mean(v) for m, v in metrics_all.items()}
    #             metrics_succ_print = {m: np.mean(v) for m, v in metrics_succ.items()} if metrics_succ else {}
                
    # Translated comment.
    #             if len(metrics_succ_print) == 0:
    #                 metrics_succ_print = metrics_all_print
                
    # Translated comment.
    #             result_metrics = {
    #                 'mpjpe_g': metrics_all_print.get('mpjpe_g', 0.0),
    #                 'mpjpe_l': metrics_all_print.get('mpjpe_l', 0.0),
    #                 'accel_dist': metrics_succ_print.get('accel_dist', 0.0),
    #                 'vel_dist': metrics_succ_print.get('vel_dist', 0.0),
    #                 'success_rate': self._phc_frame_data['success_rate']
    #             }
    #             return result_metrics
                
    #         except ImportError:
    #             default_metrics = {
    #                 'mpjpe_g': 0.0,
    #                 'mpjpe_l': 0.0,
    #                 'accel_dist': 0.0,
    #                 'vel_dist': 0.0,
    #                 'success_rate': 0.0
    #             }
                
                
    #             return default_metrics
                
    #     except Exception as e:
    #         default_metrics = {
    #             'mpjpe_g': 0.0,
    #             'mpjpe_l': 0.0,
    #             'accel_dist': 0.0,
    #             'vel_dist': 0.0,
    #             'success_rate': 0.0
    #         }
            
    #         return default_metrics
