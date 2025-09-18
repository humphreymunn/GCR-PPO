# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .g1_env_cfg import G1FlatEnvCfg, G1RoughEnvCfg

class G1Env(DirectRLEnv):
    cfg: G1FlatEnvCfg | G1RoughEnvCfg

    def __init__(self, cfg: G1FlatEnvCfg | G1RoughEnvCfg, render_mode: str | None = None,  context_size = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                #"lin_vel_z_l2",
                "feet_air_time",
                "termination",
                "feet_slide",
                "dof_pos_limits",
                "joint_deviation_hip",
                "joint_deviation_arms",
                "joint_deviation_torso",
                "joint_deviation_fingers",
                "lin_vel_z_l2" if self.cfg.lin_vel_z_l2_reward_scale != 0 else None,
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "flat_orientation_l2",
            ] if key is not None
        }
        self.reward_components = self.cfg.reward_components
        self.reward_component_names = self.cfg.reward_component_names
        self.reward_component_task_rew = self.cfg.reward_component_task_rew

        # Get specific body indices
        self._base_id = [2]#self._contact_sensor.find_bodies("torso_link")
        self._feetbody_ids, _ = self._contact_sensor.find_bodies(".*ankle.*")
        self._feet_ids = [15,16,19,20]#self._contact_sensor.find_bodies(".*ankle.*")
        self.ankle_ids = [15,16,19,20]#self._contact_sensor.find_bodies(".*ankle.*")
        self.hip_ids =  [0,1,3,4,7,8]#self._contact_sensor.find_bodies(".*(hip_yaw|hip_roll).*")
        self.arm_ids = [5,6,9,10,13,14,17,18,21,22]#self._contact_sensor.find_bodies([".*(_shoulder|_elbow).*"])
        self.finger_ids = [23,24,25,26,27,28,29,30,31,32,33,34,35,36]#self._contact_sensor.find_bodies(".*_(zero|one|two|three|four|five|six).*")
        #self._hand_ids,_ = self._contact_sensor.find_bodies(".*(hand|wrist).*")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*torso_link")
        self.time_to_sample = torch.zeros((self.num_envs),device=self.device)+10
        self.is_standing_env = torch.zeros((self.num_envs),device=self.device, dtype=torch.bool)
        self.feet_positions_prev = torch.ones((self.num_envs, 2, 2), dtype=torch.float32, device=self.sim.device) * 1e8
        self.max_z_knee = torch.zeros((self.num_envs,2), dtype=torch.float32, device=self.sim.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, G1RoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self.height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self.height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        # resample commands if time
        env_ids = torch.nonzero(self.time_to_sample <= 0, as_tuple=False).squeeze(-1)
        self._commands[env_ids,0] = torch.zeros_like(self._commands[env_ids,0]).uniform_(-1.0, 1.0)
        self._commands[env_ids,1] = torch.zeros_like(self._commands[env_ids,1]).uniform_(-1.0, 1.0)
        self._commands[env_ids,2] = torch.zeros_like(self._commands[env_ids,0]).uniform_(-1.0, 1.0)
        self.time_to_sample[:] -= self.step_dt
        self.time_to_sample[env_ids] = 10.0

        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        self._processed_actions = self._processed_actions.clip(-100.0, 100.0)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None

        if isinstance(self.cfg, G1RoughEnvCfg):
            height_data = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1) - self.height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
            height_data += torch.empty_like(height_data).uniform_(-0.1, 0.1)

        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b
        proj_grav = self._robot.data.projected_gravity_b
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel


        obs = torch.cat(
            [
                tensor
                for tensor in (
                    lin_vel + torch.empty_like(self._robot.data.root_lin_vel_b).uniform_(-0.1, 0.1),
                    ang_vel + torch.empty_like(self._robot.data.root_ang_vel_b).uniform_(-0.2, 0.2),
                    proj_grav + torch.empty_like(self._robot.data.projected_gravity_b).uniform_(-0.05, 0.05),
                    self._commands,
                    joint_pos + torch.empty_like(self._robot.data.joint_pos).uniform_(-0.01, 0.01),
                    joint_vel + torch.empty_like(self._robot.data.joint_vel).uniform_(-1.5, 1.5),
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        obs = torch.clip(obs, -1000.0, 1000.0)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # termination
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        termination = died
        # feet slide
        contact_sensor = self._contact_sensor
        contacts = contact_sensor.data.net_forces_w_history[:, :, self._feetbody_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        body_vel = self._robot.data.body_lin_vel_w[:, self.ankle_ids, :2]
        feet_slide = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
        # df pos limits
        # compute out of limits constraints
        out_of_limits = -(
            self._robot.data.joint_pos[:, self._feet_ids] - self._robot.data.soft_joint_pos_limits[:, self._feet_ids, 0]
        ).clip(max=0.0)
        out_of_limits += (
            self._robot.data.joint_pos[:, self._feet_ids] - self._robot.data.soft_joint_pos_limits[:,  self._feet_ids, 1]
        ).clip(min=0.0)
        dof_pos_limits = torch.sum(out_of_limits, dim=1)
        # joint deviation hip
        #: Articulation = self.scene[hip_asset.name]
        # compute out of limits constraints
        angle = self._robot.data.joint_pos[:, self.hip_ids] - self._robot.data.default_joint_pos[:, self.hip_ids]
        joint_deviation_hip = torch.sum(torch.abs(angle), dim=1)
        # joint deviation arms
        
        # compute out of limits constraints
        angle = self._robot.data.joint_pos[:, self.arm_ids] - self._robot.data.default_joint_pos[:, self.arm_ids]
        joint_deviation_arms = torch.sum(torch.abs(angle), dim=1)

        angle = self._robot.data.joint_pos[:, self.finger_ids] - self._robot.data.default_joint_pos[:, self.finger_ids]
        joint_deviation_fingers = torch.sum(torch.abs(angle), dim=1)

        # joint deviation torso
        # compute out of limits constraints
        angle = self._robot.data.joint_pos[:, self._base_id] - self._robot.data.default_joint_pos[:, self._base_id]
        joint_deviation_torso = torch.sum(torch.abs(angle), dim=1)

        contact_sensor = self._contact_sensor# ContactSensor = self.scene.sensors[sensor_cfg_feet.name]
        # compute the reward
        air_time = contact_sensor.data.current_air_time[:, self._feetbody_ids]
        contact_time = contact_sensor.data.current_contact_time[:, self._feetbody_ids]
        in_contact = contact_time > 0.0
        in_mode_time = torch.where(in_contact, contact_time, air_time)
        single_stance = torch.sum(in_contact.int(), dim=1) == 1
        air_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
        air_time = torch.clamp(air_time, max=self.cfg.feet_air_time_threshold)
        air_time *= torch.norm(self._commands[:, :2], dim=1) > 0.1

        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        lin_vel_z_l2 = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_xy_l2 = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        dof_torques_l2 = torch.sum(torch.square(self._robot.data.applied_torque[:, :]), dim=1)
        dof_acc_l2 = torch.sum(torch.square(self._robot.data.joint_acc[:, :]), dim=1)
        action_rate_l2 = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "termination": termination * self.cfg.termination_reward_scale * self.step_dt,
            "feet_slide": feet_slide * self.cfg.feet_slide_reward_scale * self.step_dt,
            "dof_pos_limits": dof_pos_limits * self.cfg.dof_pos_limits_reward_scale * self.step_dt,
            "joint_deviation_hip": joint_deviation_hip * self.cfg.joint_deviation_hip_reward_scale * self.step_dt,
            "joint_deviation_arms": joint_deviation_arms * self.cfg.joint_deviation_arms_reward_scale * self.step_dt,
            "joint_deviation_torso": joint_deviation_torso * self.cfg.joint_deviation_torso_reward_scale * self.step_dt,
            "joint_deviation_fingers": joint_deviation_fingers * self.cfg.joint_deviation_fingers_reward_scale * self.step_dt,
            "lin_vel_z_l2": lin_vel_z_l2 * self.cfg.lin_vel_z_l2_reward_scale * self.step_dt if self.cfg.lin_vel_z_l2_reward_scale != 0 else None,
            "ang_vel_xy_l2": ang_vel_xy_l2 * self.cfg.ang_vel_xy_l2_reward_scale * self.step_dt,
            "dof_torques_l2": dof_torques_l2 * self.cfg.dof_torques_l2_reward_scale * self.step_dt,
            "dof_acc_l2": dof_acc_l2 * self.cfg.dof_acc_l2_reward_scale * self.step_dt,
            "action_rate_l2": action_rate_l2 * self.cfg.action_rate_l2_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }

        feet_ids = [self._feetbody_ids[0],self._feetbody_ids[1]]
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, feet_ids]
        # Collect XY positions of feet with first contact (flattened over envs/feet); boolean mask must be 1D over first two dims
        feet_xy = self._robot.data.body_pos_w[:, feet_ids, :2]              # (num_envs, num_feet, 2)
        contact_mask = first_contact.to(torch.bool)                              # (num_envs, num_feet)
        contact_positions = feet_xy[contact_mask]                                # (K, 2) where K = total contacts

        # initialise
        self.feet_positions_prev[contact_mask] = torch.where(self.feet_positions_prev[contact_mask] == 1e8, contact_positions, self.feet_positions_prev[contact_mask])

        rewards = {key: value for key, value in rewards.items() if value is not None}
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.stack(list(rewards.values())).T

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def update_curriculum(self, it):
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self.is_standing_env[env_ids] = torch.where(torch.zeros_like(self.is_standing_env[env_ids]).to(torch.float32).uniform_(0.0,1.0) <= 0.02, True,False)
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        #self._commands[env_ids,1] = 0.0
        #self._commands[env_ids,0] = torch.abs(self._commands[env_ids,0])
        self._commands[torch.nonzero(self.is_standing_env)] *= 0
        self.time_to_sample[env_ids] = 10.0

        self.feet_positions_prev[env_ids] = torch.ones((len(env_ids), 2, 2), dtype=torch.float32, device=self.sim.device) * 1e8
        self.max_z_knee[env_ids] = torch.zeros((len(env_ids),2), dtype=torch.float32, device=self.sim.device)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)