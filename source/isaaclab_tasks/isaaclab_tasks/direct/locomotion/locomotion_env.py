# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensor, RayCaster
import torch.nn.functional as F
import random

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class LocomotionEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()
        self.reward_components = 7
        self.reward_component_names = [
            "progress",
            "alive",
            "upright",
            "move_to_target",
            "action_l2",
            "energy",
            "joint_pos_limits",
        ]
        self.reward_component_task_rew = ["progress", "alive", "upright", "move_to_target"]

        if self.cfg.energy_rew:
            self.reward_component_names += ["energy_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["energy_rew"]
            self.cfg.energy_cost_scale = 0.
        if self.cfg.gait_rew:
            self.reward_component_names += ["gait_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["gait_rew"]
        if self.cfg.baseh_rew:
            self.reward_component_names += ["baseh_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["baseh_rew"]
        if self.cfg.armsw_rew:
            self.reward_component_names += ["armsw_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["armsw_rew"]
        if self.cfg.armsp_rew:
            self.reward_component_names += ["armsp_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["armsp_rew"]
        if self.cfg.kneelft_rew:
            self.reward_component_names += ["kneelft_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["kneelft_rew"]

        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._knee_ids, _ = self._contact_sensor.find_bodies(".*shin")
        self._upper_arm_ids, _ = self._contact_sensor.find_bodies(".*upper_arm")
        self._pelvis_ids, _ = self._contact_sensor.find_bodies("pelvis")
        self.feet_positions_prev = torch.ones((self.num_envs, 2, 2), dtype=torch.float32, device=self.sim.device) * 1e8
        self.max_z_knee = torch.zeros((self.num_envs,2), dtype=torch.float32, device=self.sim.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        forces = self.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.torso_position[:, 2].view(-1, 1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # base (always-present) reward components
        base_components = compute_rewards(
            self.actions,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.motor_effort_ratio,
        )
        electricity_cost = torch.sum(
                torch.abs(self.actions * self.dof_vel * self.cfg.dof_vel_scale) * self.motor_effort_ratio.unsqueeze(0),
                dim=-1,
            )
        

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        # Collect XY positions of feet with first contact (flattened over envs/feet); boolean mask must be 1D over first two dims
        feet_xy = self.robot.data.body_pos_w[:, self._feet_ids, :2]              # (num_envs, num_feet, 2)
        contact_mask = first_contact.to(torch.bool)                              # (num_envs, num_feet)
        contact_positions = feet_xy[contact_mask]                                # (K, 2) where K = total contacts
        # initialise
        self.feet_positions_prev[contact_mask] = torch.where(self.feet_positions_prev[contact_mask] == 1e8, contact_positions, self.feet_positions_prev[contact_mask])

        knee_z = self.robot.data.body_pos_w[:, self._knee_ids, 2] # normal 0.59-0.7
        knee_z -= self.robot.data.body_pos_w[:, self._pelvis_ids, 2] # wrt torso 

        upper_arm_xyz = self.robot.data.body_pos_w[:, self._upper_arm_ids, :3]        # (num_envs, num_arms, 3)
        pelvis_xyz = self.robot.data.body_pos_w[:, self._pelvis_ids, :3]              # (num_envs, 1, 3)
        # measure distance between each arm and pelvis and sum together (over the arm ids)
        arm_to_pelvis_dist = torch.norm(upper_arm_xyz - pelvis_xyz, dim=-1)  # (num_envs, num_arms)
        arm_to_pelvis_sum = arm_to_pelvis_dist.sum(dim=1)                                 # (num_envs,) [0.67-0.92]/2 * 2
        
        # measure velocity of the arm
        upper_arm_velocity = self.robot.data.body_state_w[:,self._upper_arm_ids,7]
        # measure absolute difference in arm velocity vectors
        upper_arm_velocity_diff = upper_arm_velocity[:,0] - upper_arm_velocity[:,1] # normal: 1-8 units

        base_height = self.robot.data.body_pos_w[:, self._pelvis_ids, 2] # 0.99 - 1.4 (jumping high)
        # append optional components (each is (num_envs,) -> (num_envs,1))
        comp_list = [base_components]
        if "log" not in self.extras:
            self.extras["log"] = dict()
        #for idx, rew in enumerate(self.reward_component_names[:base_components.shape[1]]):
        #    self.extras["log"][f"Episode_Reward/{rew}"] = base_components[:,idx].mean().item()*self.num_envs

        if self.cfg.energy_rew:
            energy_vals_tensor = [[2,4],[5,10],[10,20],[20,40],[40,80]]
            selected_bounds = energy_vals_tensor[self.energy_rew_vec[0].int()]  # (num_envs, 2)
            min_energy = selected_bounds[0]
            max_energy = selected_bounds[1]
            reward_electricity = ((electricity_cost >= min_energy) & (electricity_cost <= max_energy)).float()
            reward_electricity *= 5
            comp_list.append(reward_electricity.unsqueeze(-1))                            
            within  = (electricity_cost >= min_energy) & (electricity_cost <= max_energy)
            den = torch.ones_like(within).sum()                        # total valid feet this step
            pct = 100.0 * within.float().sum() / den                      # scalar %
            self.extras["log"]["energy_success_pct"] = pct
        #     self.extras["log"]["Episode_Reward/energy_rew"] = reward_electricity.mean().item()*self.num_envs

        if self.cfg.gait_rew:
            gait_vals_tensor = [[0,0.25],[0.25,0.5],[0.5,1.0],[1.0,1.5],[1.5,2.0]]
            selected_bounds = gait_vals_tensor[self.gait_rew_vec[0].int()]  # (num_envs, 2)
            gait_sizes = torch.zeros((self.num_envs,2), dtype=torch.float32, device=self.sim.device)
            gait_sizes[contact_mask] = torch.norm(torch.abs(contact_positions-self.feet_positions_prev[contact_mask]), dim=1) # 0.2-4 => 0.5, 1, 2,4,6
            min_value, max_value = selected_bounds[0], selected_bounds[1]
            reward_gait = ((gait_sizes >= min_value) & (gait_sizes <= max_value)).float()
            reward_gait *= last_air_time # make it fair for different stride times
            reward_gait *= contact_mask
            reward_gait = torch.sum(reward_gait, dim=-1) # sum over feet
            reward_gait *= 250
            self.feet_positions_prev[contact_mask] = contact_positions  # update only where contact made
            comp_list.append(reward_gait.unsqueeze(-1))
            valid   = contact_mask.bool()                                  # only count feet in contact
            within  = (gait_sizes >= min_value) & (gait_sizes <= max_value)
            success = within & valid
            den = valid.float().sum().clamp(min=1)                         # total valid feet this step
            pct = 100.0 * success.float().sum() / den                      # scalar %
            self.extras["log"]["gait_success_pct"] = pct
            #self.extras["log"]["Episode_Reward/gait_rew"] = reward_gait.mean().item()*self.num_envs

        if self.cfg.baseh_rew:
            bounds_list = [[0.,0.6],[0.5,0.8],[0.7,1.],[0.9,1.2],[1,1.4]]
            selected_bounds = bounds_list[self.baseh_rew_vec[0].int()]  # (num_envs, 2)
            min_value, max_value = selected_bounds[0], selected_bounds[1]
            reward_term = ((base_height >= min_value) & (base_height <= max_value)).float()
            reward_term *= 5
            comp_list.append(reward_term)
            within  = (base_height >= min_value) & (base_height <= max_value)
            den = torch.ones_like(within).sum()                        # total valid feet this step
            pct = 100.0 * within.float().sum() / den                      # scalar %
            self.extras["log"]["base_height_success_pct"] = pct
            #self.extras["log"]["Episode_Reward/baseh_rew"] = reward_term.mean().item()*self.num_envs

        if self.cfg.armsw_rew:
            bounds_list = [[0.,2],[4,6],[8,10],[12,14],[16,18],[20,22]]
            selected_bounds = bounds_list[self.armsw_rew_vec[0].int()]  # (num_envs, 2)
            min_value, max_value = selected_bounds[0], selected_bounds[1]
            reward_term = ((upper_arm_velocity_diff >= min_value) & (upper_arm_velocity_diff <= max_value)).float()
            reward_term *= 20
            comp_list.append(reward_term.unsqueeze(-1))
            within  = (upper_arm_velocity_diff >= min_value) & (upper_arm_velocity_diff <= max_value)
            den = torch.ones_like(within).sum()                        # total valid feet this step
            pct = 100.0 * within.float().sum() / den                      # scalar %
            self.extras["log"]["arm_swing_success_pct"] = pct
            #self.extras["log"]["Episode_Reward/armsw_rew"] = reward_term.mean().item()*self.num_envs

        if self.cfg.armsp_rew:
            bounds_list = [[0.,0.1],[0.2,0.3],[0.4,0.5],[0.6,0.7],[0.8,0.9],[1.0,1.1]]
            selected_bounds = bounds_list[self.armsp_rew_vec[0].int()]  # (num_envs, 2)
            min_value, max_value = selected_bounds[0], selected_bounds[1]
            reward_term = ((arm_to_pelvis_dist >= min_value) & (arm_to_pelvis_dist <= max_value)).float().mean(dim=-1)
            reward_term *= 5
            comp_list.append(reward_term.unsqueeze(-1))
            within  = (arm_to_pelvis_dist >= min_value) & (arm_to_pelvis_dist <= max_value)
            den = torch.ones_like(within).sum()                        # total valid feet this step
            pct = 100.0 * within.float().sum() / den                      # scalar %
            self.extras["log"]["arm_span_success_pct"] = pct
            #self.extras["log"]["Episode_Reward/armsp_rew"] = reward_term.mean().item()*self.num_envs

        if self.cfg.kneelft_rew:
            self.max_z_knee = torch.maximum(knee_z, self.max_z_knee)
            # compute max knee reward
            bounds_list = [[-0.35,-0.25],[-0.25,-0.15],[-0.15,0.],[0,0.15],[0.15,0.25],[0.25,0.35]]
            selected_bounds = bounds_list[self.kneelft_rew_vec[0].int()]  # (num_envs, 2)
            min_value, max_value = selected_bounds[0], selected_bounds[1]
            reward_term = ((self.max_z_knee >= min_value) & (self.max_z_knee <= max_value)).float()
            reward_term *= 20000
            reward_term *= last_air_time
            reward_term *= contact_mask
            reward_term = torch.sum(reward_term, dim=-1) # sum over both knees
            comp_list.append(reward_term.unsqueeze(-1))
            valid   = contact_mask.bool()                                  # only count feet in contact
            within  = (self.max_z_knee >= min_value) & (self.max_z_knee <= max_value)
            success = within & valid
            den = valid.float().sum().clamp(min=1)                         # total valid feet this step
            pct = 100.0 * success.float().sum() / den                      # scalar %
            self.extras["log"]["knee_height_success_pct"] = pct
            #self.extras["log"]["Episode_Reward/kneelft_rew"] = reward_term.mean().item()*self.num_envs
            # reset
            self.max_z_knee = torch.where(contact_mask, torch.zeros_like(self.max_z_knee), self.max_z_knee) # reset max knee height if foot in contact

        total_reward = torch.cat(comp_list, dim=-1)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.torso_position[:, 2] < self.cfg.termination_height
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()

        self.feet_positions_prev[env_ids] = torch.ones((len(env_ids), 2, 2), dtype=torch.float32, device=self.sim.device) * 1e8
        
@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    motor_effort_ratio: torch.Tensor,
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials
    
    total_reward = torch.stack([
        progress_reward,
        alive_reward,
        up_reward,
        heading_reward,
        -actions_cost_scale * actions_cost,
        - energy_cost_scale * electricity_cost,
        -dof_at_limit_cost
    ], dim=-1)

    return total_reward


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
