# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import copy
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.types import ArticulationActions
from .throwing_env_cfg import ThrowingGeneralEnvCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
import torch.nn.functional as F

class ThrowingEnv(DirectRLEnv):
    cfg: ThrowingGeneralEnvCfg

    def __init__(self, cfg: ThrowingGeneralEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #print(se)
        #assert False
        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        self.throwing_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_positions = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "throwing",
                "roll",
                "stability",
                "action_rate_l2",
                "dof_torques_l2",
                "dof_acc_l2",
            ]
        }

        self.reward_components = len(self._episode_sums.keys())
        self.reward_component_names = list(self._episode_sums.keys())
        self.reward_component_task_rew = ["throwing", "roll", "stability"]
        if self.cfg.baseh_rew:
            self.reward_component_names += ["baseh_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["baseh_rew"]
            self._episode_sums["baseh_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.energy_rew:
            self.reward_component_names += ["energy_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["energy_rew"]
            self._episode_sums["energy_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.ballrel_rew:
            self.reward_component_names += ["ballrel_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["ballrel_rew"]
            self._episode_sums["ballrel_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.bodymo_rew:
            self.reward_component_names += ["bodymo_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["bodymo_rew"]
            self._episode_sums["bodymo_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.lftarm_rew:
            self.reward_component_names += ["lftarm_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["lftarm_rew"]
            self._episode_sums["lftarm_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg.rgtarmrel_rew:
            self.reward_component_names += ["rgtarmrel_rew"]
            self.reward_components += 1
            self.reward_component_task_rew += ["rgtarmrel_rew"]
            self._episode_sums["rgtarmrel_rew"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)


        self.not_released_ball = torch.ones((self.num_envs), device=self.device).bool()
        self.throwing_reward = torch.ones((self.num_envs), device=self.device)*-1 # by default it is -1 if not thrown
        self.landing_time = torch.ones((self.num_envs), device=self.device)*-1
        self.throwing_reward_given = torch.zeros((self.num_envs), device=self.device).bool()
        self.stability_penalty_this_ep = torch.zeros((self.num_envs), device=self.device).bool()
        self.sum_open_hand_action = torch.zeros((self.num_envs), device=self.device)
        self.min_base_height = 0.38
        if self.cfg.arm_only:
            self.body_actions = 0.
        else:
            self.body_actions = 1. # max is 100%
        self.stability_timer = 2. # max is 2 (seconds)

        if self.cfg.distance_throw:
            self.distance_range = [4.,4.]
            self.throwing_commands[:,1] = torch.zeros_like(self.throwing_commands[:,1]).uniform_(0, 0.) # ~theta
            self.throwing_commands[:,2] = torch.zeros_like(self.throwing_commands[:,2]).uniform_(0., 0*2*torch.pi)#2*torch.pi) # phi
        else:
            self.distance_range = [1.,1.] 
            self.theta_range = [0., 1.]
            self.throwing_commands[:,1] = torch.zeros_like(self.throwing_commands[:,1]).uniform_(self.theta_range[0], self.theta_range[1]) # ~theta
            self.throwing_commands[:,2] = torch.zeros_like(self.throwing_commands[:,2]).uniform_(0., 2*torch.pi)#2*torch.pi) # phi
        
        self.throwing_commands[:,0] = torch.zeros_like(self.throwing_commands[:,0]).uniform_(self.distance_range[0], self.distance_range[1]) # distance
        self.released_ball_t = torch.zeros((self.num_envs), device=self.device)-1
        self.action_noise = 0.

        self.ball_distances = torch.zeros((self.num_envs), device=self.device) -1 

        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False

        self.prev_velocity = torch.zeros((self.num_envs, 3), device=self.device)-1000
        
        self.default_root_states = torch.zeros((self.num_envs, 3), device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.sphere_object = RigidObject(self.cfg.sphere_cfg)
        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.target_object = RigidObject(self.cfg.target_cfg)
            self.scene.rigid_objects["target"] = self.target_object

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.rigid_objects["sphere"] = self.sphere_object
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        sky_file = r"omniverse://localhost/NVIDIA/Assets/Skies/Indoor/ZetoCGcom_ExhibitionHall_Interior1.hdr"#http://omniverse-content-production.s3-us-west-2.amazonaws.com/Skies/Indoor/ZetoCG_com_WarehouseInterior2b.hdr"
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, #color=(0.7, 0.7, 0.7), \
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr")
        light_cfg.func("/World/Light", light_cfg, orientation=(1.0, 0.0, 0.0, 0.0))
        #spherelight_cfg = sim_utils.SphereLightCfg(intensity=2600.0, color=(1., 1.,1.), radius=1., treat_as_point=True)
        #spherelight_cfg.func("/World/envs/env_.*/Robot/Light", spherelight_cfg,  translation=(0.0, 0.0, 0.5))

    '''def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # copy actions to avoid in-place modification
        actions = actions.clone()
        limits = self._robot.data.joint_limits + self._robot.data.default_joint_pos.reshape((-1,1))
        # clip actions to be within the joint limits
        actions = torch.clamp(actions, limits[:, :, 0], limits[:,:, 1])
        return actions'''
    
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()# self.policy_actions.clone()
        
        default_positions = self._robot.data.default_joint_pos[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]].clone() # non arm indices
        scalar_factor = 3
        self._processed_actions = self.cfg.action_scale * self._actions[:,:-1] + default_positions
        
        # Define finger actions
        finger_positions = torch.tensor([-0.35, -0.61, 0, 0.61, 0.35, 0.61, 0.35], device=self.device) * scalar_factor
        finger_positions[1] /= 2  # right two link
        finger_positions[0] /= 2  # right one link

        # Create full finger action array from -14 to -1 (14 positions)
        full_finger_actions = torch.zeros((self.num_envs, 14), device=self.device)

        # Map finger positions to their indices (relative to the 14-position array)
        # Index mapping: -14=0, -13=1, ..., -2=12, -1=13
        finger_indices = [-3, -1, -9, -4, -10, -5, -11]  # Original indices
        array_indices = [14 + idx for idx in finger_indices]  # Convert to 0-based indices: [11, 13, 5, 10, 4, 9, 3]

        # Set finger positions at their corresponding array positions
        for i, pos_idx in enumerate(array_indices):
            full_finger_actions[:, pos_idx] = finger_positions[i]

        # Set finger actions to 0 if hand is open (using smoothed actions)
        hand_open_mask = (self._actions[:, -1] >= 0)
        full_finger_actions[~hand_open_mask] = 0

        # Append to _processed_actions
        self._processed_actions = torch.cat([self._processed_actions, full_finger_actions], dim=1)

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(torch.arange(self.num_envs,device=self.device))

    def _apply_action(self):
        # skip last action
        self._robot.set_joint_position_target(self._processed_actions)#[:,:-1])

    def _get_observations(self) -> dict:
        
        if not hasattr(self, "non_feet_ids"):
            self.non_feet_ids,_ = self._contact_sensor.find_bodies("^(?!.*ankle).*$")#spot:  "^(?!.*lleg).*$" , chuck: "^(?!.*foot).*$", g1: ankle
            self.hand_ids,_ = self._contact_sensor.find_bodies(".*(palm|five|six|three|four|zero|one|two).*")# spot:  ".*(fngr|wr1).*", chuck:  ".*(finger|cylinder).*", g1: .*(palm|five|six|three|four|zero|one|two).*
            self._hip_ids,_ = self._contact_sensor.find_bodies(".*(hip).*")
            self.left_hand_ids,_ = self._contact_sensor.find_bodies(".*left_(palm).*")# spot:  ".*(fngr|wr1).*", chuck:  ".*(finger|cylinder).*", g1: .*(palm|five|six|three|four|zero|one|two).*
            self.right_hand_ids,_ = self._contact_sensor.find_bodies(".*right_(palm).*")# spot:  ".*(fngr|wr1).*", chuck:  ".*(finger|cylinder).*", g1: .*(palm|five|six|three|four|zero|one|two).*

        self._previous_actions = self._actions.clone()
        # remove hand observations
        # non hand indices : [:23]
        joint_pos_info = (self._robot.data.joint_pos - self._robot.data.default_joint_pos)[:,:23]
        joint_vel_info = self._robot.data.joint_vel[:,:23]
        
        estimated_displacement,estim_time = self.check_ball_displacement(torch.arange(self.num_envs,device=self.device))
        estimated_displacement = 1 - estimated_displacement
        #base_height = self._robot.data.root_pos_w[:, 2]-self.min_base_height
        w,x,y,z = self._robot.data.root_quat_w.T
        roll = torch.atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x**2 + y**2)).to(self.device)

        noise_displace = (1 - self.action_noise) * estimated_displacement+ self.action_noise * torch.randn_like(estimated_displacement)
        
        self.prev_velocity = self._robot.data.root_lin_vel_b.clone()

        ball_data = self.sphere_object.data.body_state_w[:, 0, [7,8,9]] # z vel

        obs = torch.cat( 
            [
                tensor
                for tensor in (

                    self._robot.data.root_ang_vel_b if self.cfg.obs_ang_vel else None,
                    self._robot.data.projected_gravity_b if self.cfg.obs_proj_grav else None,
                    (roll.float() + (torch.rand_like(roll) * 0.02 - 0.01)).unsqueeze(-1) if self.cfg.obs_roll else None,
                    self.throwing_commands,#
                    joint_pos_info + (torch.rand_like(joint_pos_info) * 0.02 - 0.01),
                    joint_vel_info + (torch.rand_like(joint_vel_info) * 0.1 - 0.05),
                    self._actions,
                    self.not_released_ball.unsqueeze(-1).float() if self.cfg.obs_notrelease else None,
                    noise_displace.unsqueeze(-1) if self.cfg.obs_estimdisplace else None,#estimated_displacement.unsqueeze(-1).float() if self.cfg.obs_estimdisplace else None,
                    estim_time.unsqueeze(-1) if self.cfg.obs_estimdisplace else None,
                    ball_data if self.cfg.obs_estimdisplace else None,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        obs = torch.clip(obs, -1000,1000)
        #assert False
        observations = {"policy": obs}
        
        if torch.sum(torch.isnan(obs)) > 0:
            print(torch.sum(torch.isnan(obs),dim=0))
        return observations


    def _get_rewards(self) -> torch.Tensor:
        # throwing reward:
        # measure euclidean distance between ball and hand 
        if self.cfg.no_proj_motion:
            dist = torch.norm(self.sphere_object.data.body_pos_w[:, 0, 0:3] - self._robot.data.body_pos_w[:, [-15], :].squeeze(1), dim=1) 
            throwing_reward_condition = (dist > 0.25)

            target_positions = self.target_positions[:] + self._terrain.env_origins[:]
            targ_dist = torch.norm(self.sphere_object.data.body_pos_w[:, 0, 0:3] - target_positions, dim=1).reshape(self.num_envs)
            on_ground = self.sphere_object.data.body_pos_w[:, 0, 2] < 0.05

            # set to targ_dist if ball distances -1 or if targ_distance smaller than the current amount (excluding -1)
            self.ball_distances[:] = torch.where(~on_ground & ((self.ball_distances[:] == -1) | (targ_dist < self.ball_distances[:])), targ_dist, self.ball_distances[:])
            env_ids = torch.nonzero((self.reset_buf == 1) & throwing_reward_condition).reshape(-1)
            self.throwing_reward[:] = 0
            self.throwing_reward[env_ids] = self.ball_distances[env_ids]/self.throwing_commands[env_ids,0]
            self.throwing_reward[env_ids] = 1 - torch.min(torch.tensor(1.), self.throwing_reward[env_ids])
            assert False
        else:
            dist = torch.norm(self.sphere_object.data.body_pos_w[:, 0, 0:3] - self._robot.data.body_pos_w[:, [-15], :].squeeze(1), dim=1) 
            throwing_reward_condition = (dist > 0.25) & (~self.throwing_reward_given)
            self.not_released_ball &= (dist <= 0.25)
            env_ids = torch.nonzero(throwing_reward_condition).reshape(-1)
            self.throwing_reward[:] = 0
            if len(env_ids) > 0: 
                self.throwing_reward[env_ids], self.landing_time[env_ids] = self.check_ball_displacement(env_ids)
                self.throwing_reward[env_ids] = 1 - self.throwing_reward[env_ids]
                self.throwing_reward_given[env_ids] = True
                self.landing_time[env_ids] += self.episode_length_buf[env_ids] * self.step_dt
                ball_data = self.sphere_object.data.body_state_w[env_ids, 0, 9] # z vel
        ball_released_envs = env_ids

        # roll reward
        w,x,y,z = self._robot.data.root_quat_w.T
        roll = torch.atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x**2 + y**2))
        # make sure roll not nan
        roll = torch.where(torch.isnan(roll), torch.zeros_like(roll), roll)
        roll_rew = (-1/(1+torch.exp(-10*(torch.abs(roll)-0.3))))*(1-torch.exp(-(torch.abs(roll)-0.1)/0.1))
        # stability reward
        base_height_cond = (self._robot.data.root_pos_w[:, 2] <= self.min_base_height) # 0.42 for anymal
        hand_positions = self._robot.data.body_pos_w[:, [-1], :].reshape(self.num_envs,3) # 22 for anymal
        ball_positions = self.sphere_object.data.root_pos_w.clone()
        ball_not_thrown_cond = (torch.norm((hand_positions-ball_positions),dim=1) <= 0.25) & (self.reset_buf == 1) 
        #bad_throw_cond = (self.throwing_reward > -1) & (self.throwing_reward < 0.05)

        # collision detection
        # "finger" or "cylinder"
        # include all ids except feet
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)#[:,non_feet_ids]
        mask = torch.where(torch.norm((hand_positions-ball_positions),dim=1) <= 0.25, torch.zeros(1,device=self.device),torch.ones(1,device=self.device))

        first_contact[:,self.hand_ids] &= mask.view(-1,1).expand(-1, len(self.hand_ids)).to(torch.bool) # filter out collisions with ball touching the hand
        collision = torch.sum(first_contact[:,self.non_feet_ids],dim=1) > 0

        self.stability_penalty_this_ep |= (ball_not_thrown_cond | collision) 
        stability_rew = ((self.reset_buf == 1) & (~self.stability_penalty_this_ep)).float()
        if self.cfg.nonsparse_stability_reward:
            stability_rew = ((~(base_height_cond | ball_not_thrown_cond | collision)).float() / self.max_episode_length_s) * self.step_dt

        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        rewards = {
            "throwing": torch.max(torch.zeros_like(self.throwing_reward),self.throwing_reward.clone()) * self.cfg.throwing_reward_scale * self.max_episode_length_s,
            "roll": roll_rew * self.cfg.roll_reward_scale * self.step_dt,
            "stability": stability_rew * self.cfg.stability_reward_scale * self.max_episode_length_s,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
        }

        if self.cfg.baseh_rew:
            base_height = self._robot.data.body_pos_w[:, self._hip_ids[0], 2] # 0.99 - 1.4 (jumping high)
            bounds_list = [[0.35,0.4],[0.4,0.45],[0.45,0.5],[0.5,0.55],[0.55,0.6]] # starts around 0.6
            selected_bounds = bounds_list[self.baseh_rew_vec[0].int()]  # (num_envs, 2)
            min_value, max_value = selected_bounds[0], selected_bounds[1]
            reward_term = ((base_height >= min_value) & (base_height <= max_value)).float()
            reward_term *= self.step_dt
            rewards["baseh_rew"] = reward_term
            within  = (base_height >= min_value) & (base_height <= max_value)
            den = torch.ones_like(within).sum()                        # total valid feet this step
            pct = 100.0 * within.float().sum() / den                      # scalar %
            self.extras["log"]["base_height_success_pct"] = pct

        r_scale = 1.0
        if self.cfg.energy_rew:
            electricity_cost = torch.sum(
                torch.abs(self._actions[:,:-1] * self._robot.data.joint_vel[:,:23]),
                dim=-1,
            )
            energy_vals_tensor = [[40,70],[70,100],[100,130],[130,160],[160,200]] # 26 -> 140
            selected_bounds = energy_vals_tensor[self.energy_rew_vec[0].int()]  # (num_envs, 2)
            min_energy = selected_bounds[0]
            max_energy = selected_bounds[1]
            reward_electricity = ((electricity_cost >= min_energy) & (electricity_cost <= max_energy)).float()
            reward_electricity *= self.step_dt * r_scale
            rewards["energy_rew"] = reward_electricity
            within  = (electricity_cost >= min_energy) & (electricity_cost <= max_energy)
            den = torch.ones_like(within).sum()                        # total valid feet this step
            pct = 100.0 * within.float().sum() / den                      # scalar %
            self.extras["log"]["energy_success_pct"] = pct

        if self.cfg.ballrel_rew:
            # episode_length_buf is in steps; multiply by step_dt (s/step) to get seconds
            if len(ball_released_envs) == 0:
                reward_term = torch.zeros(self.num_envs, device=self.device)
                rewards["ballrel_rew"] = reward_term
                self.extras["log"]["ballrel_success_pct"] = torch.tensor(0., device=self.device)
            else:
                ball_released_time = self.episode_length_buf[ball_released_envs].float() * self.step_dt
                measured_val = ball_released_time # around 0.2
                bounds_list = [[0.,0.3],[0.3,0.6],[0.6,0.9],[0.9,1.2],[1.2,1.5]] # starts around 0.6
                selected_bounds = bounds_list[self.ballrel_rew_vec[0].int()]  # (num_envs, 2)
                min_value, max_value = selected_bounds[0], selected_bounds[1]
                reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
                reward_term *= r_scale
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                reward_term_full[ball_released_envs] = reward_term
                rewards["ballrel_rew"] = reward_term_full
                within  = (measured_val >= min_value) & (measured_val <= max_value)
                den = torch.ones_like(within).sum()                        # total valid feet this step
                pct = 100.0 * within.float().sum() / den                      # scalar %
                self.extras["log"]["ballrel_success_pct"] = pct

        if self.cfg.bodymo_rew:
            if len(ball_released_envs) == 0:
                reward_term = torch.zeros(self.num_envs, device=self.device)
                rewards["bodymo_rew"] = reward_term
                self.extras["log"]["bodymo_success_pct"] = torch.tensor(0., device=self.device)
            else:
                measured_val = torch.norm(self._robot.data.body_vel_w[:, self._hip_ids[0], :3], dim=-1) # around 2
                bounds_list = [[0.,1.0],[1.0,2],[2,3],[3,4],[4,5]] # starts around 0.6
                selected_bounds = bounds_list[self.bodymo_rew_vec[0].int()]  # (num_envs, 2)
                min_value, max_value = selected_bounds[0], selected_bounds[1]
                reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
                reward_term *= r_scale
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                reward_term_full[ball_released_envs] = reward_term[ball_released_envs]
                rewards["bodymo_rew"] = reward_term_full
                within  = (measured_val >= min_value) & (measured_val <= max_value)
                den = torch.ones_like(within).sum()                        # total valid feet this step
                pct = 100.0 * within.float().sum() / den                      # scalar %
                self.extras["log"]["bodymo_success_pct"] = pct

        if self.cfg.lftarm_rew:
            measured_val = self._robot.data.body_pos_w[:, self.left_hand_ids[0], 2] # around 0.8
            bounds_list = [[0.6,0.8],[0.8,1.0],[1.0,1.2],[1.2,1.4],[1.4,1.6]] # starts around 0.6
            selected_bounds = bounds_list[self.lftarm_rew_vec[0].int()]  # (num_envs, 2)
            min_value, max_value = selected_bounds[0], selected_bounds[1]
            reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
            reward_term *= self.step_dt * r_scale
            rewards["lftarm_rew"] = reward_term
            within  = (measured_val >= min_value) & (measured_val <= max_value)
            den = torch.ones_like(within).sum()                        # total valid feet this step
            pct = 100.0 * within.float().sum() / den                      # scalar %
            self.extras["log"]["lftarm_success_pct"] = pct
        if self.cfg.rgtarmrel_rew:
            if len(ball_released_envs) == 0:
                reward_term = torch.zeros(self.num_envs, device=self.device)
                rewards["rgtarmrel_rew"] = reward_term
                self.extras["log"]["rgtarmrel_success_pct"] = torch.tensor(0., device=self.device)
            else:
                measured_val = self._robot.data.body_pos_w[:, self.right_hand_ids[0], 2] # around 0.5
                bounds_list = [[0.3,0.4],[0.4,0.5],[0.5,0.6],[0.6,0.7],[0.7,0.8]] # starts around 0.6
                selected_bounds = bounds_list[self.rgtarmrel_rew_vec[0].int()]  # (num_envs, 2)
                min_value, max_value = selected_bounds[0], selected_bounds[1]
                reward_term = ((measured_val >= min_value) & (measured_val <= max_value)).float()
                reward_term *= r_scale
                reward_term_full = torch.zeros(self.num_envs, device=self.device)
                reward_term_full[ball_released_envs] = reward_term[ball_released_envs]
                rewards["rgtarmrel_rew"] = reward_term_full
                within  = (measured_val >= min_value) & (measured_val <= max_value)
                den = torch.ones_like(within).sum()                        # total valid feet this step
                pct = 100.0 * within.float().sum() / den                      # scalar %
                self.extras["log"]["rgtarmrel_success_pct"] = pct

        #reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.stack(list(rewards.values())).T

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        

        if self.cfg.arm_only:
            self.body_actions = 0.
        else:
            self.body_actions = 1. # max is 100%

        if self.cfg.distance_throw:
            self.distance_range = [max(self.distance_range[0],4),max(self.distance_range[0],4)]
        
        if self.cfg.no_proj_motion:
            self.cfg.obs_estimdisplace = False



        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        if self.cfg.distance_throw:
            # sample new throwing commands uniformly
            self.throwing_commands[:,1] = torch.zeros_like(self.throwing_commands[:,1]).uniform_(0, 0.) # ~theta
            self.throwing_commands[:,2] = torch.zeros_like(self.throwing_commands[:,2]).uniform_(0., 0*2*torch.pi)#2*torch.pi) # phi
        else:
            # sample new throwing commands uniformly
            self.throwing_commands[:,1] = torch.zeros_like(self.throwing_commands[:,1]).uniform_(self.theta_range[0], self.theta_range[1]) # ~theta
            self.throwing_commands[:,2] = torch.zeros_like(self.throwing_commands[:,2]).uniform_(0., 2*torch.pi)#2*torch.pi) # phi
        self.throwing_commands[:,0] = torch.zeros_like(self.throwing_commands[:,0]).uniform_(self.distance_range[0], self.distance_range[1]) # distance


        self.target_positions[env_ids] = self.calculate_target_offset(env_ids)
        self.throwing_reward[env_ids] = -1 
        self.landing_time[env_ids] = -1
        self.throwing_reward_given[env_ids] = False
        self.not_released_ball[env_ids] = True
        self.sum_open_hand_action[env_ids] = 0
        self.stability_penalty_this_ep[env_ids] = False
        self.released_ball_t[env_ids] = -1
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] 
        #joint_pos[:,[1,3,5, 7,9, 11]] += torch.zeros_like(joint_pos[:,[1,3,5, 7,9, 11]]).uniform_(-self.cfg.arm_dr_range, self.cfg.arm_dr_range) # hand .uniform_(-1,1)
        joint_pos[:,[5,6,9,10,13,14,17,18,21,22]] += torch.zeros_like(joint_pos[:,[5,6,9,10,13,14,17,18,21,22]]).uniform_(-0.3, 0.3) # hand .uniform_(-1,1)
        joint_pos += torch.zeros_like(joint_pos).uniform_(-0.05, 0.05)

        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        if self.sim.render_mode.value != self.sim.render_mode.NO_GUI_OR_RENDERING:
            self.initialise_target_for_rendering(env_ids)

        object_default_state = self.sphere_object.data.default_root_state.clone()[env_ids]
        #object_default_state[:, 7:] = torch.zeros_like(self.sphere_object.data.default_root_state[env_ids, 7:])
        # initialise ball in hand
        finger_positions = self._robot.data.body_pos_w[:, [-1,-4,-5], :][env_ids]
        finger_positions = torch.mean(finger_positions, dim=1).reshape(len(env_ids),3)
        hand_positions = self._robot.data.body_pos_w[:, [-1], :][env_ids].reshape(len(env_ids),3)
        #hand_positions[:,2] += 1
        vector_AB = finger_positions - hand_positions 
        distance_AB = torch.norm(vector_AB,dim=1).reshape(len(env_ids),1) # 
        direction = vector_AB/distance_AB
        distance_AC = 0. * distance_AB
        position_c = finger_positions#hand_positions + direction*distance_AC
        object_default_state[:, 0:3] += position_c
        object_default_state[:, 7:] = self._robot.data.body_vel_w[:, [-1], :][env_ids].reshape(len(env_ids),6)
        self.sphere_object.write_root_state_to_sim(object_default_state, env_ids)

        if self.cfg.no_proj_motion:
            self.ball_distances[env_ids] = -1

        self.prev_velocity[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)-1000

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
            if "stability" in key:
                extras["Episode_Reward/" + key] /= self.cfg.stability_reward_scale
            if "throwing" in key:
                extras["Episode_Reward/" + key] /= self.cfg.throwing_reward_scale
            if "roll" in key:
                extras["Episode_Reward/" + key] /= self.cfg.roll_reward_scale
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Target_Distance"] = max(self.distance_range)
        extras["Theta_Range_Max"] = max(self.theta_range)
        extras["Body_Actions"] = self.body_actions
        extras["Stability_Timer"] = self.stability_timer
        self.extras["log"].update(extras)
        
    def update_curriculum(self, iter):
        stability_value = self.extras['log']['Episode_Reward/stability']
        throwing_value = self.extras['log']['Episode_Reward/throwing']
        #self.action_noise = min(1., self.action_noise + 0.001)

        if (throwing_value > self.cfg.r_throw_thresh and stability_value > self.cfg.r_stability_thresh) or \
            (iter >= 5000):
            if self.cfg.distance_throw:
                self.distance_range = [min(self.cfg.max_throw_dist, self.distance_range[0] + 0.01),min(self.cfg.max_throw_dist, self.distance_range[1]+0.01)]
            else:
                self.distance_range = [min(1., self.distance_range[0]),min(5.0, self.distance_range[1]+0.01)]
                self.theta_range = [min(0., self.theta_range[0]), min(1.0, self.theta_range[1]+0.01)]

    def calculate_target_offset(self, env_ids: torch.Tensor) -> torch.Tensor:
        res = torch.zeros((len(env_ids),3),device=self.device)
        dist, theta, phi = self.throwing_commands[env_ids, 0], torch.arccos(self.throwing_commands[env_ids,1]), self.throwing_commands[env_ids,2]
        x = dist * torch.sin(theta) * torch.cos(phi)
        y = dist * torch.sin(theta) * torch.sin(phi)
        z = dist * torch.cos(theta)
        res = torch.stack((x,y,z),dim=1)
        return res.squeeze(-1)
    
    def check_ball_displacement(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ball_data = self.sphere_object.data.body_state_w[env_ids, 0, :10]
        target_positions = self.target_positions[env_ids] + self._terrain.env_origins[env_ids] 
        target_positions[:,:2] += self.default_root_states[env_ids,:2] #self.scene.env_origins[env_ids]

        if self.cfg.air_resistance:
            # handle air resistance case if needed...
            raise NotImplementedError
        else:
            # initial velocities
            vx0 = ball_data[:, 7]
            vy0 = ball_data[:, 8]
            vz0 = ball_data[:, 9]
            a = 9.81

            # initial height minus 3cm floor offset
            z0 = ball_data[:, 2]
            z0_minus_0_03 = torch.clamp(z0 - 0.03, min=0.0)
            sqrt_term = torch.sqrt(vz0**2 + 2 * a * z0_minus_0_03)

            # times to hit the floor
            t1 = (-vz0 + sqrt_term) / -a
            t2 = (-vz0 - sqrt_term) / -a
            tz = torch.max(t1, t2)  # shape: (batch,)
            tz = torch.clamp(tz, min=0.0)

            # build a time-grid [steps x batch]
            steps = 100
            frac = torch.linspace(0, 1, steps=steps, device=self.device).unsqueeze(1)  # [steps, 1]
            time_tensor = frac * tz.unsqueeze(0)  # [steps, batch]

            # compute trajectories at each time
            new_x = ball_data[:, 0].unsqueeze(0) + vx0.unsqueeze(0) * time_tensor
            new_y = ball_data[:, 1].unsqueeze(0) + vy0.unsqueeze(0) * time_tensor
            new_z = (ball_data[:, 2].unsqueeze(0) +
                    vz0.unsqueeze(0) * time_tensor -
                    0.5 * a * time_tensor**2)

            # normalized distance to target
            distance_command = self.throwing_commands[env_ids, 0].unsqueeze(0)  # [1, batch]
            disp_matrix = torch.sqrt(
                (new_x - target_positions[:, 0].unsqueeze(0))**2 +
                (new_y - target_positions[:, 1].unsqueeze(0))**2 +
                (new_z - target_positions[:, 2].unsqueeze(0))**2
            ) / distance_command
            disp_matrix = torch.clamp(disp_matrix, max=1.0)

            # find minimum displacement and its index along the time axis
            disp_min, idx_min = torch.amin(disp_matrix, dim=0, keepdim=False), torch.argmin(disp_matrix, dim=0)

            # gather the corresponding time for each env
            batch_idx = torch.arange(len(env_ids), device=self.device)
            time_at_min = time_tensor[idx_min, batch_idx]  # [batch,]

        #print(torch.mean(time_at_min), torch.amin(time_at_min), torch.amax(time_at_min), torch.std(time_at_min))
        return disp_min, time_at_min

    
    def initialise_target_for_rendering(self, env_ids):
        target_default_state = self.target_object.data.default_root_state.clone()[env_ids]
        target_default_state[:, 7:] = torch.zeros_like(self.target_object.data.default_root_state[env_ids, 7:])
        target_default_state[:, 0:3] += self.target_positions[env_ids] + self._terrain.env_origins[env_ids] 
        target_default_state[:,:2] += self.default_root_states[env_ids,:2] #self.scene.env_origins[env_ids]

        # Compute the direction vector from current position to target position (offset -1,0,0.3)
        direction_to_target = self._robot.data.default_root_state[env_ids, :3] - (self.target_positions[env_ids]) 

        # Assuming the forward vector of the object is [1, 0, 0]
        forward_vector = torch.tensor([[1, 0, 0]], device=target_default_state.device).expand(direction_to_target.size(0), -1).float()

        direction_to_target = F.normalize(direction_to_target, p=2, dim=1)

        # Calculate the cross product to get the axis of rotation
        axis_of_rotation = torch.cross(forward_vector, direction_to_target, dim=1)

        # Calculate the angle between the forward vector and the direction to the target
        dot_product = (forward_vector * direction_to_target).sum(dim=1, keepdim=True)
        # Ensure angle has shape [batch_size, 1]
        angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))  # Shape: [batch_size, 1]

        # Apply sin to the angle and ensure it has the correct shape
        sin_half_angle = torch.sin(angle / 2)  # Shape: [batch_size, 1]

        # Multiply the axis of rotation by sin(angle/2) (element-wise)
        xyz = axis_of_rotation * sin_half_angle  # Shape: [batch_size, 3]

        # w is cos(angle/2), ensuring it has shape [batch_size, 1]
        w = torch.cos(angle / 2)  # Shape: [batch_size, 1]

        # Now concatenate w and xyz to form the quaternion, which will have shape [batch_size, 4]
        quaternion = torch.cat([w, xyz], dim=1)  # Shape: [batch_size, 4]
        current_quaternion = target_default_state[:, 3:7]
        # Now apply the quaternion to rotate the object (this depends on how your system applies rotations)
        # You may want to convert the quaternion into a rotation matrix for further usage
        # Function to multiply two quaternions
        def quaternion_multiply(q1, q2):
            w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
            w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
            
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return torch.stack([w, x, y, z], dim=1)

        # Update the quaternion by multiplying the current one with the new rotation quaternion
        new_quaternion = quaternion_multiply(current_quaternion, quaternion)

        # Set the updated orientation back into the default state
        target_default_state[:, 3:7] = new_quaternion
        self.target_object.write_root_state_to_sim(target_default_state, env_ids)