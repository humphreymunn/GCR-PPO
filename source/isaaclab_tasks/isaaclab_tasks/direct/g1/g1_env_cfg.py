# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import G1_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class G1FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 37 
    observation_space = 100 #?
    state_space = 0
    training_method = False
    context_type = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    
    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 1.0
    feet_air_time_reward_scale = 0.75
    termination_reward_scale = -200
    feet_slide_reward_scale = -0.1
    dof_pos_limits_reward_scale = -1.0
    joint_deviation_hip_reward_scale = -0.1
    joint_deviation_arms_reward_scale = -0.1
    joint_deviation_torso_reward_scale = -0.1
    joint_deviation_fingers_reward_scale = -0.05
    lin_vel_z_l2_reward_scale = -0.2
    ang_vel_xy_l2_reward_scale = -0.05
    dof_torques_l2_reward_scale = -2.0e-6
    dof_acc_l2_reward_scale = -1.0e-7
    action_rate_l2_reward_scale = -0.005
    flat_orientation_reward_scale = -1.0
    feet_air_time_threshold = 0.4

    reward_components = 16
    reward_component_names = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time", "termination","feet_slide",
                              "dof_pos_limits","joint_deviation_hip","joint_deviation_arms","joint_deviation_torso","joint_deviation_fingers",
                              "lin_vel_z_l2","ang_vel_xy_l2","dof_torques_l2","dof_acc_l2","action_rate_l2", "flat_orientation_l2"]
    # only used
    reward_component_task_rew = ["termination", "track_lin_vel_xy_exp", "track_ang_vel_z_exp"]

@configclass
class G1RoughEnvCfg(G1FlatEnvCfg):
    # env
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    

    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 2.0
    feet_air_time_reward_scale = 0.25
    termination_reward_scale = -200
    feet_slide_reward_scale = -0.1
    dof_pos_limits_reward_scale = -1.0
    joint_deviation_hip_reward_scale = -0.1
    joint_deviation_arms_reward_scale = -0.1
    joint_deviation_torso_reward_scale = -0.1
    joint_deviation_fingers_reward_scale = -0.05
    lin_vel_z_l2_reward_scale = 0.
    ang_vel_xy_l2_reward_scale = -0.05
    dof_torques_l2_reward_scale = -1.5e-7
    dof_acc_l2_reward_scale = -1.25e-7
    action_rate_l2_reward_scale = -0.01
    flat_orientation_reward_scale = -1.0
    feet_air_time_threshold = 0.6

    reward_components = 15
    reward_component_names = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time", "termination","feet_slide",
                              "dof_pos_limits","joint_deviation_hip","joint_deviation_arms","joint_deviation_torso","joint_deviation_fingers",
                               "ang_vel_xy_l2","dof_torques_l2","dof_acc_l2","action_rate_l2", "flat_orientation_l2"]
    # only used
    reward_component_task_rew = ["termination", "track_lin_vel_xy_exp", "track_ang_vel_z_exp"]

