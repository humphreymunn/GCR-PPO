# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--use_critic_multi", action="store_true", default=False)
parser.add_argument("--architecture", type=str, default=None, help="Architecture of the RL agent. Specified as '56,56' etc. for actor and critic.")
# arguments for custom multi-objective experiments:
parser.add_argument("--energy_rew", type=int, default=-1, help="Use energy reward.")
parser.add_argument("--gait_rew", type=int, default=-1, help="Use gait reward.")
parser.add_argument("--baseh_rew", type=int, default=-1, help="Use base height reward.")
parser.add_argument("--armsw_rew", type=int, default=-1, help="Use step length reward.")
parser.add_argument("--armsp_rew", type=int, default=-1, help="Use arm span reward.")
parser.add_argument("--kneelft_rew", type=int, default=-1, help="Use knee lift reward.")
parser.add_argument("--ballrel_rew", type=int, default=-1, help="Use ball release height reward.")
parser.add_argument("--bodymo_rew", type=int, default=-1, help="Use body motion reward.")
parser.add_argument("--lftarm_rew", type=int, default=-1, help="Use left arm extension reward.")
parser.add_argument("--rgtarmrel_rew", type=int, default=-1, help="Use right arm extension reward.")
parser.add_argument("--hiporr_rew", type=int, default=-1, help="Use hip orientation reward.")
parser.add_argument("--armswlft_rew", type=int, default=-1, help="Use left arm swing reward.")
parser.add_argument("--armswrgt_rew", type=int, default=-1, help="Use right arm swing reward.")
parser.add_argument("--stephlft_rew", type=int, default=-1, help="Use left step height reward.")
parser.add_argument("--stephrgt_rew", type=int, default=-1, help="Use right step height reward.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.architecture is not None:
        agent_cfg.policy.actor_hidden_dims = [int(x) for x in args_cli.architecture.split(",")]
        agent_cfg.policy.critic_hidden_dims = [int(x) for x in args_cli.architecture.split(",")]
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    if "Humanoid" in args_cli.task:
        env_cfg.energy_rew = args_cli.energy_rew != -1 or "energy" in log_dir
        env_cfg.gait_rew = args_cli.gait_rew != -1 or "gait" in log_dir
        env_cfg.baseh_rew = args_cli.baseh_rew != -1 or "baseh" in log_dir
        env_cfg.armsw_rew = args_cli.armsw_rew != -1 or "armsw" in log_dir
        env_cfg.armsp_rew = args_cli.armsp_rew != -1 or "armsp" in log_dir
        env_cfg.kneelft_rew = args_cli.kneelft_rew != -1 or "kneelft" in log_dir
    if "Throwing" in args_cli.task:
        env_cfg.baseh_rew = args_cli.baseh_rew != -1 or "baseh" in log_dir
        env_cfg.energy_rew = args_cli.energy_rew != -1 or "energy" in log_dir
        env_cfg.ballrel_rew = args_cli.ballrel_rew != -1 or "ballrel" in log_dir
        env_cfg.bodymo_rew = args_cli.bodymo_rew != -1 or "bodymo" in log_dir
        env_cfg.lftarm_rew = args_cli.lftarm_rew != -1 or "lftarm" in log_dir
        env_cfg.rgtarmrel_rew = args_cli.rgtarmrel_rew != -1 or "rgtarmrel" in log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.unwrapped.distance_range = [1.,5.] # for throwing env evaluation
    
    # update multi_objective envs
    FIELDS = {
        "energy_rew":    ("energy_rew_vec",    "energy"),
        "gait_rew":      ("gait_rew_vec",      "gait"),
        "baseh_rew":     ("baseh_rew_vec",     "baseh"),
        "armsw_rew":     ("armsw_rew_vec",     "armsw"),
        "armsp_rew":     ("armsp_rew_vec",     "armsp"),
        "kneelft_rew":   ("kneelft_rew_vec",   "kneelft"),
        "ballrel_rew":   ("ballrel_rew_vec",   "ballrel"),
        "bodymo_rew":    ("bodymo_rew_vec",    "bodymo"),
        "lftarm_rew":    ("lftarm_rew_vec",    "lftarm"),
        "rgtarmrel_rew": ("rgtarmrel_rew_vec", "rgtarmrel"),
        "hiporr_rew":    ("hiporr_rew_vec",    "hiporr"),
        "armswlft_rew":  ("armswlft_rew_vec",  "armswlft"),
        "armswrgt_rew":  ("armswrgt_rew_vec",  "armswrgt"),
        "stephlft_rew":  ("stephlft_rew_vec",  "stephlft"),
        "stephrgt_rew":  ("stephrgt_rew_vec",  "stephrgt"),
    }

    for arg_name, (env_attr, suffix) in FIELDS.items():
        val = getattr(args_cli, arg_name, -1)
        if val != -1:
            vec = torch.full(
                (env_cfg.scene.num_envs,),
                float(val),
                dtype=torch.float32,
                device=agent_cfg.device,
            )
            setattr(env.unwrapped, env_attr, vec)
    
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, multihead=args_cli.use_critic_multi)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    total_reward = torch.zeros(env.num_envs, device=env.device)
    episode_rewards = []
    episodes_completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # simulate environment
    start_time_global = time.time()
    
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)
            
            # accumulate rewards for each environment
            total_reward += rewards.sum(dim=1)
            
            # check for episode completion
            if dones.any():
                # log rewards for completed episodes
                completed_envs = dones.nonzero(as_tuple=True)[0]
                for env_idx in completed_envs:
                    if not episodes_completed[env_idx]:
                        episode_rewards.append(total_reward[env_idx].item())
                        episodes_completed[env_idx] = True
                
                # check if all environments have completed at least one episode
                if episodes_completed.all():
                    avg_reward = sum(episode_rewards) / len(episode_rewards)
                    print(f"[INFO] All {env.num_envs} environments completed one episode. Average reward: {avg_reward:.4f}")
                    #break
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # print final reward statistics
    if episode_rewards:
        print(f"[INFO] Final average episode reward: {sum(episode_rewards) / len(episode_rewards):.4f}")
        print(f"[INFO] Total episodes completed: {len(episode_rewards)}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
