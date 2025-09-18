# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from unittest import runner

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import random

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--use_critic_multi", action="store_true", default=False)
parser.add_argument("--use_pcgrad", action="store_true", default=False, help="Use PCGrad for multi-head training.")
parser.add_argument("--use_pcgrad_full", action="store_true", default=False, help="Use full PCGrad for multi-head training.")
parser.add_argument("--entropy_coef", type=float, default=None, help="Entropy coefficient for the PPO algorithm.")
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

parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl_rl")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.use_pcgrad or args_cli.use_pcgrad_full:
        # tuned entropy values for pgrad (baseline tuned values updated in cfg files)
        entropy_coef_map = {
            "Ant": 0.008446,
            "Humanoid": 0.004481,
            "Lift-Cube": 0.015918,
            "Drawer": 0.002378,
            "Quadcopter": 0.015918,
            "Reach-Franka": 0.015918,
            "Reach-Ur10": 0.008446,
            "Rough-G1": 0.004481,
            "Rough-H1": 0.008446,
            "LocoManip-Digit": 0.015918,
            "Cube-Allegro": 0.002378,
            "Cube-Shadow": 0.002378,
            "Rough-Unitree-Go2": 0.008446,
            "Throwing": 0.00066943,
        }
        if args_cli.task in entropy_coef_map:
            agent_cfg.algorithm.entropy_coef = entropy_coef_map[args_cli.task]
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    log_dir += f"_multi" if args_cli.use_critic_multi else ""
    log_dir += f"_pcgrad" if args_cli.use_pcgrad else ""
    log_dir += f"_pcgrad_full" if args_cli.use_pcgrad_full else ""
    log_dir += f"_{args_cli.entropy_coef}" if args_cli.entropy_coef is not None else ""
    log_dir += f"_{args_cli.architecture}" if args_cli.architecture is not None else ""
    log_dir += f"_{args_cli.seed}" if args_cli.seed is not None else ""
    
    # Set reward flags in a sub-config dictionary for maintainability
    if not hasattr(env_cfg, "rewards") or env_cfg.rewards is None:
        env_cfg.rewards = {}

    # set custom reward flags for humanoid and throwing envs
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

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
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
            log_dir += f"_{suffix}{int(vec[0].item())}"

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, multihead=args_cli.use_critic_multi, pcgrad=args_cli.use_pcgrad, pcgradfull=args_cli.use_pcgrad_full)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    if not args_cli.use_critic_multi:
        # run training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    else:
        runner.learn_multi(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
