import os
import sys
import argparse
import subprocess
import pandas as pd
import re
from tqdm import tqdm
from datetime import datetime, date

folder_to_task = {
    #"shadow_hand": "Isaac-Repose-Cube-Shadow-Direct-v0",
    #"humanoid": "Isaac-Humanoid-v0",
    #"g1_rough": "Isaac-Velocity-Rough-G1-v0",
    #"ant": "Isaac-Ant-v0",
    #"allegro_cube": "Isaac-Repose-Cube-Allegro-v0",
    #"franka_open_drawer": "Isaac-Open-Drawer-Franka-v0",
    #"franka_lift": "Isaac-Lift-Cube-Franka-v0",
    #"reach_ur10": "Isaac-Reach-UR10-v0",
    #"franka_reach": "Isaac-Reach-Franka-v0",
    #"unitree_go2_rough": "Isaac-Velocity-Rough-Unitree-Go2-v0",
    #"quadcopter_direct": "Isaac-Quadcopter-Direct-v0",
    #"h1_rough": "Isaac-Velocity-Rough-H1-v0",
}

# Parse optional task index argument (0-11)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_index",
    type=int,
    default=None,
    help="Index of task to run (0-11). If not provided, runs all tasks."
)
args = parser.parse_args()

ordered_folders = list(folder_to_task.keys())

log_root = "logs/rsl_rl"
output_csv = "eval_results_all_entropy_new_baseline.csv"
write_header = not os.path.exists(output_csv)

MIN_DATE_STR = "2025-08-20"
MIN_DATE = datetime.strptime(MIN_DATE_STR, "%Y-%m-%d").date()

# Determine which task directories to process
if args.task_index is not None:
    if not (0 <= args.task_index < len(ordered_folders)):
        print(f"[ERROR] --task_index must be in [0, {len(ordered_folders)-1}]")
        sys.exit(1)
    selected_folder = ordered_folders[args.task_index]
    selected_path = os.path.join(log_root, selected_folder)
    if not os.path.isdir(selected_path):
        print(f"[INFO] Task folder '{selected_folder}' not found under {log_root}.")
        sys.exit(0)
    task_dirs = [selected_folder]
else:
    task_dirs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]

for folder_name in task_dirs:
    if folder_name not in folder_to_task:
        print(f"[WARNING] Folder '{folder_name}' not in known task mapping. Skipping.")
        continue

    task_name = folder_to_task[folder_name]
    task_path = os.path.join(log_root, folder_name)
    all_run_dirs = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]

    filtered_runs = []
    for run in all_run_dirs:
        date_token = run.split('_', 1)[0]
        try:
            run_date = datetime.strptime(date_token, "%Y-%m-%d").date()
        except ValueError:
            continue
        if run_date >= MIN_DATE:
            filtered_runs.append(run)

    if not filtered_runs:
        print(f"[INFO] No runs on/after {MIN_DATE_STR} for task {task_name}")
        continue

    # Sort newest first and keep only the most recent 10
    run_dirs = sorted(filtered_runs, reverse=True)  # [:10]

    for run in tqdm(run_dirs, desc=f"Evaluating {task_name}"):
        parts = run.split('_')
        if len(parts) < 3 or not parts[-1].isdigit():
            print(f"[WARNING] Skipping unrecognized run name format: {run}")
            continue

        seed = int(parts[-1])
        is_pcgrad = "pcgrad" in run
        method = "pcgrad" if is_pcgrad else "baseline"

        cmd = [
            "./isaaclab.sh",
            "-p", "scripts/reinforcement_learning/rsl_rl/play.py",
            f"--task={task_name}",
            "--num_envs", "1024",
            "--headless",
            "--load_run", run,
        ]
        if is_pcgrad:
            cmd.append("--use_critic_multi")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600
            )
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to run {run}: {e}")
            continue
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Run {run} took too long.")
            continue

        # Combine stdout and stderr and try multiple patterns to robustly extract reward
        output_text = f"{stdout}\n{stderr}" if stderr else stdout
        reward = None
        patterns = [
            r"Final\s+average\s+episode\s+reward[:\s]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"Average\s+episode\s+reward[:\s]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"Final\s+average\s+reward[:\s]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"Average\s+reward[:\s]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, output_text, flags=re.IGNORECASE)
            if m:
                try:
                    reward = float(m.group(1))
                    break
                except ValueError:
                    continue
        if reward is None:
            reward_lines = [line for line in output_text.splitlines() if "reward" in line.lower()]
            if reward_lines:
                nums = re.findall(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", reward_lines[-1])
                if nums:
                    try:
                        reward = float(nums[-1])
                    except ValueError:
                        pass
        if reward is None:
            print(f"[WARNING] Reward not found for run: {run}")

        row = {"seed": seed, "reward": reward, "name": run, "method": method, "task": task_name}
        pd.DataFrame([row], columns=["seed", "reward", "name", "method", "task"]).to_csv(
            output_csv,
            mode="a",
            header=write_header,
            index=False
        )
        write_header = False

print(f"Done. Results written to {output_csv}.")
