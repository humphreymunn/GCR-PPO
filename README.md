# GCR-PPO for IsaacLab

![GCR-PPO](docs/examples_result_comparison-1.pdf)

**GCR-PPO** is a modification of PPO for multi-objective robot RL that:
- uses a **multi-head critic** to obtain *per-reward* advantages and gradients,
- applies **priority-aware gradient surgery** (PCGrad-style projection) to protect *task* objectives from *regularisers*,
- runs at **massively parallel GPU** scale within IsaacLab/RSL-RL.

This repo is a focused fork/adaptation of [IsaacLab](https://github.com/isaac-sim/IsaacLab) targeting **IsaacLab 2.1.0** with **RSL-RL**.

---

## Quick Start

### Requirements
- Isaac Sim compatible with **IsaacLab 2.1.0**
- Python 3.10
- CUDA-capable GPU

### Installation

1) **Clone** this repository (assumes you have IsaacLab set up per their docs).
2) **Remove** the IsaacLab-packaged RSL-RL wheel:
```bash
./isaaclab.sh -p -m pip uninstall rsl-rl-lib
```
3) **Install** the local RSL-RL fork:
```bash
./isaaclab.sh -p -m pip install -e rsl_rl
```

## HOW TO RUN

### Baselines and Variants
#### GCR-PPO (multi-head critic + priority-aware PCGrad):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Humanoid-v0 \
  --num_envs 4096 --seed 0 --headless \
  --use_critic_multi --use_pcgrad
```
#### Multi-head critic only (no conflict resolution):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Humanoid-v0 \
  --num_envs 4096 --seed 0 --headless \
  --use_critic_multi
```

### Multi-Objective Experiments
Supported example tasks:

Isaac-Humanoid-Direct-v0 (multi-objective humanoid running)

Throwing-G1-General (full-body throwing)

Example (set a style objective, e.g., lower base height):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Humanoid-Direct-v0 \
  --num_envs 4096 --seed 0 --headless \
  --use_critic_multi --use_pcgrad \
  --baseh 0
```

See scripts/reinforcement_learning/rsl_rl/train.py for available flags
(e.g., --energy, --gait, --armsp, etc.) and their ranges.

### Adding New Tasks

#### Direct (single-class) task
In your task’s __init__:
```python
self.reward_components = len(self._episode_sums.keys())
self.reward_component_names = list(self._episode_sums.keys())
# Names of the *task* rewards (vs. regularisers). Used for priority.
self.reward_component_task_rew = ["names", "of", "task", "rewards"]
```

#### Manager (cfg-driven) task
In your task cfg’s __post_init__:
```python
self.reward_components = sum(
    isinstance(getattr(self.rewards, attr), RewTerm)
    for attr in dir(self.rewards)
    if not attr.startswith("__")
)
# All component names
self.reward_component_names = [
    attr for attr in dir(self.rewards)
    if isinstance(getattr(self.rewards, attr), RewTerm) and not attr.startswith("__")
]
# Subset that are *task* rewards (priority > regularisers)
self.reward_component_task_rew = ["names", "of", "task", "rewards"]
```
The reward_component_task_rew list controls priority during gradient surgery:
task components are protected from being weakened by regularisers.

## Paper
Methodology, analysis, and results are described in our paper:

We include:
 - comparisons against massively parallel GPU PPO across 13 IsaacLab tasks,
 - two custom multi-objective suites (Humanoid Running, Full-Body Throwing),
 - ablations (multi-head only vs. GCR) and conflict–performance analyses.

![Method Overview](docs/actor_critic_pcgrad_method-3-1.pdf)

### Citation

If you use this code, please cite our work:

Also cite IsaacLab/Orbit, on which this repository is based:

@article{mittal2023orbit,
   author  = {Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal = {IEEE Robotics and Automation Letters},
   title   = {Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year    = {2023},
   volume  = {8},
   number  = {6},
   pages   = {3740-3747},
   doi     = {10.1109/LRA.2023.3270034}
}

## Acknowledgements
 - This project builds upon IsaacLab (NVIDIA) and the Orbit framework.
 - Thanks to the IsaacLab and RSL-RL communities for their libraries, examples, and documentation.
