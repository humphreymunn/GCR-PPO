# GCR-PPO
GCR-PPO is a modification of PPO for multi-objective robot RL that: - uses a multi-head critic to obtain per-reward advantages and gradients, - applies priority-aware gradient surgery (PCGrad-style projection) to protect task objectives from regularisers, - runs at massively parallel GPU scale within IsaacLab/RSL-RL.
