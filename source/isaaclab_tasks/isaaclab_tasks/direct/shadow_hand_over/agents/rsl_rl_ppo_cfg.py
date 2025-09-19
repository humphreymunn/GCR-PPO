# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ShadowHandoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16                       # Matches horizon_length
    max_iterations = 5000                        # Matches max_epochs
    save_interval = 200                          # Matches save_frequency
    experiment_name = "shadow_handover"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                      # Can map to policy noise if not learned
        actor_hidden_dims=[512, 512, 256, 128],  # Matches MLP units
        critic_hidden_dims=[512, 512, 256, 128], # Matching actor unless known to differ
        activation="elu",                        # Matches activation: elu
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=4.0,                     # Matches critic_coef
        use_clipped_value_loss=True,             # Matches clip_value: True
        clip_param=0.2,                          # Matches e_clip
        entropy_coef=0.0,                        # Matches entropy_coef
        num_learning_epochs=5,                   # Matches mini_epochs
        num_mini_batches=4,      # You'll need to set X = num_envs, or hardcode 4
        learning_rate=5e-4,                      # Matches learning_rate
        schedule="adaptive",                     # Matches schedule_type: standard, lr_schedule: adaptive
        gamma=0.99,                              # Matches gamma
        lam=0.95,                                 # Matches tau
        desired_kl=0.016,                        # Matches kl_threshold
        max_grad_norm=1.0,                       # Matches grad_norm
    )