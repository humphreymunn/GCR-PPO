# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AutomatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  # matches horizon_length
    max_iterations = 200  # matches max_epochs
    save_interval = 10  # matches save_frequency
    experiment_name = "automate"  # from full_experiment_name
    empirical_normalization = False  # using normalize_input/value manually below

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # default; not explicitly given
        actor_hidden_dims=[512, 128, 64],  # from model.network.mlp.units
        critic_hidden_dims=[512, 128, 64],  # same as actor here
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # from critic_coef
        use_clipped_value_loss=True,  # from clip_value
        clip_param=0.2,  # from e_clip
        entropy_coef=0.0,  # from entropy_coef
        num_learning_epochs=4,  # from mini_epochs
        num_mini_batches=128 * 128 // 512,  # =32; from minibatch_size
        learning_rate=1.0e-4,
        schedule="adaptive",  # from lr_schedule
        gamma=0.995,
        lam=0.95,  # tau
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
