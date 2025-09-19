# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg


@configclass
class FactoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 2  # matches horizon_length
    max_iterations = 200  # matches max_epochs
    save_interval = 5  # matches save_frequency
    experiment_name = "factory"  # from full_experiment_name
    empirical_normalization = False  # using normalize_input/value manually below

    policy = RslRlPpoActorCriticRecurrentCfg(
        rnn_type="lstm",
        rnn_hidden_dim=1024,
        rnn_num_layers=2,
        init_noise_std=1.0,  # default; not explicitly given
        actor_hidden_dims=[512, 128, 64],
        critic_hidden_dims=[512, 128, 64],  # same as actor here
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,  # from critic_coef
        use_clipped_value_loss=True,  # from clip_value
        clip_param=0.2,  # from e_clip
        entropy_coef=0.0,  # from entropy_coef
        num_learning_epochs=5,  # from mini_epochs
        num_mini_batches=16,
        learning_rate=1.0e-4,
        schedule="adaptive",  # from lr_schedule
        gamma=0.99,
        lam=0.95,  # tau
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
