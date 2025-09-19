#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories


class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_reward_components,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        rnn_type="gru",
        concatenate_rnn_with_input=True,
        untrained=False,
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        init_noise_std=0.5,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        self.num_reward_components = num_reward_components

        if concatenate_rnn_with_input:
            # Adjust the input sizes for the actor and critic to include the RNN hidden size
            adjusted_num_actor_obs = num_actor_obs + rnn_hidden_dim
            adjusted_num_critic_obs = num_critic_obs + rnn_hidden_dim
        else:
            adjusted_num_actor_obs = rnn_hidden_dim
            adjusted_num_critic_obs = rnn_hidden_dim

        super().__init__(num_actor_obs=adjusted_num_actor_obs,
                         num_critic_obs=adjusted_num_critic_obs,
                         num_actions=num_actions,
                         num_reward_components=num_reward_components,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std)
        
        activation = get_activation(activation)

        
        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim, untrained=untrained)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim, untrained=untrained)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

        self.concatenate_rnn_with_input = concatenate_rnn_with_input
        self.untrained = untrained

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        if not self.concatenate_rnn_with_input:
            input_a = self.memory_a(observations, masks, hidden_states)
            return super().act(input_a.squeeze(0))
        rnn_output_a = self.memory_a(observations, masks, hidden_states).squeeze(0)
        batch_mode = masks is not None
        if batch_mode:
            observations_processed = unpad_trajectories(observations, masks)
            combined_input_a = torch.cat((observations_processed, rnn_output_a), dim=-1)
        else:
            combined_input_a = torch.cat((observations, rnn_output_a), dim=-1)
        return super().act(combined_input_a)


    def act_inference(self, observations):
        if not self.concatenate_rnn_with_input:
            input_a = self.memory_a(observations)
            return super().act_inference(input_a.squeeze(0))
        rnn_output_a = self.memory_a(observations).squeeze(0)
        combined_input_a = torch.cat((observations, rnn_output_a), dim=-1)
        return super().act_inference(combined_input_a)

    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        if not self.concatenate_rnn_with_input:
            input_c = self.memory_c(critic_observations, masks, hidden_states)
            return super().evaluate(input_c.squeeze(0))
        rnn_output_c = self.memory_c(critic_observations, masks, hidden_states).squeeze(0)
        batch_mode = masks is not None
        if batch_mode:
            critic_observations_processed = unpad_trajectories(critic_observations, masks)
            combined_input_c = torch.cat((critic_observations_processed, rnn_output_c), dim=-1)
        else:
            combined_input_c = torch.cat((critic_observations, rnn_output_c), dim=-1)

        return super().evaluate(combined_input_c)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256, untrained=False):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
        # Freeze the weights of the RNN layer
        if untrained:
            for param in self.rnn.parameters():
                param.requires_grad = False

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None, hidden_states=None):
        if dones is None:  # reset all hidden states
            if hidden_states is None:
                self.hidden_states = None
            else:
                self.hidden_states = hidden_states
        elif self.hidden_states is not None:  # reset hidden states of done environments
            if hidden_states is None:
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = 0.0
                else:
                    self.hidden_states[..., dones == 1, :] = 0.0
            else:
                NotImplementedError(
                    "Resetting hidden states of done environments with custom hidden states is not implemented"
                )

    def detach_hidden_states(self, dones=None):
        if self.hidden_states is not None:
            if dones is None:  # detach all hidden states
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    self.hidden_states = tuple(hidden_state.detach() for hidden_state in self.hidden_states)
                else:
                    self.hidden_states = self.hidden_states.detach()
            else:  # detach hidden states of done environments
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:
                    self.hidden_states[..., dones == 1, :] = self.hidden_states[..., dones == 1, :].detach()

