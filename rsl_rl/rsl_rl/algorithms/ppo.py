#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

import copy
import random
import math

class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        reward_component_names=None,
        reward_component_task_rew=None,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.reward_component_names = reward_component_names
        self.reward_component_task_rew = reward_component_task_rew

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, reward_components):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, reward_components, self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device)
            
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_returns_multi(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns_multi(last_values, self.gamma, self.lam)

    @torch.no_grad()
    def apply_pcgrad(self, objective_grads, full: bool = False):
        num_objectives = len(objective_grads)
        eps = 1e-8

        # Flatten per-objective grads: [m, D]
        flat_grads = torch.stack([
            torch.cat([g.flatten() for g in grad_list])
            for grad_list in objective_grads
        ])  # [num_objectives, D]

        device = flat_grads.device
        projected_grads = flat_grads.clone()

        # Task mask (aligned with reward order)
        if full: # if no priority, set all objectives to 'task'
            is_task = torch.tensor(
            [True for _ in self.reward_component_names],
            device=device, dtype=torch.bool
            )

        else:
            is_task = torch.tensor(
                [name in self.reward_component_task_rew for name in self.reward_component_names],
                device=device, dtype=torch.bool
            )

        # Non-deterministic order for i
        perm_i = torch.randperm(num_objectives, device=device)
        for i in perm_i.tolist():
            gi = projected_grads[i]

            # Non-deterministic order for j each time
            perm_j = torch.randperm(num_objectives, device=device).tolist()

            if is_task[i]:
                # Task–task: project task i against other tasks
                for j in perm_j:
                    if j == i or not is_task[j]:
                        continue
                    gj = projected_grads[j]
                    dot = torch.dot(gi, gj)
                    nt2 = torch.dot(gj, gj)
                    if (dot < 0.0) and (nt2 > 1e-12):
                        gi = gi - (dot / (nt2 + eps)) * gj
                projected_grads[i] = gi
            else:
                # Penalty–task: project penalty i against tasks only
                for j in perm_j:
                    if not is_task[j]:
                        continue
                    gj = projected_grads[j]
                    dot = torch.dot(gi, gj)
                    nt2 = torch.dot(gj, gj)
                    if (dot < 0.0) and (nt2 > 1e-12):
                        gi = gi - (dot / (nt2 + eps)) * gj
                projected_grads[i] = gi

        # Combine grads (sum semantics)
        combined_flat = projected_grads.sum(dim=0)

        # Unflatten back to param shapes
        combined = []
        start = 0
        for g in objective_grads[0]:
            n = g.numel()
            combined.append(combined_flat[start:start + n].view_as(g))
            start += n
        return combined

    '''
    Copy of update(), but with multi-head critic support and pcgrad option.
    '''
    def update_multihead(self, pcgrad=False, pcgradfull=False):
        mean_value_loss = 0
        mean_component_value_loss = torch.zeros((self.actor_critic.num_reward_components), device=self.device)
        mean_surrogate_loss = 0
        mean_entropy = 0.0
        mean_approx_kl = 0.0
        mean_clip_fraction = 0.0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            component_advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            mean_entropy += entropy_batch.mean().item()

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    mean_approx_kl += kl.mean().item()
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            log_ratio = (actions_log_prob_batch - old_actions_log_prob_batch.squeeze()).clamp(-20, 20)
            ratio = log_ratio.exp().unsqueeze(1)
            
            clip_fraction = ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)).float().mean()
            mean_clip_fraction += clip_fraction.item()

            component_advantages_batch = component_advantages_batch  # shape: [B, C]

            surrogate_per_component = -component_advantages_batch * ratio  # shape: [B, C]
            surrogate_per_component_clipped = -component_advantages_batch * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )

            component_surrogate_losses = torch.max(
                surrogate_per_component, surrogate_per_component_clipped
            )  # [B, C]

            mean_component_surrogate_loss = component_surrogate_losses.mean(dim=0)  # [C]

            surrogate_loss = mean_component_surrogate_loss.sum()
            
            # Value function loss
            if self.use_clipped_value_loss:

                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2) 
                value_losses_clipped = (value_clipped - returns_batch).pow(2) 
                dim_use = 2 if len(value_losses.shape) > 2 else 1
                value_loss = torch.max(value_losses.sum(dim=dim_use), value_losses_clipped.sum(dim=dim_use)).mean()
                component_value_loss = torch.max(value_losses, value_losses_clipped).mean(dim=dim_use-1)  # shape: [C]
                if dim_use == 2:
                    mean_component_value_loss += component_value_loss.sum(dim=0).detach()
                else:
                    mean_component_value_loss += component_value_loss.detach()
            else:
                assert False # not implemented
                value_loss = (returns_batch - value_batch).pow(2).sum(dim=2).mean()
                mean_component_value_loss += (returns_batch - value_batch).pow(2).sum(dim=2).mean(dim=0).mean(dim=0)

            # More efficient per-component gradient computation
            self.optimizer.zero_grad()

            self.current_losses = mean_component_surrogate_loss.detach()

            # Compute gradients for all components in one backward pass
            # Create a tensor of ones for each component to compute gradients
            if not pcgrad and not pcgradfull:
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                loss.backward()
            else: # pcgrad
                num_components = mean_component_surrogate_loss.size(0)
                component_grads = []
                
                for i in range(num_components):
                    grads = torch.autograd.grad(
                        outputs=mean_component_surrogate_loss[i],
                        inputs=self.actor_critic.parameters(),
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )
                    # Convert None gradients to zero tensors
                    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, self.actor_critic.parameters())]
                    component_grads.append(grads)
                
                # Apply PCGrad
                projected_grads = self.apply_pcgrad(component_grads, full=pcgradfull)

                # Assign PCGrad-updated gradients to model
                for param, g in zip(self.actor_critic.parameters(), projected_grads):
                    if param.requires_grad:
                        param.grad = g

                # Value loss + entropy (backward separately and add)
                value_loss.backward(retain_graph=True)
                if self.entropy_coef > 0:
                    (-self.entropy_coef * entropy_batch.mean()).backward()

            actor_critic_params = list(self.actor_critic.parameters())
            nn.utils.clip_grad_norm_(actor_critic_params, self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_entropy /= num_updates
        mean_approx_kl /= num_updates
        mean_clip_fraction /= num_updates
        mean_value_loss /= num_updates
        mean_component_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        def _flatten_grads(grads_list):
            return torch.stack([torch.cat([g.view(-1) for g in grads])
                                for grads in grads_list])  # [C, D]

        grad_overall_conflict_pct = None
        if pcgrad or pcgradfull:
            G = _flatten_grads(component_grads).detach()          # [C, D]
            # Pairwise dot matrix
            D = G @ G.t()                                         # [C, C]
            # Only count each pair once (upper triangle, no diagonal)
            C = G.size(0)
            iu = torch.triu_indices(C, C, offset=1, device=G.device)
            # Angle > 90°  <=> dot < 0  (no need to compute acos)
            conflicts = (D[iu[0], iu[1]] < 0.0).float()           # [P], P=C*(C-1)/2
            grad_overall_conflict_pct = (conflicts.mean() * 100.0).item()

        def compute_grad_angles(grads_list):
            """Returns (angles, labels, pairs).
            angles: list[float] (degrees)
            labels: list[str]  e.g., 'rewardA_vs_rewardB'
            pairs:  list[tuple[int,int]] (i,j) indices
            """
            flat_grads = [torch.cat([g.view(-1) for g in grads]) for grads in grads_list]
            angles, labels, pairs = [], [], []

            # Prefer names from PPO; fallback to generic
            try:
                names = list(self.reward_component_names)
            except Exception:
                names = [f"C{i}" for i in range(len(flat_grads))]

            for i in range(len(flat_grads)):
                for j in range(i + 1, len(flat_grads)):
                    cos_sim = torch.nn.functional.cosine_similarity(flat_grads[i], flat_grads[j], dim=0).clamp(-1, 1)
                    angle = torch.acos(cos_sim) * (180 / torch.pi)
                    angles.append(angle.item())
                    labels.append(f"{names[i]}_vs_{names[j]}")
                    pairs.append((i, j))
            return angles, labels, pairs

        grad_angles, grad_angle_labels, grad_angle_pairs = None, None, None
        if pcgrad or pcgradfull:
            grad_angles, grad_angle_labels, grad_angle_pairs = compute_grad_angles(component_grads)

        return {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_component_value_loss": mean_component_value_loss,
            "mean_component_surrogate_loss": mean_component_surrogate_loss,
            "mean_entropy": entropy_batch.mean().item(),
            "mean_clip_fraction": (torch.logical_or(ratio < 1.0 - self.clip_param, ratio > 1.0 + self.clip_param).float().mean().item()),
            "mean_approx_kl": kl_mean.item() if self.desired_kl is not None else 0.0,
            "per_head_advantages": component_advantages_batch.mean(dim=0).detach().cpu(),
            "grad_norms": [
                torch.norm(torch.stack([torch.norm(g.detach()) for g in grads if g is not None]))
                for grads in component_grads
            ] if pcgrad or pcgradfull else None,
            "grad_angles": grad_angles,
            "grad_overall_conflict_pct": grad_overall_conflict_pct,
            "grad_angle_labels": grad_angle_labels,   # <— NEW
            "grad_angle_pairs": grad_angle_pairs,     # <— optional (indices)
            "grad_projection_magnitude": (sum((torch.norm(g.detach()) for g in projected_grads)) / len(projected_grads)) if pcgrad or pcgradfull else None,
            "action_magnitudes": actions_batch.abs().mean(dim=0).detach().cpu(),
        }
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0.0
        mean_approx_kl = 0.0
        mean_clip_fraction = 0.0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            component_advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            mean_entropy += entropy_batch.mean().item()

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    mean_approx_kl += kl.mean().item()
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            clip_fraction = ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param)).float().mean()
            mean_clip_fraction += clip_fraction.item()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                dim_use = 2 if len(value_losses.shape) > 2 else 1
                value_loss = torch.max(value_losses.sum(dim=dim_use), value_losses_clipped.sum(dim=dim_use)).mean()

            else:
                assert False # not implemented
                value_loss = (returns_batch - value_batch).pow(2).sum(dim=2).mean()
                mean_component_value_loss += (returns_batch - value_batch).pow(2).sum(dim=2).mean(dim=0).mean(dim=0)

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            self.optimizer.zero_grad()
            loss.backward()

            actor_critic_params = list(self.actor_critic.actor.parameters()) + list(self.actor_critic.critic.parameters())
            nn.utils.clip_grad_norm_(actor_critic_params, self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_entropy /= num_updates
        mean_approx_kl /= num_updates
        mean_clip_fraction /= num_updates
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return {
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_component_value_loss": None,
            "mean_component_surrogate_loss": None,
            "mean_entropy": entropy_batch.mean().item(),
            "mean_clip_fraction": (torch.logical_or(ratio < 1.0 - self.clip_param, ratio > 1.0 + self.clip_param).float().mean().item()),
            "mean_approx_kl": kl_mean.item() if self.desired_kl is not None else 0.0,
            "per_head_advantages": None,
            "grad_norms": None,
            "grad_projection_magnitude": None,
            "action_magnitudes": actions_batch.abs().mean(dim=0).detach().cpu(),
        }

