import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from utils import (
    RunningMoments,
    deep_idx,
    encode_samples,
    flat_size,
    get_agent_class,
    grad_norm,
    shuffled_within_classes,
    squash,
)


class RewardSignal(ABC):
    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        return self

    @abstractmethod
    def get_rewards(self, rollout) -> Dict[str, np.ndarray]:
        pass

    def update(self, rollout) -> None:
        pass

    def normalized(self, gamma=0.99):
        return NormalizationWrapper(self, gamma)


class NormalizationWrapper(RewardSignal):
    def __init__(self, wrapped_reward: RewardSignal, gamma=0.99):
        if isinstance(wrapped_reward, NormalizationWrapper):
            raise ValueError("a reward signal is being normalized twice")
        self.gamma = gamma
        self.wrapped_reward = wrapped_reward
        self.discounted_reward_totals = defaultdict(float)
        self.reward_stats = defaultdict(RunningMoments)
        self.logger = None

    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        self.logger = logger
        self.wrapped_reward.configured_with(obs_spaces, action_spaces, logger, device)
        return self

    def get_rewards(self, rollout) -> Dict[str, np.ndarray]:
        rewards = self.wrapped_reward.get_rewards(rollout)
        ret = {
            agent_id: np.zeros_like(rewards[agent_id]) for agent_id in rewards.keys()
        }
        for agent_id in rewards.keys():
            assert isinstance(agent_id, str)
            for i in range(len(rewards[agent_id])):
                self.discounted_reward_totals[agent_id] = (
                    self.gamma * self.discounted_reward_totals[agent_id]
                    + rewards[agent_id][i]
                )
                self.reward_stats[agent_id].push(
                    self.discounted_reward_totals[agent_id]
                )
                ret[agent_id][i] = rewards[agent_id][i] / (
                    self.reward_stats[agent_id].std() + 1e-8
                )
        return ret

    def update(self, rollout) -> None:
        self.wrapped_reward.update(rollout)

        if self.logger is not None:
            self.logger.log(
                {
                    f"reward_norm_mean_{agent_id}": self.reward_stats[agent_id].mean()
                    for agent_id in self.reward_stats.keys()
                }
            )
            self.logger.log(
                {
                    f"reward_norm_std_{agent_id}": self.reward_stats[agent_id].std()
                    for agent_id in self.reward_stats.keys()
                }
            )


# DreamerV3-style normalization
class StablyNormalized(RewardSignal):
    def __init__(self, wrapped_reward: RewardSignal, gamma=0.99, ema_decay=0.99):
        if isinstance(wrapped_reward, NormalizationWrapper):
            raise ValueError("a reward signal is being normalized twice")
        self.gamma = gamma
        self.wrapped_reward = wrapped_reward
        self.discounted_reward_totals = defaultdict(float)

        self.beta = ema_decay
        self.lo_ema = defaultdict(float)
        self.hi_ema = defaultdict(float)
        self.t = 0

        self.logger = None

    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        self.logger = logger
        self.wrapped_reward.configured_with(obs_spaces, action_spaces, logger, device)
        return self

    def get_rewards(self, rollout) -> Dict[str, np.ndarray]:
        rewards = self.wrapped_reward.get_rewards(rollout)
        ret = {
            agent_id: np.zeros_like(rewards[agent_id]) for agent_id in rewards.keys()
        }
        self.t += 1
        for agent_id in rewards.keys():
            assert isinstance(agent_id, str)
            batch_returns = []
            for i in range(len(rewards[agent_id])):
                self.discounted_reward_totals[agent_id] = (
                    self.gamma * self.discounted_reward_totals[agent_id]
                    + rewards[agent_id][i]
                )
                batch_returns.append(self.discounted_reward_totals[agent_id])

            self.lo_ema[agent_id] = self.beta * self.lo_ema[agent_id] + (
                1 - self.beta
            ) * np.percentile(batch_returns, 5)
            self.hi_ema[agent_id] = self.beta * self.hi_ema[agent_id] + (
                1 - self.beta
            ) * np.percentile(batch_returns, 95)
            ret[agent_id] = rewards[agent_id] / max(
                1,
                (self.hi_ema[agent_id] - self.lo_ema[agent_id])
                / (1 - self.beta**self.t),
            )

        return ret

    def update(self, rollout) -> None:
        self.wrapped_reward.update(rollout)

        if self.logger is not None:
            self.logger.log(
                {
                    f"reward_norm_scale_{agent_id}": max(
                        1,
                        (self.hi_ema[agent_id] - self.lo_ema[agent_id])
                        / (1 - self.beta**self.t),
                    )
                    for agent_id in self.lo_ema.keys()
                }
            )


class ExplicitReward(RewardSignal):
    def __init__(self):
        self.stats = defaultdict(RunningMoments)

    def get_rewards(self, rollout) -> Dict[str, np.ndarray]:
        for agent_id in rollout["env_rewards"].keys():
            for i in range(len(rollout["env_rewards"][agent_id])):
                self.stats[agent_id].push(float(rollout["env_rewards"][agent_id][i]))
        return rollout["env_rewards"]


# One centralized discriminator which takes in (all obs) + (all actions)
class FullyCentralizedGAILReward(RewardSignal, nn.Module):
    def __init__(
        self,
        demo_filename: str,
        lr: float,
        fc_dim=64,
        minibatch_size=128,
        demo_limit: int = None,
    ):
        nn.Module.__init__(self)
        self.fc_dim = fc_dim
        self.lr = lr
        self.minibatch_size = minibatch_size

        with open(demo_filename, "rb") as f:
            data = pickle.load(f)
        if demo_limit is None:
            self.demo_data = data
        else:
            self.demo_data = deep_idx(data, slice(demo_limit), copy=True)
        self.demo_size = self.demo_data["terminal"].shape[0]
        self.demo_idxs = np.arange(self.demo_size)
        np.random.shuffle(self.demo_idxs)
        self.ptr = 0

        self.obs_spaces = None
        self.action_spaces = None
        self.logger = None
        self.device = None

        self.model = None
        self.optim = None

    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        self.obs_spaces = obs_spaces
        self.action_spaces = action_spaces
        self.logger = logger
        self.device = device

        input_size = sum(
            flat_size(obs_space) for obs_space in obs_spaces.values()
        ) + sum(flat_size(action_space) for action_space in action_spaces.values())

        self.model = nn.Sequential(
            nn.Linear(input_size, self.fc_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(self.fc_dim, self.fc_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(self.fc_dim, 1, device=self.device),
            nn.Sigmoid(),
        )

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        return self

    def forward(self, all_obs, all_actions):
        model_inputs = []
        for agent_id in sorted(self.obs_spaces.keys()):
            model_inputs.append(all_obs[agent_id])
            model_inputs.append(all_actions[agent_id])
        return self.model.forward(torch.cat(model_inputs, dim=1)).reshape(-1)

    def get_rewards(self, rollout) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            rewards = (
                torch.log(self.forward(rollout["obs"], rollout["actions"]))
                .cpu()
                .numpy()
            )

            return {agent_id: rewards for agent_id in self.obs_spaces.keys()}

    def update(self, rollout) -> None:
        rollout_length = rollout["terminal"].shape[0]

        rollout_idxs = np.arange(rollout_length)
        np.random.shuffle(rollout_idxs)

        for i in range(rollout_length // self.minibatch_size):
            rollout_idx = rollout_idxs[
                i * self.minibatch_size : (i + 1) * self.minibatch_size
            ]
            if self.ptr + self.minibatch_size > self.demo_size:
                self.ptr = 0
                np.random.shuffle(self.demo_idxs)
            demo_idx = self.demo_idxs[self.ptr : self.ptr + self.minibatch_size]
            self.ptr += self.minibatch_size

            rollout_preds = self.forward(
                all_obs=deep_idx(rollout["obs"], rollout_idx),
                all_actions=deep_idx(rollout["actions"], rollout_idx),
            )
            expert_preds = self.forward(
                all_obs={
                    agent_id: encode_samples(
                        agent_obs, self.obs_spaces[agent_id], device=self.device
                    )
                    for agent_id, agent_obs in deep_idx(
                        self.demo_data["obs"], demo_idx
                    ).items()
                },
                all_actions={
                    agent_id: encode_samples(
                        agent_action, self.action_spaces[agent_id], device=self.device
                    )
                    for agent_id, agent_action in deep_idx(
                        self.demo_data["actions"], demo_idx
                    ).items()
                },
            )

            loss = -torch.mean(torch.log(1 - rollout_preds)) - torch.mean(
                torch.log(expert_preds)
            )

            if self.logger is not None:
                self.logger.push(discrim_rollout_pred=rollout_preds.mean().item())
                self.logger.push(discrim_expert_pred=expert_preds.mean().item())
                self.logger.push(discrim_loss=loss.item())

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()

        self.logger.step()


# Each agent has their own discriminator which takes in (all OTHER obs) + (own action)
class MixedGAILReward(RewardSignal, nn.Module):
    def __init__(
        self,
        demo_filename: str,
        lr: float,
        fc_dim=64,
        minibatch_size=128,
        share_parameters=True,
        demo_limit: int = None,
    ):
        nn.Module.__init__(self)
        self.fc_dim = fc_dim
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.share_parameters = share_parameters

        self.obs_spaces = None
        self.action_spaces = None
        self.agent_ids = None
        self.policy_ids = None
        self.pid = None

        self.discrims = None
        self.optims = None

        with open(demo_filename, "rb") as f:
            data = pickle.load(f)
        if demo_limit is None:
            self.demo_data = data
        else:
            self.demo_data = deep_idx(data, slice(demo_limit), copy=True)
        self.demo_size = self.demo_data["terminal"].shape[0]
        assert self.minibatch_size <= self.demo_size
        self.demo_idxs = np.arange(self.demo_size)
        np.random.shuffle(self.demo_idxs)
        self.ptr = 0

        self.logger = None
        self.device = None

    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        self.logger = logger
        self.device = device

        if self.share_parameters:
            self.obs_spaces = {}
            self.action_spaces = {}
            self.pid = {}
            self.policy_ids = []
            for agent_id in obs_spaces.keys():
                policy_id = get_agent_class(agent_id)

                self.pid[agent_id] = policy_id

                if policy_id in self.obs_spaces.keys():
                    if self.obs_spaces[policy_id] != obs_spaces[agent_id]:
                        raise ValueError(
                            f"multiple agents have {policy_id=} but have different "
                            f"observation spaces"
                        )
                else:
                    self.obs_spaces[policy_id] = obs_spaces[agent_id]
                    self.policy_ids.append(policy_id)

                if policy_id in self.action_spaces.keys():
                    if self.action_spaces[policy_id] != action_spaces[agent_id]:
                        raise ValueError(
                            f"multiple agents have {policy_id=} but have different "
                            f"action spaces"
                        )
                else:
                    self.action_spaces[policy_id] = action_spaces[agent_id]
        else:
            self.pid = {agent_id: agent_id for agent_id in obs_spaces.keys()}
            self.obs_spaces = obs_spaces
            self.action_spaces = action_spaces
            self.policy_ids = list(obs_spaces.keys())

        self.agent_ids = sorted(list(self.pid.keys()))
        self.policy_ids.sort()

        all_obs_size = sum(flat_size(obs_space) for obs_space in obs_spaces.values())
        self.discrims = nn.ModuleDict()
        self.optims = []
        for policy_id in self.policy_ids:
            input_size = all_obs_size + flat_size(self.action_spaces[policy_id])
            self.discrims[policy_id] = nn.Sequential(
                nn.Linear(input_size, self.fc_dim, device=self.device),
                nn.ReLU(),
                nn.Linear(self.fc_dim, self.fc_dim, device=self.device),
                nn.ReLU(),
                nn.Linear(self.fc_dim, 1, device=self.device),
                nn.Sigmoid(),
            )
            self.optims.append(torch.optim.Adam(self.discrims[policy_id].parameters()))
        return self

    def forward(self, all_obs, all_actions):
        obs_inputs = [
            all_obs[agent_id]
            for agent_id in shuffled_within_classes(self.agent_ids, self.pid)
        ]

        ret = {}
        for agent_id in self.agent_ids:
            own_action_input = all_actions[agent_id]
            ret[agent_id] = (
                self.discrims[self.pid[agent_id]]
                .forward(torch.cat(obs_inputs + [own_action_input], dim=1))
                .reshape(-1)
            )
        return ret

    def get_rewards(self, rollout) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            predictions = self.forward(rollout["obs"], rollout["actions"])
            return {
                agent_id: torch.log(predictions[agent_id]).cpu().numpy()
                for agent_id in self.agent_ids
            }

    def update(self, rollout) -> None:
        rollout_length = rollout["terminal"].shape[0]

        rollout_idxs = np.arange(rollout_length)
        np.random.shuffle(rollout_idxs)

        for i in range(rollout_length // self.minibatch_size):
            rollout_idx = rollout_idxs[
                i * self.minibatch_size : (i + 1) * self.minibatch_size
            ]
            if self.ptr + self.minibatch_size > self.demo_size:
                self.ptr = 0
                np.random.shuffle(self.demo_idxs)
            demo_idx = self.demo_idxs[self.ptr : self.ptr + self.minibatch_size]
            self.ptr += self.minibatch_size

            rollout_preds = self.forward(
                all_obs=deep_idx(rollout["obs"], rollout_idx),
                all_actions=deep_idx(rollout["actions"], rollout_idx),
            )

            expert_preds = self.forward(
                all_obs={
                    agent_id: encode_samples(
                        agent_obs,
                        self.obs_spaces[self.pid[agent_id]],
                        device=self.device,
                    )
                    for agent_id, agent_obs in deep_idx(
                        self.demo_data["obs"], demo_idx
                    ).items()
                    if agent_id != "global_state"
                },
                all_actions={
                    agent_id: encode_samples(
                        agent_action,
                        self.action_spaces[self.pid[agent_id]],
                        device=self.device,
                    )
                    for agent_id, agent_action in deep_idx(
                        self.demo_data["actions"], demo_idx
                    ).items()
                },
            )

            rollout_preds_flat = torch.cat(list(rollout_preds.values()))
            expert_preds_flat = torch.cat(list(expert_preds.values()))
            loss = -torch.mean(torch.log(1 - rollout_preds_flat)) - torch.mean(
                torch.log(expert_preds_flat)
            )

            if self.logger is not None:
                self.logger.push(
                    {
                        f"discrim_rollout_pred_{agent_id}": rollout_preds[agent_id]
                        .mean()
                        .item()
                        for agent_id in self.agent_ids
                    }
                )
                self.logger.push(
                    {
                        f"discrim_expert_pred_{agent_id}": expert_preds[agent_id]
                        .mean()
                        .item()
                        for agent_id in self.agent_ids
                    }
                )
                self.logger.push(discrim_loss=loss.item())

            for optim in self.optims:
                optim.zero_grad(set_to_none=True)
            loss.backward()
            for optim in self.optims:
                optim.step()

        self.logger.step()


# Each agent has their own discriminator which takes in (own obs) + (own action)
class FullyLocalGAILReward(RewardSignal, nn.Module):
    def __init__(
        self,
        demo_filename: str,
        lr: float,
        fc_dim=64,
        minibatch_size=128,
        share_parameters=True,
        demo_limit: int = None,
    ):
        nn.Module.__init__(self)
        self.fc_dim = fc_dim
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.share_parameters = share_parameters

        self.obs_spaces = None
        self.action_spaces = None
        self.agent_ids = None
        self.policy_ids = None
        self.pid = None

        self.discrims = None
        self.optims = None

        with open(demo_filename, "rb") as f:
            data = pickle.load(f)
        if demo_limit is None:
            self.demo_data = data
        else:
            self.demo_data = deep_idx(data, slice(demo_limit), copy=True)
        self.demo_size = self.demo_data["terminal"].shape[0]
        self.demo_idxs = np.arange(self.demo_size)
        np.random.shuffle(self.demo_idxs)
        self.ptr = 0

        self.logger = None
        self.device = None

    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        self.logger = logger
        self.device = device

        if self.share_parameters:
            self.obs_spaces = {}
            self.action_spaces = {}
            self.pid = {}
            self.policy_ids = []
            for agent_id in obs_spaces.keys():
                policy_id = get_agent_class(agent_id)

                self.pid[agent_id] = policy_id

                if policy_id in self.obs_spaces.keys():
                    if self.obs_spaces[policy_id] != obs_spaces[agent_id]:
                        raise ValueError(
                            f"multiple agents have {policy_id=} but have different "
                            f"observation spaces"
                        )
                else:
                    self.obs_spaces[policy_id] = obs_spaces[agent_id]
                    self.policy_ids.append(policy_id)

                if policy_id in self.action_spaces.keys():
                    if self.action_spaces[policy_id] != action_spaces[agent_id]:
                        raise ValueError(
                            f"multiple agents have {policy_id=} but have different "
                            f"action spaces"
                        )
                else:
                    self.action_spaces[policy_id] = action_spaces[agent_id]
        else:
            self.pid = {agent_id: agent_id for agent_id in obs_spaces.keys()}
            self.obs_spaces = obs_spaces
            self.action_spaces = action_spaces
            self.policy_ids = list(obs_spaces.keys())

        self.agent_ids = sorted(list(self.pid.keys()))
        self.policy_ids.sort()

        self.discrims = nn.ModuleDict()
        self.optims = []
        for policy_id in self.policy_ids:
            input_size = flat_size(self.obs_spaces[policy_id]) + flat_size(
                self.action_spaces[policy_id]
            )
            self.discrims[policy_id] = nn.Sequential(
                nn.Linear(input_size, self.fc_dim, device=self.device),
                nn.ReLU(),
                nn.Linear(self.fc_dim, self.fc_dim, device=self.device),
                nn.ReLU(),
                nn.Linear(self.fc_dim, 1, device=self.device),
                nn.Sigmoid(),
            )
            self.optims.append(torch.optim.Adam(self.discrims[policy_id].parameters()))
        return self

    def forward(self, all_obs, all_actions):
        ret = {}
        for agent_id in self.agent_ids:
            own_obs_input = all_obs[agent_id]
            own_action_input = all_actions[agent_id]
            ret[agent_id] = (
                self.discrims[self.pid[agent_id]]
                .forward(torch.cat([own_obs_input, own_action_input], dim=1))
                .reshape(-1)
            )
        return ret

    def get_rewards(self, rollout) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            predictions = self.forward(rollout["obs"], rollout["actions"])
            return {
                agent_id: torch.log(predictions[agent_id]).cpu().numpy()
                for agent_id in self.agent_ids
            }

    def update(self, rollout) -> None:
        rollout_length = rollout["terminal"].shape[0]

        rollout_idxs = np.arange(rollout_length)
        np.random.shuffle(rollout_idxs)

        for i in range(rollout_length // self.minibatch_size):
            rollout_idx = rollout_idxs[
                i * self.minibatch_size : (i + 1) * self.minibatch_size
            ]
            if self.ptr + self.minibatch_size > self.demo_size:
                self.ptr = 0
                np.random.shuffle(self.demo_idxs)
            demo_idx = self.demo_idxs[self.ptr : self.ptr + self.minibatch_size]
            self.ptr += self.minibatch_size

            rollout_preds = self.forward(
                all_obs=deep_idx(rollout["obs"], rollout_idx),
                all_actions=deep_idx(rollout["actions"], rollout_idx),
            )

            expert_preds = self.forward(
                all_obs={
                    agent_id: encode_samples(
                        agent_obs,
                        self.obs_spaces[self.pid[agent_id]],
                        device=self.device,
                    )
                    for agent_id, agent_obs in deep_idx(
                        self.demo_data["obs"], demo_idx
                    ).items()
                },
                all_actions={
                    agent_id: encode_samples(
                        agent_action,
                        self.action_spaces[self.pid[agent_id]],
                        device=self.device,
                    )
                    for agent_id, agent_action in deep_idx(
                        self.demo_data["actions"], demo_idx
                    ).items()
                },
            )

            rollout_preds_flat = torch.cat(list(rollout_preds.values()))
            expert_preds_flat = torch.cat(list(expert_preds.values()))
            loss = -torch.mean(torch.log(1 - rollout_preds_flat)) - torch.mean(
                torch.log(expert_preds_flat)
            )

            if self.logger is not None:
                self.logger.push(
                    {
                        f"discrim_rollout_pred_{agent_id}": rollout_preds[agent_id]
                        .mean()
                        .item()
                        for agent_id in self.agent_ids
                    }
                )
                self.logger.push(
                    {
                        f"discrim_expert_pred_{agent_id}": expert_preds[agent_id]
                        .mean()
                        .item()
                        for agent_id in self.agent_ids
                    }
                )
                self.logger.push(discrim_loss=loss.item())

            for optim in self.optims:
                optim.zero_grad(set_to_none=True)
            loss.backward()
            self.logger.push(discrim_grad_norm=grad_norm(self))
            for optim in self.optims:
                optim.step()

        self.logger.step()
