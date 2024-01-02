from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import gym
import torch
import torch.nn as nn

from action_distributions import get_policy_head
from utils import deep_idx, flat_size, get_agent_class, shuffled_within_classes


class AgentGroup(ABC, nn.Module):
    """Abstract base class defining the set of all agents in a multiagent environment"""

    @property
    def optims(self) -> List[torch.optim.Optimizer]:
        return []

    def configured_with(
        self,
        obs_spaces: Dict[str, gym.Space],
        action_spaces: Dict[str, gym.Space],
        logger=None,
        device=torch.device("cpu"),
    ):
        """Sets up the agent group with a specific environment.

        In typical usage, an agent group will initialize the agents (possibly with some
        parameters, such as learning rate), and a trainer class will handle the
        environment observation and action space boilerplate.
        """
        return self

    def reset(self) -> None:
        """May be defined for recurrent policies to reset internal hidden states."""
        pass

    @abstractmethod
    def rollout_forward(
        self, all_obs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """Returns actions and action log probability given observations.

        May depend on or modify the agent's internal hidden state.
        """
        pass

    @abstractmethod
    def train_forward(
        self, rollout, batch_idx
    ) -> Union[
        Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
    ]:
        """Returns action log probabilities given a batch of observations and actions"""
        pass

    @abstractmethod
    def predict_values(self, all_obs, all_actions) -> Dict[str, torch.Tensor]:
        """Predicts all agent's values given own obs and other agent's actions."""
        pass

    def custom_loss(self, loss, rollout, batch_idx, iteration) -> torch.Tensor:
        """May be defined to include additional surrogate losses."""
        pass


class RandomAgents(AgentGroup):
    def __init__(self):
        super().__init__()
        self.agent_ids = None
        self.policy_heads = None
        self.logger = None
        self.device = None

    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        self.logger = logger
        self.device = device

        self.agent_ids = sorted(list(obs_spaces.keys()))
        self.policy_heads = {
            agent_id: get_policy_head(space)
            for agent_id, space in action_spaces.items()
        }
        return self

    def rollout_forward(self, all_obs):
        actions = {}
        logps = {}
        for agent_id, policy_head in self.policy_heads.items():
            actions[agent_id], logps[agent_id] = policy_head.step(
                torch.zeros((1, policy_head.input_size))
            )
        return actions, logps

    def train_forward(self, rollout, batch_idx):
        ret = {
            agent_id: policy_head.logp(
                torch.zeros((len(batch_idx), policy_head.input_size)),
                deep_idx(rollout["actions"][agent_id], batch_idx),
            )
            for agent_id, policy_head in self.policy_heads.items()
        }
        return ret

    def predict_values(self, all_obs, all_actions):
        return {
            agent_id: torch.zeros((all_obs[agent_id].shape[0]), device=self.device)
            for agent_id in self.agent_ids
        }


class CTDE(AgentGroup):
    def __init__(
        self,
        policy_lr: float,
        critic_lr: float,
        fc_dim=64,
        reduce_head_weights=True,
        share_parameters=True,
        permute_within_classes=True,
    ):
        super().__init__()

        self.policy_lr = policy_lr
        self.critic_lr = critic_lr
        self.fc_dim = fc_dim
        self.reduce_head_weights = reduce_head_weights
        self.share_parameters = share_parameters
        self.permute_within_classes = permute_within_classes
        if self.permute_within_classes and not self.share_parameters:
            raise ValueError(
                "permute_within_classes is only applicable if share_parameters==True"
            )

        self.obs_spaces = None
        self.action_spaces = None
        self.agent_ids = None
        self.policy_ids = None
        self.pid = None

        self.policies = None
        self.policy_heads = None
        self.critics = None
        self.logger = None
        self.device = None

        self._optims = []

    @property
    def optims(self) -> List[torch.optim.Optimizer]:
        return self._optims

    def _assign_spaces(self, obs_spaces, action_spaces):
        if self.share_parameters:
            self.obs_spaces = {}
            self.action_spaces = {}
            self.policy_ids = []
            self.pid = {}
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

    def _setup_critics(self):
        total_size = 0
        for agent_id in self.agent_ids:
            total_size += flat_size(self.obs_spaces[self.pid[agent_id]])
            total_size += flat_size(self.action_spaces[self.pid[agent_id]])

        self.critics = nn.ModuleDict()
        for policy_id in self.policy_ids:
            critic_input_size = total_size - flat_size(self.action_spaces[policy_id])
            self.critics[policy_id] = nn.Sequential(
                nn.Linear(critic_input_size, self.fc_dim),
                nn.ReLU(),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.ReLU(),
                nn.Linear(self.fc_dim, 1),
            ).to(self.device)

        for policy_id in self.policy_ids:
            self.optims.append(
                torch.optim.Adam(
                    self.critics[policy_id].parameters(), lr=self.critic_lr
                )
            )

        with torch.no_grad():
            if self.reduce_head_weights:
                for critic in self.critics.values():
                    critic[-1].weight *= 0.01

    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        self.logger = logger
        self.device = device
        self._assign_spaces(obs_spaces, action_spaces)

        self.policy_heads = {
            policy_id: get_policy_head(space)
            for policy_id, space in self.action_spaces.items()
        }

        self.policies = nn.ModuleDict(
            {
                policy_id: nn.Sequential(
                    nn.Linear(flat_size(self.obs_spaces[policy_id]), self.fc_dim),
                    nn.ReLU(),
                    nn.Linear(self.fc_dim, self.fc_dim),
                    nn.ReLU(),
                    nn.Linear(self.fc_dim, self.policy_heads[policy_id].input_size),
                ).to(self.device)
                for policy_id in self.policy_ids
            }
        )

        with torch.no_grad():
            if self.reduce_head_weights:
                for policy in self.policies.values():
                    policy[-1].weight *= 0.01

        self._optims = []
        for policy_id in self.policy_ids:
            self.optims.append(
                torch.optim.Adam(
                    self.policies[policy_id].parameters(), lr=self.policy_lr
                )
            )

        self._setup_critics()
        return self

    def rollout_forward(self, all_obs: Dict[str, torch.Tensor]):
        actions, logps = {}, {}
        for agent_id in self.agent_ids:
            pid = self.pid[agent_id]
            actions[agent_id], logps[agent_id] = self.policy_heads[pid].step(
                self.policies[pid](all_obs[agent_id])
            )
        return actions, logps

    def train_forward(self, rollout, batch_idx):
        ret = {}
        for agent_id in self.agent_ids:
            pid = self.pid[agent_id]
            logits = self.policies[pid](rollout["obs"][agent_id][batch_idx])
            ret[agent_id] = self.policy_heads[pid].logp(
                logits, deep_idx(rollout["actions"][agent_id], batch_idx)
            )
        return ret

    def predict_values(self, all_obs, all_actions) -> Dict[str, torch.Tensor]:
        ret = {}
        for agent_id in self.agent_ids:
            model_inputs = []
            for other_id in (
                shuffled_within_classes(self.agent_ids, self.pid)
                if self.permute_within_classes
                else self.agent_ids
            ):
                if other_id == agent_id:
                    continue
                # observations AND actions should be tensors
                model_inputs.append(all_obs[other_id])
                model_inputs.append(all_actions[other_id])
            model_inputs.append(all_obs[agent_id])
            ret[agent_id] = self.critics[self.pid[agent_id]](
                torch.cat(model_inputs, dim=1)
            ).reshape(-1)
        return ret
