import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import gym
import numpy as np
import torch
import torch.distributions as distributions
import torch.nn.functional as F

from utils import flat_size


class PolicyHead(ABC):
    """Used for converting raw nn outputs to action samples or distribution info."""

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        pass

    @property
    @abstractmethod
    def input_size(self) -> int:
        pass

    def get_distr(self, logits: torch.Tensor) -> distributions.Distribution:
        raise NotImplementedError

    def step(self, logits: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        """Given the raw output of a policy nn, samples a single action + its log prob.

        Intended for use during rollouts / experience collection.

        Args:
            logits: A single tensor (size-1 batch dim) which parameterizes the action
                distribution

        Returns:
            A single action (typically a tensor or a dictionary mapping action space
                keys to tensors, and the corresponding log probability
        """
        distr = self.get_distr(logits)
        action = distr.sample()
        return action.squeeze(0).cpu().numpy(), distr.log_prob(action).squeeze(0)

    @abstractmethod
    def logp(self, logits: torch.Tensor, actions: Any) -> torch.Tensor:
        """Given the raw output of a policy nn, returns log prob of a batch of actions.

        Intended for use during the update loop to compute policy loss.

        Args:
            logits: A batch of tensors which parameterizes the action distribution.
            actions: A batch of actions, typically a tensor or a dictionary mapping
                action space keys to tensors.

        Returns:
            A batch of log probabilities.

        """
        pass
        # return self.get_distr(logits).log_prob(actions)


class DiscretePolicy(PolicyHead):
    def __init__(self, action_space: gym.spaces.Discrete, *_args, **_kwargs):
        assert isinstance(action_space, gym.spaces.Discrete)
        self._action_space = action_space

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self._action_space

    @property
    def input_size(self):
        return self._action_space.n

    def get_distr(self, logits):
        return distributions.Categorical(logits=logits)

    def logp(self, logits: torch.Tensor, actions: Any) -> torch.Tensor:
        return self.get_distr(logits).log_prob(torch.argmax(actions, dim=1))


class GaussianPolicy(PolicyHead):
    def __init__(self, action_space: gym.spaces.Box, init_std=0.5, *_args, **_kwargs):
        assert isinstance(action_space, gym.spaces.Box)
        if len(action_space.shape) > 1:
            warnings.warn(
                f"continuous action_space has shape={action_space.shape}, but typical "
                f"use cases are expected to involve only a 1D list of Gaussian outputs"
            )
        if action_space.bounded_below.any() != action_space.bounded_above.any():
            warnings.warn(
                f"expected either unbounded action space or action space bounded both "
                f"above and below"
            )
        elif action_space.bounded_above.any():
            self.bounds = (
                torch.tensor(action_space.low),
                torch.tensor(action_space.high),
            )
        else:
            self.bounds = None
        self._action_space = action_space
        self._std_offset = torch.log(torch.exp(torch.as_tensor(init_std)) - 1.0)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def input_size(self):
        return 2 * np.prod(self._action_space.shape)

    def get_distr(self, logits) -> distributions.Distribution:
        assert logits.shape[1:] == (self.input_size,)

        mu = logits[:, : np.prod(self.action_space.shape)].reshape(
            (-1,) + self.action_space.shape
        )
        sigma = F.softplus(
            logits[:, np.prod(self.action_space.shape) :]
            + self._std_offset.to(device=logits.device)
        ).reshape((-1,) + self.action_space.shape)

        distr = distributions.Normal(mu, sigma)

        if self.bounds is not None:
            transforms = [
                distributions.SigmoidTransform(),
                distributions.AffineTransform(
                    self.bounds[0], self.bounds[1] - self.bounds[0]
                ),
            ]
            distr = distributions.TransformedDistribution(distr, transforms)

        return distr

    def step(self, logits: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        distr = self.get_distr(logits)
        action = distr.sample()
        return action.squeeze(0).cpu().numpy(), torch.sum(
            distr.log_prob(action), dim=1
        ).squeeze(0)

    def logp(self, logits: torch.Tensor, actions: Any) -> torch.Tensor:
        return torch.sum(self.get_distr(logits).log_prob(actions), dim=1)


class MultiHeadPolicy(PolicyHead):
    def __init__(self, action_space: gym.spaces.Dict, init_std=0.5, *_args, **_kwargs):
        assert isinstance(action_space, gym.spaces.Dict)
        self._action_space = action_space
        self.keys = []
        self.policies = []
        self.logit_intervals = []
        self.action_intervals = []

        curr = 0
        for key in sorted(action_space.spaces.keys()):
            space = action_space[key]
            if isinstance(space, gym.spaces.Discrete):
                new_pi = DiscretePolicy(space)
            elif isinstance(space, gym.spaces.Box):
                new_pi = GaussianPolicy(space, init_std=init_std)
            else:
                assert isinstance(space, gym.spaces.Dict)
                new_pi = MultiHeadPolicy(space, init_std=init_std)
            self.keys.append(key)
            self.policies.append(new_pi)
            self.logit_intervals.append((curr, curr + new_pi.input_size))
            self.action_intervals.append((curr, curr + flat_size(space)))
            curr += new_pi.input_size

        self._input_size = curr

    @property
    def action_space(self) -> gym.spaces.Dict:
        return self._action_space

    @property
    def input_size(self) -> int:
        return self._input_size

    def step(
        self, logits: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        assert logits.shape == (1, self.input_size)
        actions = {}
        logp_total = torch.tensor(0.0, device=logits.device)
        for key, policy_head, interval in zip(
            self.keys, self.policies, self.logit_intervals
        ):
            action, logp = policy_head.step(logits[:, interval[0] : interval[1]])
            actions[key] = action
            logp_total += logp
        return actions, logp_total

    def logp(self, logits: torch.Tensor, actions: torch.tensor) -> torch.Tensor:
        assert logits.shape[1:] == (self.input_size,)
        logp_total = torch.tensor(0.0, device=logits.device)
        for policy_head, logit_int, action_int in zip(
            self.policies, self.logit_intervals, self.action_intervals
        ):
            logp = policy_head.logp(
                logits[:, logit_int[0] : logit_int[1]],
                actions[:, action_int[0] : action_int[1]],
            )
            logp_total = logp_total + logp
        return logp_total


def get_policy_head(action_space: gym.Space, *args, **kwargs):
    if isinstance(action_space, gym.spaces.Discrete):
        return DiscretePolicy(action_space, *args, **kwargs)
    elif isinstance(action_space, gym.spaces.Box):
        return GaussianPolicy(action_space, *args, **kwargs)
    elif isinstance(action_space, gym.spaces.Dict):
        return MultiHeadPolicy(action_space, *args, **kwargs)
    raise TypeError
