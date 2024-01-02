from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Union

import gym
import numpy as np

ObsType = Union[Dict[str, Any], np.ndarray]


@dataclass
class Direction:
    dx: int
    dy: int

    def ordinal(self):
        if abs(self.dx) + abs(self.dy) > 1:
            raise ValueError

        if self.dx == 0:
            if self.dy == -1:
                return 0
            if self.dy == 1:
                return 2
            return 4
        if self.dx == 1:
            return 1
        return 3

    def rot90(self):
        return Direction(-self.dy, self.dx)

    def __mul__(self, other):
        return Direction(other * self.dx, other * self.dy)

    def __rmul__(self, other):
        return Direction(other * self.dx, other * self.dy)


CARDINALS = [Direction(0, -1), Direction(1, 0), Direction(0, 1), Direction(-1, 0)]


@dataclass
class Position:
    x: int
    y: int

    def manhattan(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)

    def chebyshev(self, other):
        return max(abs(self.x - other.x), abs(self.y - other.y))

    def direction_to(self, other: "Position"):
        dx = other.x - self.x
        dy = other.y - self.y
        if abs(dx) >= abs(dy):
            if dx > 0:
                return Direction(1, 0)
            if dx < 0:
                return Direction(-1, 0)
            return Direction(0, 0)
        if dy > 0:
            return Direction(0, 1)
        return Direction(0, -1)

    def in_bounds(self, dim=5):
        return 0 <= self.x < dim and 0 <= self.y < dim

    def __add__(self, other: Direction):
        return Position(self.x + other.dx, self.y + other.dy)

    def __hash__(self):
        return int((self.x + 1) * 997 + self.y)

    def __repr__(self):
        return f"[{self.x},{self.y}]"


class MultiAgentEnv(ABC):
    @property
    @abstractmethod
    def observation_spaces(self) -> Dict[str, gym.Space]:
        pass

    @property
    @abstractmethod
    def action_spaces(self) -> Dict[str, gym.Space]:
        pass

    @abstractmethod
    def get_obs(self) -> Dict[str, ObsType]:
        pass

    @abstractmethod
    def reset(self, seed: int | None = None) -> Dict[str, ObsType]:
        pass

    @abstractmethod
    def step(self, actions: Dict[str, Any]):
        pass

    @property
    def metrics(self) -> OrderedDict:
        return OrderedDict()
