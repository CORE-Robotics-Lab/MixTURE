from collections import OrderedDict
from typing import Dict, List, Set

import gym
import numpy as np

from envs.base_multi_agent import CARDINALS, MultiAgentEnv, Position


class SimpleFireCommander(MultiAgentEnv):
    def __init__(
        self,
        map_size=5,
        n_perception=2,
        n_action=1,
        n_fires=1,
        vision=1,
        fire_propagation_period=None,
        max_timesteps=100,
    ):
        self.map_size = map_size
        self.n_perception = n_perception
        self.n_action = n_action
        self.n_fires = n_fires
        self.vision = vision
        self.max_timesteps = max_timesteps
        self.fire_propagation_period = fire_propagation_period

        self.t = 0
        self.perception_pos: List[Position] = []
        self.action_pos: List[Position] = []
        self.fire_pos: Set[Position] = set()
        self.firefronts: List[Position] = []

        self._observation_spaces: Dict[str, gym.spaces.Space] = {}
        self._action_spaces: Dict[str, gym.spaces.Space] = {}
        for i in range(self.n_perception):
            self._observation_spaces[f"perception{i}"] = gym.spaces.Dict(
                {
                    "vision": gym.spaces.Box(
                        0, 1, (4, self.vision * 2 + 1, self.vision * 2 + 1)
                    ),
                    "pos": gym.spaces.Box(-1, 1, (2,)),
                }
            )
            self._action_spaces[f"perception{i}"] = gym.spaces.Discrete(5)
        for i in range(self.n_action):
            self._observation_spaces[f"action{i}"] = gym.spaces.Dict(
                {"pos": gym.spaces.Box(-1, 1, (2,))}
            )
            self._action_spaces[f"action{i}"] = gym.spaces.Discrete(6)

        self._metrics = OrderedDict()

    @property
    def observation_spaces(self) -> Dict[str, gym.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[str, gym.Space]:
        return self._action_spaces

    @property
    def metrics(self) -> OrderedDict:
        return self._metrics

    def get_obs(self):
        global_state = np.zeros(
            shape=(4, self.map_size + 2 * self.vision, self.map_size + 2 * self.vision)
        )
        for pos in self.perception_pos:
            global_state[0, pos.y + self.vision, pos.x + self.vision] = 1
        for pos in self.action_pos:
            global_state[1, pos.y + self.vision, pos.x + self.vision] = 1
        for pos in self.fire_pos:
            global_state[2, pos.y + self.vision, pos.x + self.vision] = 1

        if self.vision > 0:
            global_state[3] = 1
            global_state[3, self.vision : -self.vision, self.vision : -self.vision] = 0

        ret = {}
        for i in range(self.n_perception):
            pos = self.perception_pos[i]
            ret[f"perception{i}"] = {
                "vision": global_state[
                    :,
                    pos.y : pos.y + 2 * self.vision + 1,
                    pos.x : pos.x + 2 * self.vision + 1,
                ],
                "pos": np.asarray(
                    (
                        (pos.x / (self.map_size - 1) * 2) - 1,
                        (pos.y / (self.map_size - 1) * 2) - 1,
                    )
                ),
            }

        for i in range(self.n_action):
            pos = self.action_pos[i]
            ret[f"action{i}"] = {
                "pos": np.asarray(
                    (
                        (pos.x / (self.map_size - 1) * 2) - 1,
                        (pos.y / (self.map_size - 1) * 2) - 1,
                    )
                )
            }

        return ret

    def reset(self, seed: int | None = None):
        self.perception_pos.clear()
        self.action_pos.clear()
        self.fire_pos.clear()

        self.t = 0

        rng = np.random.default_rng(
            np.random.randint(2**30) if seed is None else seed
        )
        loc_flat_idx = rng.choice(
            self.map_size**2,
            size=self.n_perception + self.n_action + self.n_fires,
            replace=False,
        )
        yy, xx = np.unravel_index(loc_flat_idx, (self.map_size, self.map_size))

        for i in range(self.n_perception):
            self.perception_pos.append(Position(yy[i], xx[i]))
        for i in range(self.n_action):
            self.action_pos.append(
                Position(yy[self.n_perception + i], xx[self.n_perception + i])
            )
        for i in range(self.n_fires):
            self.fire_pos.add(
                Position(
                    yy[self.n_perception + self.n_action + i],
                    xx[self.n_perception + self.n_action + i],
                )
            )

        self.firefronts = list(self.fire_pos)
        self._metrics["fires_extinguished"] = 0
        self._metrics["won"] = False

        return self.get_obs()

    def step(self, actions):
        self.t += 1

        for key in actions.keys():
            if not (
                (key[:-1] == "perception" and 0 <= int(key[-1]) < self.n_perception)
                or (key[:-1] == "action" and 0 <= int(key[-1]) < self.n_action)
            ):
                raise ValueError(f"unexpected action key: {key}")

        rewards = {
            agent_id: -0.1 * len(self.fire_pos)
            for agent_id in self.observation_spaces.keys()
        }

        for i in range(self.n_perception):
            if 0 <= actions[f"perception{i}"] < 4:
                if (
                    self.perception_pos[i] + CARDINALS[actions[f"perception{i}"]]
                ).in_bounds(self.map_size):
                    self.perception_pos[i] += CARDINALS[actions[f"perception{i}"]]
            elif actions[f"perception{i}"] != 4:
                raise ValueError(
                    f"unexpected action for perception{i}: {actions[f'perception{i}']}"
                )

        extinguished_source = False
        for i in range(self.n_action):
            if 0 <= actions[f"action{i}"] < 4:
                if (self.action_pos[i] + CARDINALS[actions[f"action{i}"]]).in_bounds(
                    self.map_size
                ):
                    self.action_pos[i] += CARDINALS[actions[f"action{i}"]]
            elif actions[f"action{i}"] == 5:  # extinguish action
                if self.action_pos[i] in self.fire_pos:
                    self.fire_pos.remove(self.action_pos[i])
                    self._metrics["fires_extinguished"] += 1

                    if self.action_pos[i] in self.firefronts:
                        self.firefronts.remove(self.action_pos[i])
            elif actions[f"action{i}"] != 4:
                raise ValueError(
                    f"unexpected action for action{i}: {actions[f'action{i}']}"
                )

        if extinguished_source:
            for i in range(self.n_perception):
                rewards[f"perception{i}"] += 10

        self.propogate_fires()

        done = len(self.fire_pos) == 0 or self.t >= self.max_timesteps

        if len(self.fire_pos) == 0:
            self._metrics["won"] = True

        return self.get_obs(), rewards, done, {}

    def propogate_fires(self):
        if self.fire_propagation_period is None:
            return

        if self.t % self.fire_propagation_period == 0:
            new_firefronts = []
            for old_pos in self.firefronts:
                new_pos = Position(
                    old_pos.x + np.random.randint(-1, 2),
                    old_pos.y + np.random.randint(-1, 2),
                )
                if new_pos.in_bounds(self.map_size):
                    new_firefronts.append(new_pos)
                else:
                    new_firefronts.append(old_pos)
            self.firefronts = new_firefronts
        self.fire_pos = self.fire_pos.union(self.firefronts)

    def render(self):
        from termcolor import colored

        p_set = set(self.perception_pos)
        a_set = set(self.action_pos)

        print("=" * (self.map_size + 2))
        for y in range(self.map_size):
            line = ""
            for x in range(self.map_size):
                if Position(x, y) in a_set:
                    n_units = sum(
                        pos == Position(x, y)
                        for pos in self.perception_pos + self.action_pos
                    )
                    if n_units == 1:
                        char = "^"
                    elif n_units <= 9:
                        char = str(n_units)
                    else:
                        char = "@"
                    if Position(x, y) in self.fire_pos:
                        line += colored(char, color="red")
                    else:
                        line += colored(char, color="blue")
                elif Position(x, y) in p_set:
                    if Position(x, y) in self.fire_pos:
                        line += colored("o", color="red")
                    else:
                        line += colored("o", color="green")
                else:
                    if Position(x, y) in self.fire_pos:
                        line += colored("*", color="red")
                    else:
                        line += colored(".", color="white")
            print("|" + line + "|")
        print("=" * (self.map_size + 2))
