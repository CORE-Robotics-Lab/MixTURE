from typing import Dict, List

import gym
import numpy as np

from envs.base_multi_agent import CARDINALS, MultiAgentEnv, Position


class PredatorPrey(MultiAgentEnv):
    def __init__(
        self,
        map_size=5,
        n_predator=3,
        n_prey=1,
        vision=0,
        max_timesteps=20,
        onehot_pos=False,
    ):
        self.map_size = map_size
        self.n_predator = n_predator
        self.n_prey = n_prey
        self.vision = vision
        self.max_timesteps = max_timesteps
        self.onehot_pos = onehot_pos

        self.t = 0
        self.predator_pos: List[Position] = []
        self.prey_pos: List[Position] = []

        self._observation_spaces: Dict[str, gym.spaces.Space] = {}
        self._action_spaces: Dict[str, gym.spaces.Space] = {}
        for i in range(self.n_predator):
            self._observation_spaces[f"predator{i}"] = gym.spaces.Dict(
                {
                    "vision": gym.spaces.Box(
                        0, 1, (3, self.vision * 2 + 1, self.vision * 2 + 1)
                    ),
                    "pos": gym.spaces.Discrete(map_size**2)
                    if onehot_pos
                    else gym.spaces.Box(-1, 1, (2,)),
                }
            )
            self._action_spaces[f"predator{i}"] = gym.spaces.Discrete(5)

    @property
    def observation_spaces(self) -> Dict[str, gym.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[str, gym.Space]:
        return self._action_spaces

    def get_obs(self):
        # 3 channels: predator, prey, map boundary
        global_state = np.zeros(
            shape=(3, self.map_size + 2 * self.vision, self.map_size + 2 * self.vision)
        )
        for pos in self.predator_pos:
            global_state[0, pos.y + self.vision, pos.x + self.vision] = 1
        for pos in self.prey_pos:
            global_state[1, pos.y + self.vision, pos.x + self.vision] = 1

        if self.vision > 0:
            global_state[2] = 1
            global_state[2, self.vision : -self.vision, self.vision : -self.vision] = 0

        ret = {}
        for i in range(self.n_predator):
            pos = self.predator_pos[i]
            ret[f"predator{i}"] = {
                "vision": global_state[
                    :,
                    pos.y : pos.y + 2 * self.vision + 1,
                    pos.x : pos.x + 2 * self.vision + 1,
                ],
                "pos": np.array(pos.y * self.map_size + pos.x)
                if self.onehot_pos
                else np.asarray(
                    (
                        (pos.x / (self.map_size - 1) * 2) - 1,
                        (pos.y / (self.map_size - 1) * 2) - 1,
                    )
                ),
            }

        return ret

    def reset(self, seed: int | None = None):
        self.predator_pos.clear()
        self.prey_pos.clear()

        rng = np.random.default_rng(
            np.random.randint(2**30) if seed is None else seed
        )
        loc_flat_idx = rng.choice(
            self.map_size**2, size=self.n_predator + self.n_prey, replace=False
        )
        yy, xx = np.unravel_index(loc_flat_idx, (self.map_size, self.map_size))

        for i in range(self.n_predator):
            self.predator_pos.append(Position(yy[i], xx[i]))
        for i in range(self.n_prey):
            self.prey_pos.append(
                Position(yy[self.n_predator + i], xx[self.n_predator + i])
            )

        self.t = 0

        return self.get_obs()

    def step(self, actions):
        self.t += 1

        for key in actions.keys():
            assert key[:-1] == "predator" and int(key[-1]) in range(
                self.n_predator
            ), f"Bad action key: {key}"

        prey_set = set(self.prey_pos)
        rewards = {f"predator{i}": -0.05 for i in range(self.n_predator)}
        for i in range(self.n_predator):
            if self.predator_pos[i] in prey_set:
                rewards[f"predator{i}"] = 0
                continue

            if actions[f"predator{i}"] != 4 and (
                self.predator_pos[i] + CARDINALS[actions[f"predator{i}"]]
            ).in_bounds(self.map_size):
                self.predator_pos[i] += CARDINALS[actions[f"predator{i}"]]
                if self.predator_pos[i] in prey_set:
                    rewards[f"predator{i}"] = 0

        done = (
            all(pos in prey_set for pos in self.predator_pos)
            or self.t >= self.max_timesteps
        )

        return self.get_obs(), rewards, done, {}

    def render(self):
        from termcolor import colored

        predator_set = set(self.predator_pos)
        prey_set = set(self.prey_pos)

        print("=" * (self.map_size + 2))
        for y in range(self.map_size):
            line = ""
            for x in range(self.map_size):
                if Position(x, y) in predator_set:
                    if Position(x, y) in prey_set:
                        line += colored(
                            str(
                                sum(pos == Position(x, y) for pos in self.predator_pos)
                            ),
                            color="green",
                        )
                    else:
                        line += colored(
                            str(
                                sum(pos == Position(x, y) for pos in self.predator_pos)
                            ),
                            color="blue",
                        )
                else:
                    if Position(x, y) in prey_set:
                        line += colored("*", color="red")
                    else:
                        line += colored(".", color="white")
            print("|" + line + "|")
        print("=" * (self.map_size + 2))
