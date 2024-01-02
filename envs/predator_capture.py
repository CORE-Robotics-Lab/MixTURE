from typing import Dict, List

import gym
import numpy as np

from envs.base_multi_agent import CARDINALS, MultiAgentEnv, Position


class PredatorCapture(MultiAgentEnv):
    def __init__(
        self,
        map_size=5,
        n_predator=2,
        n_capture=1,
        n_prey=1,
        vision=0,
        max_timesteps=40,
        freeze_predator_touched=True,
        freeze_capture_touched=True,
        freeze_captured=True,
        time_reward=-0.05,
        capture_reward=0.0,
        onehot_pos=False,
    ):
        self.map_size = map_size
        self.n_predator = n_predator
        self.n_capture = n_capture
        self.n_prey = n_prey
        self.vision = vision
        self.max_timesteps = max_timesteps
        self.freeze_predator_touched = freeze_predator_touched
        self.freeze_capture_touched = freeze_capture_touched
        self.freeze_captured = freeze_captured
        if freeze_capture_touched and not freeze_captured:
            raise ValueError
        self.onehot_pos = onehot_pos

        self.time_reward = time_reward
        self.capture_reward = capture_reward

        self.t = 0
        self.predator_pos: List[Position] = []
        self.capture_pos: List[Position] = []
        self.prey_pos: List[Position] = []
        self.has_captured = [False for _ in range(self.n_capture)]

        self._observation_spaces: Dict[str, gym.spaces.Space] = {}
        self._action_spaces: Dict[str, gym.spaces.Space] = {}
        for i in range(self.n_predator):
            self._observation_spaces[f"predator{i}"] = gym.spaces.Dict(
                {
                    "vision": gym.spaces.Box(
                        0, 1, (4, self.vision * 2 + 1, self.vision * 2 + 1)
                    ),
                    "pos": gym.spaces.Discrete(self.map_size**2)
                    if onehot_pos
                    else gym.spaces.Box(-1, 1, (2,)),
                }
            )
            self._action_spaces[f"predator{i}"] = gym.spaces.Discrete(5)
        for i in range(self.n_capture):
            self._observation_spaces[f"capture{i}"] = gym.spaces.Dict(
                {
                    "pos": gym.spaces.Discrete(self.map_size**2)
                    if onehot_pos
                    else gym.spaces.Box(-1, 1, (2,))
                }
            )
            self._action_spaces[f"capture{i}"] = gym.spaces.Discrete(6)

    @property
    def observation_spaces(self) -> Dict[str, gym.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[str, gym.Space]:
        return self._action_spaces

    def get_obs(self):
        # 4 channels: predator, capture, prey, map boundary
        global_state = np.zeros(
            shape=(4, self.map_size + 2 * self.vision, self.map_size + 2 * self.vision)
        )
        for pos in self.predator_pos:
            global_state[0, pos.y + self.vision, pos.x + self.vision] = 1
        for pos in self.capture_pos:
            global_state[1, pos.y + self.vision, pos.x + self.vision] = 1
        for pos in self.prey_pos:
            global_state[2, pos.y + self.vision, pos.x + self.vision] = 1

        if self.vision > 0:
            global_state[3] = 1
            global_state[3, self.vision : -self.vision, self.vision : -self.vision] = 0

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

        for i in range(self.n_capture):
            pos = self.capture_pos[i]
            ret[f"capture{i}"] = {
                "pos": np.array(pos.y * self.map_size + pos.x)
                if self.onehot_pos
                else np.asarray(
                    (
                        (pos.x / (self.map_size - 1) * 2) - 1,
                        (pos.y / (self.map_size - 1) * 2) - 1,
                    )
                )
            }

        return ret

    def reset(self, seed: int | None = None):
        self.predator_pos.clear()
        self.capture_pos.clear()
        self.prey_pos.clear()

        self.has_captured = [False for _ in range(self.n_capture)]
        self.t = 0

        rng = np.random.default_rng(
            np.random.randint(2**30) if seed is None else seed
        )
        loc_flat_idx = rng.choice(
            self.map_size**2,
            size=self.n_predator + self.n_capture + self.n_prey,
            replace=False,
        )
        yy, xx = np.unravel_index(loc_flat_idx, (self.map_size, self.map_size))

        for i in range(self.n_predator):
            self.predator_pos.append(Position(yy[i], xx[i]))
        for i in range(self.n_capture):
            self.capture_pos.append(
                Position(yy[self.n_predator + i], xx[self.n_predator + i])
            )
        for i in range(self.n_prey):
            self.prey_pos.append(
                Position(
                    yy[self.n_predator + self.n_capture + i],
                    xx[self.n_predator + self.n_capture + i],
                )
            )

        return self.get_obs()

    def step(self, actions):
        self.t += 1

        for key in actions.keys():
            if not (
                (key[:-1] == "predator" and 0 <= int(key[-1]) < self.n_predator)
                or (key[:-1] == "capture" and 0 <= int(key[-1]) < self.n_capture)
            ):
                raise ValueError(f"unexpected action key: {key}")

        prey_set = set(self.prey_pos)
        rewards = {
            agent_id: self.time_reward for agent_id in self.observation_spaces.keys()
        }

        for i in range(self.n_predator):
            if self.predator_pos[i] in prey_set:
                rewards[f"predator{i}"] = 0
                if self.freeze_predator_touched:
                    continue

            if 0 <= actions[f"predator{i}"] < 4:
                if (
                    self.predator_pos[i] + CARDINALS[actions[f"predator{i}"]]
                ).in_bounds(self.map_size):
                    self.predator_pos[i] += CARDINALS[actions[f"predator{i}"]]
                    if self.predator_pos[i] in prey_set:
                        rewards[f"predator{i}"] = 0
            elif actions[f"predator{i}"] != 4:
                raise ValueError(
                    f"unexpected action for predator{i}: {actions[f'predator{i}']}"
                )

        for i in range(self.n_capture):
            if self.capture_pos[i] in prey_set:
                if self.has_captured[i]:
                    rewards[f"capture{i}"] = 0
                    if self.freeze_captured:
                        continue
                elif actions[f"capture{i}"] == 5:
                    self.has_captured[i] = True
                    rewards[f"capture{i}"] = self.capture_reward

                if self.freeze_capture_touched:
                    continue

            if 0 <= actions[f"capture{i}"] < 4:
                if (self.capture_pos[i] + CARDINALS[actions[f"capture{i}"]]).in_bounds(
                    self.map_size
                ):
                    self.capture_pos[i] += CARDINALS[actions[f"capture{i}"]]
            elif actions[f"capture{i}"] not in (4, 5):
                raise ValueError(
                    f"unexpected action for capture{i}: {actions[f'capture{i}']}"
                )

        done = (
            all(self.has_captured) and all(pos in prey_set for pos in self.predator_pos)
        ) or self.t >= self.max_timesteps

        return self.get_obs(), rewards, done, {}

    def render(self):
        from termcolor import colored

        predator_set = set(self.predator_pos)
        capture_set = set(self.capture_pos)
        prey_set = set(self.prey_pos)

        print("=" * (self.map_size + 2))
        for y in range(self.map_size):
            line = ""
            for x in range(self.map_size):
                if Position(x, y) in capture_set:
                    n_units = sum(
                        pos == Position(x, y)
                        for pos in self.predator_pos + self.capture_pos
                    )
                    if n_units == 1:
                        char = "^"
                    elif n_units <= 9:
                        char = str(n_units)
                    else:
                        char = "@"
                    if Position(x, y) in prey_set:
                        if any(
                            not self.has_captured[i]
                            and self.capture_pos[i] == Position(x, y)
                            for i in range(self.n_capture)
                        ):
                            line += colored(char, color="yellow")
                        else:
                            line += colored(char, color="green")
                    else:
                        line += colored(char, color="red")
                elif Position(x, y) in predator_set:
                    if Position(x, y) in prey_set:
                        line += colored("o", color="yellow")
                    else:
                        line += colored("o", color="red")
                else:
                    if Position(x, y) in prey_set:
                        line += colored("*", color="red")
                    else:
                        line += colored(".", color="white")
            print("|" + line + "|")
        print("=" * (self.map_size + 2))
