from collections import defaultdict
from typing import Dict

import gym
import numpy as np

from envs.base_multi_agent import MultiAgentEnv
from utils import get_agent_class


def null_comm(comm_space: gym.Space, has_null_comm):
    if isinstance(comm_space, gym.spaces.Discrete):
        if has_null_comm:
            return np.zeros(comm_space.n - 1)
        else:
            return np.array(0)
    elif isinstance(comm_space, gym.spaces.Box):
        return np.zeros(comm_space.shape)
    elif isinstance(comm_space, gym.spaces.Dict):
        ret = {}
        for k in comm_space.spaces.keys():
            ret[k] = null_comm(comm_space[k], has_null_comm)
        return ret
    else:
        raise TypeError


class CommWrapper(MultiAgentEnv):
    def __init__(
        self,
        base_env: MultiAgentEnv,
        comm_spaces: Dict[str, gym.Space],
        has_null_comm=False,
    ):
        self.base_env = base_env
        self.comm_spaces = comm_spaces
        self.has_null_comm = has_null_comm

        self.ids_to_commid = {}
        self.last_comms = None

        self._obs_spaces = {}
        for agent_id, env_obs_space in base_env.observation_spaces.items():
            new_space = {"env_obs": env_obs_space}
            new_comms = {}

            count = defaultdict(int)
            for other_id, other_comm_space in self.comm_spaces.items():
                if other_id == agent_id:
                    continue
                other_class = get_agent_class(other_id)

                name = f"other_{other_class}_{count[other_class]}"
                if self.has_null_comm and isinstance(
                    other_comm_space, gym.spaces.Discrete
                ):
                    new_comms[name] = gym.spaces.Box(
                        0, 1, shape=(other_comm_space.n - 1,)
                    )
                else:
                    new_comms[name] = other_comm_space
                self.ids_to_commid[(agent_id, other_id)] = name
                count[other_class] += 1

            if new_comms:
                new_space["comm_obs"] = gym.spaces.Dict(new_comms)
            self._obs_spaces[agent_id] = gym.spaces.Dict(new_space)

        self._action_spaces = {}
        for agent_id, env_action_space in self.base_env.action_spaces.items():
            if agent_id in self.comm_spaces.keys():
                self._action_spaces[agent_id] = gym.spaces.Dict(
                    {
                        "env_action": env_action_space,
                        "comm_action": self.comm_spaces[agent_id],
                    }
                )
            else:
                self._action_spaces[agent_id] = gym.spaces.Dict(
                    {
                        "env_action": env_action_space,
                    }
                )

    @property
    def observation_spaces(self) -> Dict[str, gym.Space]:
        return self._obs_spaces

    @property
    def action_spaces(self) -> Dict[str, gym.Space]:
        return self._action_spaces

    def get_obs(self):
        base_obs = self.base_env.get_obs()
        comm_obs = {}
        for agent_id in base_obs.keys():
            if agent_id == "global_state":
                continue
            if self.last_comms is None:
                comm_obs[agent_id] = {
                    self.ids_to_commid[(agent_id, other_id)]: null_comm(
                        self.comm_spaces[other_id], self.has_null_comm
                    )
                    for other_id in self.comm_spaces.keys()
                    if other_id != agent_id
                }
            else:
                if self.has_null_comm:
                    comm_obs[agent_id] = {}
                    for other_id, other_comm in self.last_comms.items():
                        if agent_id == other_id:
                            continue
                        other_comm_space = self.comm_spaces[other_id]
                        if isinstance(other_comm_space, gym.spaces.Discrete):
                            new_comm = np.zeros(other_comm_space.n - 1)
                            if other_comm != other_comm_space.n - 1:
                                new_comm[other_comm] = 1
                        else:
                            new_comm = other_comm
                        comm_obs[agent_id][
                            self.ids_to_commid[(agent_id, other_id)]
                        ] = new_comm
                    # comm_obs[agent_id] = {
                    #     self.ids_to_commid[(agent_id, other_id)]: other_comm
                    #     for other_id, other_comm in self.last_comms.items()
                    #     if other_id != agent_id
                    # }
                else:
                    comm_obs[agent_id] = {
                        self.ids_to_commid[(agent_id, other_id)]: other_comm
                        for other_id, other_comm in self.last_comms.items()
                        if other_id != agent_id
                    }
        return {
            agent_id: {
                "env_obs": agent_obs,
                "comm_obs": comm_obs[agent_id],
            }
            for agent_id, agent_obs in base_obs.items()
            if agent_id != "global_state"
        }

    def reset(self, seed: int | None = None):
        self.base_env.reset(seed)
        self.last_comms = None
        return self.get_obs()

    def step(self, actions):
        base_actions = {}
        new_comms = {}
        for agent_id, action in actions.items():
            base_actions[agent_id] = action["env_action"]
            if "comm_action" in action.keys():
                new_comms[agent_id] = action["comm_action"]
        self.last_comms = new_comms

        _, rewards, dones, infos = self.base_env.step(base_actions)
        return self.get_obs(), rewards, dones, infos


def add_uniform_comms(
    env: MultiAgentEnv, comm_dim: int, include_null_comm=False
) -> CommWrapper:
    return CommWrapper(
        env,
        comm_spaces={
            agent_id: gym.spaces.Discrete(comm_dim)
            for agent_id in env.observation_spaces.keys()
        },
        has_null_comm=include_null_comm,
    )
