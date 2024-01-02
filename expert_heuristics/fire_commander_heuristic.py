import itertools
import pickle
import random
from dataclasses import dataclass, field
from typing import List, Union

# import blosc
import networkx as nx
import numpy as np
import tqdm
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy

from envs.base_multi_agent import CARDINALS, Direction, Position
from envs.fire_commander import FireCommander
from expert_heuristics.add_state_repr_comms import add_comms
from utils import stack_dicts


def pair_pos(starts: List[Position], targets: List[Position]):
    if len(starts) == 0 or len(targets) == 0:
        return {}

    graph = nx.Graph()
    for i1 in range(len(starts)):
        graph.add_node(i1)
    for i2 in range(len(targets)):
        graph.add_node(10000 + i2)
        for i1 in range(len(starts)):
            graph.add_edge(10000 + i2, i1, weight=starts[i1].manhattan(targets[i2]))

    matches = nx.algorithms.bipartite.minimum_weight_full_matching(graph)
    return {i: matches[i] - 10000 for i in range(len(starts)) if i in matches}


@dataclass
class Agent:
    id: int
    pos: Position
    obs: Union[np.ndarray, None] = field(repr=False)
    next_pos: Position = field(default=None, repr=False)


def run_ep(env_args, render=False, save=False, seed=None):
    rng = random.Random(seed)

    env = FireCommander(**env_args)
    obs = env.reset(rng.randrange(2**32))

    map_size = env_args["map_size"]
    vis = env_args["vision"]

    fire_prob = np.full((map_size, map_size), fill_value=0.1)
    actively_observed = np.zeros((map_size, map_size), dtype=bool)

    timesteps = 0

    saved = []

    while True:
        timesteps += 1

        p_agents = []
        a_agents = []

        for i in range(env.n_perception):
            if env.onehot_pos:
                y = obs[f"perception{i}"]["pos"] // env.map_size
                x = int(obs[f"perception{i}"]["pos"]) % env.map_size
            else:
                x = int((obs[f"perception{i}"]["pos"][0] + 1) * (env.map_size - 1) / 2)
                y = int((obs[f"perception{i}"]["pos"][1] + 1) * (env.map_size - 1) / 2)
            p_agents.append(Agent(i, Position(x, y), obs[f"perception{i}"]["vision"]))
        for i in range(env.n_action):
            if env.onehot_pos:
                y = obs[f"action{i}"]["pos"] // env.map_size
                x = int(obs[f"action{i}"]["pos"]) % env.map_size
            else:
                x = int((obs[f"action{i}"]["pos"][0] + 1) * (env.map_size - 1) / 2)
                y = int((obs[f"action{i}"]["pos"][1] + 1) * (env.map_size - 1) / 2)
            a_agents.append(Agent(i, Position(x, y), None))

        for agent in p_agents:
            for ox in range(2 * vis + 1):
                for oy in range(2 * vis + 1):
                    real_x = agent.pos.x - vis + ox
                    real_y = agent.pos.y - vis + oy
                    if not Position(real_x, real_y).in_bounds(map_size):
                        continue

                    fire_prob[real_y, real_x] = float(agent.obs[2, oy, ox])
                    actively_observed[real_y, real_x] = True

        fire_prob = np.maximum(fire_prob, gaussian_filter(fire_prob, sigma=0.5))
        entropies = np.empty_like(fire_prob)
        known_fire_pos = []

        for py in range(map_size):
            for px in range(map_size):
                entropies[py, px] = entropy([fire_prob[py, px], 1 - fire_prob[py, px]])
                if fire_prob[py, px] == 1:
                    known_fire_pos.append(Position(px, py))
                # fixes for edge cases, precision
                if not actively_observed[py, px]:
                    if fire_prob[py, px] <= 1e-4:
                        fire_prob[py, px] = 1e-4
                    fire_prob[py, px] = round(fire_prob[py, px], ndigits=5)

        if render:
            with np.printoptions(precision=5, suppress=True):
                print(fire_prob)
                print(entropies)
                print(known_fire_pos)

        percep_move_candidates = []
        for agent in p_agents:
            new_move_candidates = []
            best_dist = 9999
            for direction in CARDINALS:
                if not (agent.pos + direction).in_bounds(map_size):
                    continue
                dist = min((agent.pos + direction).manhattan(aa.pos) for aa in a_agents)
                if dist < best_dist:
                    if best_dist > vis + 2:
                        new_move_candidates.clear()
                    best_dist = dist
                if dist == best_dist or dist <= vis + 2:
                    new_move_candidates.append(direction)
            percep_move_candidates.append(new_move_candidates)

        if np.prod([len(entry) for entry in percep_move_candidates]) > 100:
            joint_actions = [
                [rng.choice(entry) for entry in percep_move_candidates]
                for _ in range(100)
            ]
        else:
            joint_actions = list(itertools.product(*percep_move_candidates))

        best_joint_action = None
        best_score = None
        for joint_action in joint_actions:
            end_pos = [agent.pos + joint_action[i] for i, agent in enumerate(p_agents)]

            score = 0

            revealed_pos = set()
            for i, bot in enumerate(p_agents):
                for perp_dist in range(-vis, vis + 1):
                    forward: Direction = joint_action[i]
                    pos = bot.pos + (vis + 1) * forward + perp_dist * forward.rot90()
                    if pos.in_bounds(map_size):
                        revealed_pos.add(pos)
            rm = set()
            for i, bot in enumerate(p_agents):
                for pos in revealed_pos:
                    if abs(bot.pos.x - pos.x) <= 1 and abs(bot.pos.y - pos.y) <= 1:
                        rm.add(pos)
            revealed_pos -= rm
            score += sum(entropies[p.y, p.x] for p in revealed_pos)

            for py in range(map_size):
                for px in range(map_size):
                    for i in range(len(p_agents)):
                        score += (
                            1e-6
                            * entropies[py, px]
                            / (10 + end_pos[i].manhattan(Position(py, px)))
                        )

            if best_score is None or score > best_score:
                best_score = score
                best_joint_action = joint_action

        actions = {
            f"perception{i}": best_joint_action[i].ordinal()
            for i in range(len(p_agents))
        }

        fire_action_pairs = pair_pos(known_fire_pos, [agent.pos for agent in a_agents])
        global_targets = {
            f"action{v}": known_fire_pos[k] for k, v in fire_action_pairs.items()
        }

        for i, agent in enumerate(a_agents):
            if f"action{i}" not in global_targets.keys():
                global_targets[f"action{i}"] = min(
                    (other.pos for other in p_agents),
                    key=lambda other_pos: agent.pos.manhattan(other_pos),
                )

        for i, agent in enumerate(a_agents):
            if fire_prob[agent.pos.y, agent.pos.x] == 1:
                actions[f"action{i}"] = 5
                fire_prob[agent.pos.y, agent.pos.x] = 0
                continue

            target_pos = global_targets[f"action{i}"]

            nearby_fires = [
                fire_pos
                for fire_pos in known_fire_pos
                if agent.pos.chebyshev(fire_pos)
                and all(other.pos != fire_pos for other in a_agents)
            ]
            if len(nearby_fires) > 0:
                best_fire = min(
                    nearby_fires, key=lambda fire_pos: fire_pos.manhattan(target_pos)
                )
                actions[f"action{i}"] = agent.pos.direction_to(best_fire).ordinal()
                # if debug:
                #     print(f"action{i} ext (greedy) -> {best_fire} .. {target_pos}")
                continue

            if agent.pos.manhattan(target_pos) <= 2:
                nearby_fires = [
                    fire_pos
                    for fire_pos in known_fire_pos
                    if target_pos.manhattan(fire_pos) <= 2
                    and all(other.pos != fire_pos for other in a_agents)
                ]
                if len(nearby_fires) > 0:
                    best_fire = min(
                        nearby_fires,
                        key=lambda fire_pos: fire_pos.manhattan(target_pos),
                    )
                    actions[f"action{i}"] = agent.pos.direction_to(best_fire).ordinal()
                    # if debug:
                    #     print(f"action{i} ext (assign) -> {best_fire} .. {target_pos}")
                    continue

            actions[f"action{i}"] = agent.pos.direction_to(target_pos).ordinal()
            # if debug:
            #     print(f"action{i} nav -> {target_pos}")

        if render:
            env.render()
        old_obs = obs

        obs, rewards, done, _ = env.step(actions)
        if save:
            saved.append(
                {
                    "obs": old_obs.copy(),
                    "actions": actions.copy(),
                    "new_obs": obs.copy(),
                    "terminal": done,
                }
            )

        if done:
            if render:
                env.render()
            break

    if save:
        terminal = np.zeros(shape=timesteps, dtype=bool)
        terminal[-1] = True
        return timesteps, saved
    else:
        return timesteps


def main():
    names = [
        "firecommander_5x5",
        "firecommander_10x10_ez",
        "firecommander_10x10",
        "firecommander_20x20_ez",
        "firecommander_20x20",
    ]
    env_args = [
        {
            "map_size": 5,
            "n_perception": 2,
            "n_action": 1,
            "n_fires": 1,
            "vision": 1,
            "max_timesteps": 80,
        },
        {
            "map_size": 10,
            "n_perception": 3,
            "n_action": 3,
            "n_fires": 1,
            "vision": 1,
            "max_timesteps": 80,
        },
        {
            "map_size": 10,
            "n_perception": 3,
            "n_action": 3,
            "n_fires": 2,
            "vision": 1,
            "max_timesteps": 80,
        },
        {
            "map_size": 20,
            "n_perception": 6,
            "n_action": 4,
            "n_fires": 1,
            "vision": 2,
            "max_timesteps": 800,
        },
        {
            "map_size": 20,
            "n_perception": 6,
            "n_action": 4,
            "n_fires": 3,
            "vision": 2,
            "max_timesteps": 800,
        },
    ]

    for name, args in zip(names, env_args):
        all_saved = []
        desired_timesteps = 10_000
        curr_timesteps = 0
        with tqdm.tqdm(total=desired_timesteps, smoothing=0.05) as pbar:
            while curr_timesteps < desired_timesteps:
                timesteps, data = run_ep(args, save=True)
                all_saved.extend(data)
                pbar.update(timesteps)
                curr_timesteps += timesteps

        dataset = stack_dicts(all_saved)
        print(f"Total Timesteps: {dataset['terminal'].shape[0]}")
        print(f"Total Episodes: {dataset['terminal'].sum()}")
        print(f"Mean Timesteps: {1 / dataset['terminal'].mean()}")

        with open(f"../demos/{name}.pickle", "wb") as f:
            pickle.dump(dataset, f)

        comm_dataset = add_comms(
            FireCommander(**args), f"../demos/{name}.pickle", comm_size=26
        )
        with open(f"../demos/{name}_comm.pickle", "wb") as f:
            pickle.dump(comm_dataset, f)


if __name__ == "__main__":
    main()
