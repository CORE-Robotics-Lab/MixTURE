import copy
import pickle

import numpy as np
import tqdm
from scipy.optimize import linear_sum_assignment

from envs import PredatorCapture
from envs.base_multi_agent import CARDINALS, Position
from expert_heuristics.add_state_repr_comms import add_comms
from expert_heuristics.predator_prey_heuristic import get_explore_actions
from utils import stack_dicts


def assign_targets(unit_pos: list[Position], target_pos: list[Position]):
    assert len(unit_pos) == len(target_pos)

    cost_mat = np.empty((len(unit_pos), len(target_pos)))
    for i, p1 in enumerate(unit_pos):
        for j, p2 in enumerate(target_pos):
            cost_mat[i, j] = p1.manhattan(p2)
    _, ind = linear_sum_assignment(cost_mat)
    return [target_pos[i] for i in ind]


def run_ep(env: PredatorCapture, render=False, save=False, seed=None):
    rng = np.random.default_rng(seed)

    env_obs = env.reset(rng.integers(2**32).item())

    map_size = env.map_size
    vis = env.vision

    prey = np.zeros((map_size, map_size), dtype=bool)
    been_observed = np.zeros((map_size, map_size), dtype=bool)

    timesteps = 0

    saved = []

    last_capture_pos: list[Position] = None

    while True:
        timesteps += 1

        predator_pos = []
        predator_obs = []
        capture_pos = []

        for i in range(env.n_predator):
            if env.onehot_pos:
                y = env_obs[f"predator{i}"]["pos"] // env.map_size
                x = int(env_obs[f"predator{i}"]["pos"]) % env.map_size
            else:
                x = int(
                    (env_obs[f"predator{i}"]["pos"][0] + 1) * (env.map_size - 1) / 2
                )
                y = int(
                    (env_obs[f"predator{i}"]["pos"][1] + 1) * (env.map_size - 1) / 2
                )
            predator_pos.append(Position(x, y))
            predator_obs.append(env_obs[f"predator{i}"]["vision"])
        for i in range(env.n_capture):
            if env.onehot_pos:
                y = env_obs[f"capture{i}"]["pos"] // env.map_size
                x = int(env_obs[f"capture{i}"]["pos"]) % env.map_size
            else:
                x = int((env_obs[f"capture{i}"]["pos"][0] + 1) * (env.map_size - 1) / 2)
                y = int((env_obs[f"capture{i}"]["pos"][1] + 1) * (env.map_size - 1) / 2)
            capture_pos.append(Position(x, y))

        if last_capture_pos is not None:
            for i, pos in enumerate(capture_pos):
                if last_capture_pos[i] == pos:
                    prey[pos.y, pos.x] = True
                else:
                    been_observed[last_capture_pos[i].y, last_capture_pos[i].x] = True

        for pos, obs in zip(predator_pos, predator_obs):
            for ox in range(2 * vis + 1):
                for oy in range(2 * vis + 1):
                    real_x = pos.x - vis + ox
                    real_y = pos.y - vis + oy
                    if not Position(real_x, real_y).in_bounds(map_size):
                        continue

                    if obs[2, oy, ox]:
                        prey[real_y, real_x] = True
                    been_observed[real_y, real_x] = True

        nz = np.nonzero(prey)
        prey_pos = [Position(nz[1][i], nz[0][i]) for i in range(len(nz[0]))]

        if len(prey_pos) > 0:
            actions = {}
            for i, pos in enumerate(predator_pos):
                nearest_prey = min(prey_pos, key=lambda p: p.manhattan(pos))
                actions[f"predator{i}"] = pos.direction_to(nearest_prey).ordinal()
            for i, pos in enumerate(capture_pos):
                nearest_prey = min(prey_pos, key=lambda p: p.manhattan(pos))
                actions[f"capture{i}"] = pos.direction_to(nearest_prey).ordinal()
                if actions[f"capture{i}"] == 4:
                    actions[f"capture{i}"] = 5
        else:
            actions = {}
            predator_actions = get_explore_actions(
                vis, predator_pos, been_observed, seed=rng.integers(2**32).item()
            )

            new_been_observed = been_observed.copy()
            for i in range(env.n_predator):
                new_pos = predator_pos[i] + CARDINALS[predator_actions[i]]
                new_been_observed[
                    max(0, new_pos.y - vis) : new_pos.y + vis + 1,
                    max(0, new_pos.x - vis) : new_pos.x + vis + 1,
                ] = True
                actions[f"predator{i}"] = predator_actions[i]

            capture_actions = get_explore_actions(
                0, capture_pos, new_been_observed, seed=rng.integers(2**32).item()
            )
            for i in range(env.n_capture):
                actions[f"capture{i}"] = capture_actions[i]

        last_capture_pos = copy.deepcopy(capture_pos)

        if render:
            env.render()
            print(timesteps)
        old_obs = env_obs

        env_obs, rewards, done, _ = env.step(actions)
        if save:
            saved.append(
                {
                    "obs": old_obs.copy(),
                    "actions": actions.copy(),
                    "new_obs": env_obs.copy(),
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
    rng = np.random.default_rng(42)

    names = ["predatorcapture_" + s for s in ("5x5", "10x10", "20x20")]
    map_sizes = [5, 10, 20]
    n_predators = [2, 3, 6]
    n_captures = [1, 3, 4]
    visions = [0, 1, 2]
    max_lens = [20, 80, 200]

    for name, map_size, n_predator, n_capture, vision, max_len in zip(
        names, map_sizes, n_predators, n_captures, visions, max_lens
    ):
        env = PredatorCapture(map_size, n_predator, n_capture, 1, vision, max_len)

        all_saved = []
        desired_timesteps = 10_000
        curr_timesteps = 0
        with tqdm.tqdm(total=desired_timesteps, smoothing=0.05) as pbar:
            while curr_timesteps < desired_timesteps:
                timesteps, data = run_ep(
                    env, save=True, seed=rng.integers(2**32).item()
                )
                all_saved.extend(data)
                pbar.update(timesteps)
                curr_timesteps += timesteps

        dataset = stack_dicts(all_saved)
        print(f"Total Timesteps: {dataset['terminal'].shape[0]}")
        print(f"Total Episodes: {dataset['terminal'].sum()}")
        print(f"Mean Timesteps: {1 / dataset['terminal'].mean()}")

        with open(f"../demos/{name}.pickle", "wb") as f:
            pickle.dump(dataset, f)

        comm_dataset = add_comms(env, f"../demos/{name}.pickle", comm_size=26)
        with open(f"../demos/{name}_comm.pickle", "wb") as f:
            pickle.dump(comm_dataset, f)


if __name__ == "__main__":
    main()
