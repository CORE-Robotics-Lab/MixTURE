import itertools
import pickle

import numpy as np
import tqdm
from scipy.optimize import linear_sum_assignment

from envs import PredatorPrey
from envs.base_multi_agent import CARDINALS, Direction, Position
from expert_heuristics.add_state_repr_comms import add_comms
from utils import stack_dicts


def assign_targets(unit_pos: list[Position], target_pos: list[Position]):
    assert len(unit_pos) == len(target_pos)

    cost_mat = np.empty((len(unit_pos), len(target_pos)))
    for i, p1 in enumerate(unit_pos):
        for j, p2 in enumerate(target_pos):
            cost_mat[i, j] = p1.manhattan(p2)
    _, ind = linear_sum_assignment(cost_mat)
    return [target_pos[i] for i in ind]


def get_explore_actions(
    vis, unit_pos, been_observed, iters=10, seed=None
) -> list[Direction]:
    rng = np.random.default_rng(seed)

    size = been_observed.shape[0]

    # greedily select high-level targets which have lots of unrevealed tiles nearby,
    # while avoiding overlaps
    v = np.zeros_like(been_observed, dtype=float)
    for x, y in itertools.product(range(size), repeat=2):
        if not been_observed[y, x]:
            v[max(0, y - vis) : y + vis + 1, max(0, x - vis) : x + vis + 1] += 1

    # incentives revealing locations which have few unrevealed neighbors (helps prevent
    # unnecessary backtracking)
    v[1:] -= ~been_observed[:-1] * 0.1
    v[:-1] -= ~been_observed[1:] * 0.1
    v[:, 1:] -= ~been_observed[:, :-1] * 0.1
    v[:, :-1] -= ~been_observed[:, 1:] * 0.1

    targets = []
    score = 0
    for _ in range(len(unit_pos)):
        greedy_best_val = np.amax(v)
        score += greedy_best_val
        ys, xs = (v == greedy_best_val).nonzero()
        best_idx = rng.choice(len(ys))
        y, x = ys[best_idx].item(), xs[best_idx].item()
        targets.append(Position(x, y))
        v[max(0, y - vis) : y + vis + 1, max(0, x - vis) : x + vis + 1] -= 1

    unit_targets = assign_targets(unit_pos, targets)

    # greedily select low-level actions which 1) reveal max # tiles in one step, 2) head
    # towards high-level target, 3) close to teammtaes, 4) randomly
    best_actions = None
    best_score = None
    for _ in range(iters):
        idx = np.arange(len(unit_pos))
        rng.shuffle(idx)

        v2 = (~been_observed).astype(float)

        # incentives revealing locations which have few unrevealed neighbors (helps
        # prevent unnecessary backtracking)
        v2[1:] -= ~been_observed[:-1] * 0.1
        v2[:-1] -= ~been_observed[1:] * 0.1
        v2[:, 1:] -= ~been_observed[:, :-1] * 0.1
        v2[:, :-1] -= ~been_observed[:, 1:] * 0.1
        v2[v2 < 0] = 0

        total_score = None
        actions = [None for _ in range(len(unit_pos))]
        revealed_mask = np.zeros(v2.shape, dtype=bool)
        for i in idx:
            pos = unit_pos[i]
            action_scores = []

            for direction in CARDINALS:
                end_pos = pos + direction

                new_revealed_mask = revealed_mask.copy()
                new_revealed_mask[
                    max(0, end_pos.y - vis) : end_pos.y + vis + 1,
                    max(0, end_pos.x - vis) : end_pos.x + vis + 1,
                ] = True
                revealed_score = np.sum((~revealed_mask & new_revealed_mask) * v2)

                target_closeness = -end_pos.manhattan(unit_targets[i])
                action_scores.append(
                    (
                        end_pos.in_bounds(size),
                        revealed_score,
                        target_closeness,
                        # ally_closeness,
                        rng.random(),
                    )
                )

            best_action_i = max(
                range(len(action_scores)), key=lambda j: action_scores[j]
            )
            best_end_pos = pos + CARDINALS[best_action_i]
            revealed_mask[
                max(0, best_end_pos.y - vis) : best_end_pos.y + vis + 1,
                max(0, best_end_pos.x - vis) : best_end_pos.x + vis + 1,
            ] = True

            best_action = CARDINALS[best_action_i].ordinal()
            actions[i] = best_action

            if total_score is None:
                total_score = list(action_scores[best_action_i])
            else:
                total_score = [
                    total_score[k] + action_scores[best_action_i][k]
                    for k in range(len(total_score))
                ]

        if best_score is None or total_score > best_score:
            best_score = total_score
            best_actions = actions
    return best_actions


def run_ep(env: PredatorPrey, render=False, save=False, seed=None):
    rng = np.random.default_rng(seed)

    env_obs = env.reset(rng.integers(2**32).item())

    map_size = env.map_size
    vis = env.vision

    prey = np.zeros((map_size, map_size), dtype=bool)
    been_observed = np.zeros((map_size, map_size), dtype=bool)

    timesteps = 0

    saved = []

    while True:
        timesteps += 1

        predator_pos = []
        predator_obs = []

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

        for pos, obs in zip(predator_pos, predator_obs):
            for ox in range(2 * vis + 1):
                for oy in range(2 * vis + 1):
                    real_x = pos.x - vis + ox
                    real_y = pos.y - vis + oy
                    if not Position(real_x, real_y).in_bounds(map_size):
                        continue

                    if obs[1, oy, ox]:
                        prey[real_y, real_x] = True
                    been_observed[real_y, real_x] = True

        nz = np.nonzero(prey)
        prey_pos = [Position(nz[1][i], nz[0][i]) for i in range(len(nz[0]))]

        if len(prey_pos) > 0:
            actions = {}
            for i, pos in enumerate(predator_pos):
                nearest = min(prey_pos, key=lambda p: p.manhattan(pos))
                actions[f"predator{i}"] = pos.direction_to(nearest).ordinal()
        else:
            actions = get_explore_actions(
                vis, predator_pos, been_observed, seed=rng.integers(2**32).item()
            )
            actions = {f"predator{i}": action for i, action in enumerate(actions)}

        if render:
            env.render()
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

    names = ["predatorprey_" + s for s in ("5x5", "10x10", "20x20")]
    map_sizes = [5, 10, 20]
    n_predators = [3, 6, 10]
    visions = [0, 1, 2]
    max_lens = [20, 80, 200]

    for name, map_size, n_predator, vision, max_len in zip(
        names, map_sizes, n_predators, visions, max_lens
    ):
        env = PredatorPrey(map_size, n_predator, 1, vision, max_len)

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

        stacked = stack_dicts(all_saved)
        print(f"Total Timesteps: {stacked['terminal'].shape[0]}")
        print(f"Total Episodes: {stacked['terminal'].sum()}")
        print(f"Mean Timesteps: {1 / stacked['terminal'].mean()}")

        with open(f"../demos/{name}.pickle", "wb") as f:
            pickle.dump(stacked, f)
        comm_dataset = add_comms(env, f"../demos/{name}.pickle", comm_size=26)
        with open(f"../demos/{name}_comm.pickle", "wb") as f:
            pickle.dump(comm_dataset, f)


if __name__ == "__main__":
    main()
