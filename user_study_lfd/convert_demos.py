import copy
import glob
import json
import pickle

from envs import FireCommander
from envs.base_multi_agent import CARDINALS, Position
from envs.comm_wrapper import add_uniform_comms
from utils import stack_dicts

HYPERPARAMS = {
    8: {"n_perception": 3, "n_action": 2, "vision": 1},
    10: {"n_perception": 2, "n_action": 2, "vision": 1},
    20: {"n_perception": 4, "n_action": 5, "vision": 2},
}
RAW_ACTION_MAP = {
    0: 4,
    1: 0,  # UP
    2: 2,  # DOWN
    3: 1,  # RIGHT
    4: 3,  # LEFT
    5: 5,  # water
}


def convert_no_comm(filename):
    with open(filename, "rb") as f:
        raw = pickle.load(f)
    n_timesteps = max(int(k[1:]) for k in raw.keys()) + 1
    size = raw["t0"]["state_space"].shape[1]

    n_perception = HYPERPARAMS[size]["n_perception"]
    n_action = HYPERPARAMS[size]["n_action"]
    vision = HYPERPARAMS[size]["vision"]
    assert len(raw) == n_timesteps

    buf = []

    for t in range(n_timesteps - 1):
        env = FireCommander(
            map_size=size, n_perception=n_perception, n_action=n_action, vision=vision
        )
        env.reset()
        env.perception_pos = []
        env.action_pos = []
        env.fire_pos = set()

        actions = {}

        for i in range(n_perception):
            y, x = map(int, raw[f"t{t}"]["state_space"][1 + i].nonzero())
            env.perception_pos.append(Position(x, y))

            raw_action = raw[f"t{t + 1}"]["action_space"][i]
            real_action = RAW_ACTION_MAP[int(raw_action)]
            actions[f"perception{i}"] = real_action

        for j in range(n_action):
            y, x = map(int, raw[f"t{t}"]["state_space"][1 + n_perception + j].nonzero())
            env.action_pos.append(Position(x, y))

            raw_action = raw[f"t{t + 1}"]["action_space"][n_perception + j]
            real_action = RAW_ACTION_MAP[int(raw_action)]
            actions[f"action{j}"] = real_action

        fire_x, fire_y = raw[f"t{t}"]["state_space"][0].nonzero()
        for y, x in zip(fire_x, fire_y):
            env.fire_pos.add(Position(int(x), int(y)))

        nxt = copy.deepcopy(env)
        nxt.step(actions)
        for i in range(n_perception):
            if actions[f"perception{i}"] == 4:
                assert env.perception_pos[i] == nxt.perception_pos[i]
            else:
                new_pos = env.perception_pos[i] + CARDINALS[actions[f"perception{i}"]]
                assert not new_pos.in_bounds(size) or new_pos == nxt.perception_pos[i]

        for j in range(n_action):
            if actions[f"action{j}"] in (4, 5):
                assert env.action_pos[j] == nxt.action_pos[j]
            else:
                new_pos = env.action_pos[j] + CARDINALS[actions[f"action{j}"]]
                assert not new_pos.in_bounds(size) or new_pos == nxt.action_pos[j]

        buf.append(
            {
                "obs": env.get_obs(),
                "actions": actions,
                "new_obs": nxt.get_obs(),
                "terminal": True if t == n_timesteps - 2 else False,
            }
        )

    assert all(not transition["terminal"] for transition in buf[:-1])
    assert buf[-1]["terminal"]
    return buf


def convert_with_comm(filename):
    with open(filename, "rb") as f:
        raw = pickle.load(f)
    n_timesteps = max(int(k[1:]) for k in raw.keys()) + 1
    size = raw["t0"]["state_space"].shape[1]

    n_perception = HYPERPARAMS[size]["n_perception"]
    n_action = HYPERPARAMS[size]["n_action"]
    vision = HYPERPARAMS[size]["vision"]
    assert len(raw) == n_timesteps

    buf = []

    env = add_uniform_comms(
        FireCommander(
            map_size=size,
            n_perception=n_perception,
            n_action=n_action,
            vision=vision,
        ),
        comm_dim=26,
        include_null_comm=True,
    )
    env.reset()
    for t in range(n_timesteps - 1):
        env.base_env.perception_pos = []
        env.base_env.action_pos = []
        env.fire_pos = set()

        base_env: FireCommander = env.base_env

        actions = {}

        for i in range(n_perception):
            y, x = map(int, raw[f"t{t}"]["state_space"][1 + i].nonzero())
            base_env.perception_pos.append(Position(x, y))

            raw_action = raw[f"t{t + 1}"]["action_space"][i]
            real_action = RAW_ACTION_MAP[int(raw_action)]

            actions[f"perception{i}"] = {
                "env_action": real_action,
                "comm_action": raw[f"t{t + 1}"]["message_space"][i].astype(int) - 1,
            }

        for j in range(n_action):
            y, x = map(int, raw[f"t{t}"]["state_space"][1 + n_perception + j].nonzero())
            base_env.action_pos.append(Position(x, y))

            raw_action = raw[f"t{t + 1}"]["action_space"][n_perception + j]
            real_action = RAW_ACTION_MAP[int(raw_action)]

            actions[f"action{j}"] = {
                "env_action": real_action,
                "comm_action": raw[f"t{t + 1}"]["message_space"][
                    n_perception + j
                ].astype(int)
                - 1,
            }

        fire_x, fire_y = raw[f"t{t}"]["state_space"][0].nonzero()
        for y, x in zip(fire_x, fire_y):
            base_env.fire_pos.add(Position(int(x), int(y)))

        nxt = copy.deepcopy(env)
        nxt.step(actions)

        buf.append(
            {
                "obs": env.get_obs(),
                "actions": actions,
                "new_obs": nxt.get_obs(),
                "terminal": True if t == n_timesteps - 2 else False,
            }
        )
        env = nxt

    assert all(not transition["terminal"] for transition in buf[:-1])
    assert buf[-1]["terminal"]
    return buf


def main():
    with_comm = True
    level = "hard"
    min_rating = 0
    wins_only = False

    name = (
        f"{level}_{'with_comm' if with_comm else 'no_comm'}_{min_rating=}"
        f"{'_wins_only' if wins_only else ''}"
    )
    print(f"Converting with spec {name}")

    with open("../demos/user_study_raw/ratings.json", "rb") as f:
        ratings = json.load(f)
    ratings = {int(k): v for k, v in ratings.items()}

    saved = []
    count = 0
    for filename in glob.glob("../demos/user_study_raw/*.pkl"):
        if with_comm != ("withComm" in filename):
            continue
        if level not in filename:
            continue
        if wins_only and "lose" in filename:
            continue
        idx = int(filename.split("_")[-3])
        rating = ratings[idx]

        if rating < min_rating:
            continue

        if with_comm:
            saved.extend(convert_with_comm(filename))
        else:
            saved.extend(convert_no_comm(filename))

        count += 1
    print(f"Converted {count} demonstrations")
    dataset = stack_dicts(saved)
    with open(f"../demos/user_study/{name}.pickle", "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
