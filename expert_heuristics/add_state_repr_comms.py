import pickle
from collections import defaultdict

import blosc
import numpy as np
from sklearn.cluster import KMeans

from envs.base_multi_agent import MultiAgentEnv
from utils import PROJECT_DIR, encode_samples, get_agent_class


def add_comms(env: MultiAgentEnv, demo_filename, comm_size=26):
    with open(demo_filename, "rb") as f:
        data = pickle.load(f)
    n = len(data["terminal"])

    datasets = defaultdict(dict)
    for agent_id in data["obs"].keys():
        if agent_id == "global_state":
            continue
        squashed_obs = encode_samples(
            data["obs"][agent_id], env.observation_spaces[agent_id]
        ).numpy()

        # adds observations from preceding timesteps (padding episode starts with zeros)
        framestacked = np.zeros((2,) + squashed_obs.shape)
        mask = np.ones(n)
        for i in range(2):
            framestacked[i, i:] = mask[: n - i, None] * squashed_obs[: n - i]
            mask[np.arange(n - i)[data["terminal"][: n - i]] - i] = 0

        flattened = np.moveaxis(framestacked, 0, 1).reshape(len(data["terminal"]), -1)
        datasets[get_agent_class(agent_id)][agent_id] = flattened

    new_data = {
        "obs": {
            agent_id: {"comm_obs": {}}
            for agent_id in data["obs"].keys()
            if agent_id != "global_state"
        },
        "actions": {
            agent_id: {}
            for agent_id in data["actions"].keys()
            if agent_id != "global_state"
        },
        "terminal": data["terminal"],
    }
    models = {}
    for agent_id in data["obs"].keys():
        if agent_id == "global_state":
            continue
        agent_class = get_agent_class(agent_id)
        if agent_class in models:
            kmeans = models[agent_class]
        else:
            kmeans = KMeans(n_clusters=comm_size, n_init="auto")
            kmeans.fit(np.concatenate(list(datasets[agent_class].values()), axis=0))
            models[agent_class] = kmeans

        clusters = kmeans.predict(datasets[agent_class][agent_id])

        new_data["obs"][agent_id]["env_obs"] = data["obs"][agent_id]
        for other_id in data["obs"].keys():
            if other_id == "global_state":
                continue
            if agent_id == other_id:
                new_data["actions"][agent_id]["env_action"] = data["actions"][agent_id]
                new_data["actions"][agent_id]["comm_action"] = clusters
            else:
                i = 0
                while (name := f"other_{agent_class}_{i}") in new_data["obs"][other_id][
                    "comm_obs"
                ].keys():
                    i += 1
                new_data["obs"][other_id]["comm_obs"][name] = np.zeros_like(clusters)
                new_data["obs"][other_id]["comm_obs"][name][1:] = clusters[:-1]
                # clear comms at episode starts
                ep_starts = np.zeros(n, dtype=bool)
                ep_starts[0] = True
                ep_starts[1:] = data["terminal"][:-1]
                new_data["obs"][other_id]["comm_obs"][name][ep_starts] = 0

    return new_data


def main():
    from envs import FireCommander, PredatorCapture, PredatorPrey

    spec = {
        PredatorPrey: ("predator_prey", 7),
        PredatorCapture: ("predator_capture", 7),
        FireCommander: ("fire_commander", 7),
    }
    for env, (name, dim) in spec.items():
        data = add_comms(
            env(),
            f"{PROJECT_DIR}/data/{name}_demos.dat",
            comm_size=dim,
        )
        with open(f"{PROJECT_DIR}/data/{name}_heuristic_comm_demos.dat", "wb") as f:
            f.write(blosc.compress(pickle.dumps(data)))


if __name__ == "__main__":
    main()
