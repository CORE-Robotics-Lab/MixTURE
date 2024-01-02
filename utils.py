import math
import os
import pickle
import random
import re
import warnings
from collections import defaultdict
from typing import List

import gym
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import trim_mean

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def squash(obj, dtype=torch.float32, device=torch.device("cpu"), preserve_batch=True):
    """
    Squashes an object recursively, typically preserving the batch dimension.
    """
    if isinstance(obj, dict):
        return torch.cat(
            [
                squash(
                    obj[k], dtype=dtype, device=device, preserve_batch=preserve_batch
                )
                for k in sorted(obj.keys())
            ],
            dim=1,
        )
    return torch.as_tensor(obj, dtype=dtype, device=device).flatten(
        start_dim=min(len(obj.shape) - 1, 1) if preserve_batch else 0
    )


def batchify(obj):
    """
    Adds a batch dimension (of size 1) to data for a single timestep
    """
    if isinstance(obj, dict):
        return {k: batchify(v) for k, v, in obj.items()}
    return np.expand_dims(obj, axis=0)


def encode_samples(samples: np.ndarray, space: gym.Space, device=torch.device("cpu")):
    if isinstance(space, gym.spaces.Discrete):
        return F.one_hot(
            torch.as_tensor(samples, dtype=torch.long, device=device),
            num_classes=space.n,
        ).to(torch.float32)
    elif isinstance(space, gym.spaces.Box):
        return squash(samples, dtype=torch.float32, device=device)
    elif isinstance(space, gym.spaces.Dict):
        return torch.cat(
            [
                encode_samples(samples[k], space[k], device=device)
                for k in sorted(space.spaces.keys())
            ],
            dim=1,
        )
    else:
        raise TypeError


def flat_size(space: gym.spaces.Space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Box):
        return np.prod(space.shape)
    elif isinstance(space, gym.spaces.Dict):
        return sum(flat_size(subspace) for subspace in space.spaces.values())
    else:
        raise NotImplementedError


def stack_dicts(obj: List[dict]):
    if len(obj) == 0:
        return {}
    ret = {}
    for k in obj[0].keys():
        if isinstance(obj[0][k], dict):
            ret[k] = stack_dicts([entry[k] for entry in obj])
        elif isinstance(obj[0][k], torch.Tensor):
            ret[k] = torch.stack([entry[k] for entry in obj], dim=0)
        else:
            ret[k] = np.stack([entry[k] for entry in obj], axis=0)
    return ret


def deep_idx(obj, idx, copy=False):
    # copy may be useful if memory needs to be freed
    if isinstance(obj, dict):
        return {k: deep_idx(v, idx, copy) for k, v in obj.items()}
    else:
        try:
            if copy:
                return obj[idx].copy()
            return obj[idx]
        except TypeError:
            warnings.warn(f"cant index object of type {type(obj)}: {obj}")
            return obj


class RunningMoments:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.s = 0

    def push(self, x):
        assert isinstance(x, float) or isinstance(x, int)
        self.n += 1
        if self.n == 1:
            self.m = x
        else:
            old_m = self.m
            self.m = old_m + (x - old_m) / self.n
            self.s = self.s + (x - old_m) * (x - self.m)

    def mean(self):
        return self.m

    def std(self):
        if self.n > 1:
            return math.sqrt(self.s / (self.n - 1))
        else:
            return self.m


class Logger:
    def __init__(self):
        self.buffer = defaultdict(RunningMoments)

        self.data = defaultdict(list)
        self.std_data = defaultdict(list)

        self.seen_plot_directories = set()

    # log metrics reported once per epoch
    def log(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in {**metrics, **kwargs}.items():
            if hasattr(v, "shape"):
                v = v.item()
            self.data[k].append(v)

    # push metrics logged many times per epoch, to aggregate later
    def push(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in {**metrics, **kwargs}.items():
            if hasattr(v, "shape"):
                v = v.item()
            self.buffer[k].push(v)

    # computes mean and std of metrics pushed many times per epoch
    def step(self):
        for k, v in self.buffer.items():
            self.data[k].append(v.mean())
            self.std_data[k].append(v.std())
        self.buffer.clear()

    def save(self, filename):
        if not filename.endswith(".pickle"):
            filename = filename + ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def generate_plots(self, dirname="plotgen"):
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns

        matplotlib.use("Agg")
        sns.set_theme()

        if dirname not in self.seen_plot_directories:
            self.seen_plot_directories.add(dirname)
            os.makedirs(dirname, exist_ok=True)

            for filename in os.listdir(dirname):
                file_path = os.path.join(dirname, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)

        for name, values in self.data.items():
            fig, ax = plt.subplots()
            fig: plt.Figure
            ax: plt.Axes

            x = np.arange(len(self.data[name]))
            values = np.array(values)

            (line,) = ax.plot(x, values)
            if name in self.std_data:
                stds = np.array(self.std_data[name])
                ax.fill_between(
                    x,
                    values - stds,
                    values + stds,
                    color=line.get_color(),
                    alpha=0.3,
                )

            if len(values) <= 100:  # add thick circles for clarity
                ax.scatter(x, values, color=line.get_color())

            ax.set_title(name.replace("_", " "))
            ax.set_xlabel("epochs")

            fig.savefig(os.path.join(dirname, name))
            plt.close(fig)


def get_agent_class(agent_id):
    if not re.fullmatch(r"[a-z]+\d+", agent_id):
        raise ValueError(f"agent_id {agent_id} is in unexpected format")
    return "".join(c for c in agent_id if not c.isdigit())


def shuffled_within_classes(agent_ids, policy_id_map):
    all_policy_ids = set(policy_id_map[agent_id] for agent_id in agent_ids)
    ret = [None for _ in range(len(agent_ids))]
    for policy_id in all_policy_ids:
        idxs = [
            i for i in range(len(agent_ids)) if policy_id_map[agent_ids[i]] == policy_id
        ]
        shuffled = idxs.copy()
        random.shuffle(shuffled)

        for i, idx in enumerate(idxs):
            ret[idx] = agent_ids[shuffled[i]]
    return ret


def iqm(scores):
    return trim_mean(scores, proportiontocut=0.25, axis=None)


def grad_norm(module):
    with torch.no_grad():
        if components := [
            torch.norm(p.grad.detach(), 2.0)
            for p in module.parameters()
            if p.grad is not None
        ]:
            return torch.norm(
                torch.stack(components),
                2.0,
            ).item()
        return 0
