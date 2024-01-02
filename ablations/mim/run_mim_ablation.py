import random

import numpy as np
import torch

from agents import (
    FullAttentionMIMAgents,
    RecurrentAttentionCommAgents,
    RecurrentCommChannelAgents,
    RecurrentMIMAgents2,
)
from envs import (
    FireCommander5x5,
    FireCommander10x10,
    PredatorCapture5x5,
    PredatorCapture10x10,
    PredatorPrey5x5,
    PredatorPrey10x10,
)
from ez_tuning import Tuner
from marl import PPOTrainer
from reward_signals import MixedGAILReward


def trial(config):
    random.seed(config["trial_idx"])
    np.random.seed(random.randint(0, 2**32))
    torch.random.manual_seed(random.randint(0, 2**32))

    match config["env_name"]:
        case "predatorprey_5x5":
            env = PredatorPrey5x5()
        case "predatorcapture_5x5":
            env = PredatorCapture5x5()
        case "firecommander_5x5":
            env = FireCommander5x5()
        case "predatorprey_10x10":
            env = PredatorPrey10x10()
        case "predatorcapture_10x10":
            env = PredatorCapture10x10()
        case "firecommander_10x10":
            env = FireCommander10x10(n_fires=1)
        case _:
            raise ValueError(f"unexpected env: {config['env_name']}")

    demo_filename = f"../../demos/{config['env_name']}.pickle"

    if config["env_name"].endswith("5x5"):
        if config["use_mim"]:
            agents = RecurrentMIMAgents2(
                config["lr"], config["lr"], fc_dim=64, mim_coeff=config["mim_coeff"]
            )
        else:
            agents = RecurrentCommChannelAgents(config["lr"], config["lr"], fc_dim=64)
    else:
        if config["use_mim"]:
            agents = FullAttentionMIMAgents(
                config["lr"], config["lr"], fc_dim=256, mim_coeff=config["mim_coeff"]
            )
        else:
            agents = RecurrentAttentionCommAgents(
                config["lr"], config["lr"], fc_dim=256
            )

    trainer = PPOTrainer(
        env,
        agents,
        MixedGAILReward(demo_filename, 1e-5).normalized(),
        gae_lambda=0.5,
        minibatch_size=32,
    )

    trainer.evaluate()
    for i in range(500):
        trainer.run()
        if i % 10 == 9:
            trainer.evaluate()
            yield trainer.logger


def main():
    tuner = Tuner(
        {
            "env_name": "task",
            "use_mim": "science",
            "mim_coeff": "nuisance",
            "lr": "nuisance",
            "trial_idx": "id",
        },
        trial,
        metric="episode_len",
        mode="min",
    )
    for trial_idx in range(3):
        for env_name in (
            "predatorprey_5x5",
            "predatorcapture_5x5",
            "firecommander_5x5",
            "predatorprey_10x10",
            "predatorcapture_10x10",
            "firecommander_10x10",
            # "predatorprey_20x20",
            # "predatorcapture_20x20",
            # "firecommander_20x20",
        ):
            tuner.add(
                {
                    "env_name": env_name,
                    "use_mim": False,
                    "mim_coeff": 0,
                    "lr": 1e-3,
                    "trial_idx": trial_idx,
                }
            )
            for mim_coeff in (0.1, 0.01):
                tuner.add(
                    {
                        "env_name": env_name,
                        "use_mim": True,
                        "mim_coeff": mim_coeff,
                        "lr": 1e-3,
                        "trial_idx": trial_idx,
                    }
                )

    tuner.run()


if __name__ == "__main__":
    main()
