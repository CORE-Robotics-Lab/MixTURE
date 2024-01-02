import random

import numpy as np
import torch

from ablations.simultaneous import CombinedTrainer, ExposedBCTrainer, ExposedPPOTrainer
from agents import RecurrentAttentionCommAgents, RecurrentCommChannelAgents
from envs import (
    FireCommander5x5,
    FireCommander10x10,
    FireCommander20x20,
    PredatorCapture5x5,
    PredatorCapture10x10,
    PredatorCapture20x20,
    PredatorPrey5x5,
    PredatorPrey10x10,
    PredatorPrey20x20,
)
from ez_tuning import Tuner
from marl import PPOTrainer
from reward_signals import (
    FullyCentralizedGAILReward,
    FullyLocalGAILReward,
    MixedGAILReward,
)


def trial(config):
    assert config["method"] in ("local", "global", "mixed")
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
        # case "predatorprey_20x20":
        #     env = PredatorPrey20x20()
        # case "predatorcapture_20x20":
        #     env = PredatorCapture20x20()
        # case "firecommander_20x20":
        #     env = FireCommander20x20(n_fires=1)
        case _:
            raise ValueError(f"unexpected env: {config['env_name']}")

    demo_filename = f"../../demos/{config['env_name']}.pickle"

    match config["method"]:
        case "local":
            reward = FullyLocalGAILReward(demo_filename, lr=1e-5)
        case "global":
            reward = FullyCentralizedGAILReward(demo_filename, lr=1e-3)
        case "mixed":
            reward = MixedGAILReward(demo_filename, lr=1e-5)
        case _:
            raise ValueError(f"unexpected method: {config['method']}")

    if config["env_name"].endswith("5x5"):
        agents = RecurrentCommChannelAgents(config["lr"], config["lr"], fc_dim=64)
    else:
        agents = RecurrentAttentionCommAgents(config["lr"], config["lr"], fc_dim=256)
    trainer = PPOTrainer(
        env,
        agents,
        reward.normalized(),
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
            "method": "science",
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
            for method in ("local", "global", "mixed"):
                for lr in (10**-3,):
                    tuner.add(
                        {
                            "env_name": env_name,
                            "method": method,
                            "lr": lr,
                            "trial_idx": trial_idx,
                        }
                    )

    tuner.run()


if __name__ == "__main__":
    main()
