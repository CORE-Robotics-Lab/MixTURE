import os
import random

import numpy as np
import torch

from ablations.simultaneous import CombinedTrainer, ExposedBCTrainer, ExposedPPOTrainer
from agents import FullAttentionMIMAgents, RecurrentAgents
from envs.comm_wrapper import add_uniform_comms
from envs.simple_fire_commander import SimpleFireCommander
from ez_tuning import Tuner
from reward_signals import ExplicitReward, FullyLocalGAILReward, MixedGAILReward


def trial(config):
    random.seed(config["trial_idx"])
    np.random.seed(random.randint(0, 2**32))
    torch.random.manual_seed(random.randint(0, 2**32))

    if config["difficulty"] == "easy":
        map_size = 8
        n_perception = 3
        n_action = 2
        n_fires = 1
        vision = 1

        demo_difficulty = "easy"
    elif config["difficulty"].startswith("medium"):
        assert any(config["difficulty"] == f"medium_{i}" for i in range(1, 6))

        map_size = 10
        n_perception = 2
        n_action = 2
        n_fires = int(config["difficulty"].split("_")[-1])
        vision = 1

        demo_difficulty = "moderate"
    else:
        assert any(config["difficulty"] == f"hard_{i}" for i in range(1, 11))

        map_size = 20
        n_perception = 4
        n_action = 5
        n_fires = int(config["difficulty"].split("_")[-1])
        vision = 2

        demo_difficulty = "hard"

    env = SimpleFireCommander(
        map_size=map_size,
        n_perception=n_perception,
        n_action=n_action,
        n_fires=n_fires,
        vision=vision,
        fire_propagation_period=12,
        max_timesteps=80,
    )

    with_comm_filename = (
        f"../demos/user_study/"
        f"augmented_{demo_difficulty}_with_comm_min_rating=0.pickle"
    )
    no_comm_filename = (
        f"../demos/user_study/"
        f"augmented_{demo_difficulty}_no_comm_min_rating=0.pickle"
    )

    if config["method"] == "RL":
        agents = RecurrentAgents(0, 0, fc_dim=256)
        demo_filename = with_comm_filename
        env = add_uniform_comms(env, 26, True)
        reward = ExplicitReward()
    elif config["method"] == "no-comm LFD":
        agents = RecurrentAgents(0, 0, fc_dim=256)
        demo_filename = no_comm_filename
        reward = FullyLocalGAILReward(demo_filename, lr=1e-5)
    elif config["method"] == "with-comm LFD":
        agents = RecurrentAgents(0, 0, fc_dim=256)
        demo_filename = with_comm_filename
        env = add_uniform_comms(env, 26, True)
        reward = FullyLocalGAILReward(demo_filename, lr=1e-5)
    else:
        assert config["method"] == "MixTURE"
        agents = FullAttentionMIMAgents(0, 0, mim_coeff=0.01, fc_dim=256)
        demo_filename = no_comm_filename
        reward = MixedGAILReward(demo_filename, lr=1e-5)

    trainer = CombinedTrainer(
        config["lr"],
        bc_trainer=ExposedBCTrainer(env, agents, demo_filename, minibatch_size=32),
        ppo_trainer=ExposedPPOTrainer(
            env, agents, reward.normalized(), gae_lambda=0.5, minibatch_size=32
        ),
        bc_weight=config["bc_weight"],
    )
    trainer.optim = torch.optim.Adam(
        trainer.ppo_trainer.agents.parameters(), lr=config["lr"], weight_decay=1e-5
    )

    os.makedirs(f"saved_agents/{config['difficulty']}", exist_ok=True)

    best_score = float("inf")
    trainer.evaluate()
    for i in range(250):
        trainer.run()
        if i % 10 == 9:
            trainer.evaluate()
            score = trainer.logger.data["episode_len"][-1]
            if score < best_score:
                best_score = score
                torch.save(
                    trainer.ppo_trainer.agents.state_dict(),
                    f"saved_agents/{config['difficulty']}/"
                    f"{config['method']}_best_{config['trial_idx']}.pth",
                )
            yield trainer.logger


def main():
    tuner = Tuner(
        {
            "difficulty": "task",
            "method": "science",
            "lr": "nuisance",
            "bc_weight": "nuisance",
            "trial_idx": "id",
        },
        trial,
        metric="episode_len",
        mode="min",
        plot_dir="other_baselines_plots",
        ckpt_filename="other_baselines.ckpt",
    )
    for trial_idx in range(3):
        for difficulty in ("easy", "medium_1", "hard_1"):
            for method in ("no-comm LFD", "RL"):
                tuner.add(
                    {
                        "difficulty": difficulty,
                        "method": method,
                        "lr": 1e-3,
                        "bc_weight": 0.1,
                        "trial_idx": trial_idx,
                    }
                )
    tuner.run()


if __name__ == "__main__":
    main()
