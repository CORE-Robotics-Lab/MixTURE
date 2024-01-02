import os
import random

import numpy as np
import torch

from agents import RecurrentAgents, RecurrentMIMAgents2
from envs import FireCommander, PredatorCapture, PredatorPrey
from envs.comm_wrapper import add_uniform_comms
from ez_tuning import Tuner
from marl import PPOTrainer
from reward_signals import ExplicitReward, FullyLocalGAILReward, MixedGAILReward


def trial(config):
    random.seed(config["trial_idx"])
    np.random.seed(random.randint(0, 2**32))
    torch.random.manual_seed(random.randint(0, 2**32))

    envs = {
        "predatorprey": (
            "Predator-Prey",
            PredatorPrey(5, 3, vision=0, max_timesteps=20),
        ),
        "predatorcapture": (
            "Predator-Capture-Prey",
            PredatorCapture(5, 2, 1, vision=0, max_timesteps=40),
        ),
        "firecommander": (
            "FireCommander",
            FireCommander(5, 2, 1, vision=1, max_timesteps=80),
        ),
    }

    lr = config["lr"]
    env_name, env = envs[config["env"]]
    comm_dim = 26

    if config["method"] == "RL":
        agents = RecurrentAgents(lr, lr, fc_dim=64)
        env = add_uniform_comms(env, comm_dim)
        reward = ExplicitReward()
    elif config["method"] == "no-comm LFD":
        agents = RecurrentAgents(lr, lr, fc_dim=64)
        reward = FullyLocalGAILReward(
            f"../demos/{config['env']}_5x5.pickle",
            lr=config["discrim_lr"],
        )
    elif config["method"] == "comm LFD":
        agents = RecurrentAgents(lr, lr, fc_dim=64)
        env = add_uniform_comms(env, comm_dim)
        reward = FullyLocalGAILReward(
            f"../demos/{config['env']}_5x5_comm.pickle",
            lr=config["discrim_lr"],
        )
    else:
        assert config["method"] == "ours"
        agents = RecurrentMIMAgents2(lr, lr, mim_coeff=0.1, fc_dim=64)
        reward = MixedGAILReward(
            f"../demos/{config['env']}_5x5.pickle", lr=config["discrim_lr"]
        )

    trainer = PPOTrainer(env, agents, reward.normalized(), gae_lambda=0.5)

    os.makedirs(f"saved_agents/{config['env']}_5x5/", exist_ok=True)

    best_score = float("inf")
    trainer.evaluate()
    for i in range(500):
        trainer.run()
        if i % 10 == 9:
            trainer.evaluate()
            score = trainer.logger.data["episode_len"][-1]
            if score < best_score:
                best_score = score
                torch.save(
                    trainer.agents.state_dict(),
                    f"saved_agents/{config['env']}_5x5/"
                    f"{config['method']}_best_{config['trial_idx']}.pth",
                )
            yield trainer.logger


# see comment below for reproducing the hyperparameter sweep
BEST_LRS = {
    ("predatorprey", "RL"): 10**-3.5,
    ("predatorprey", "no-comm LFD"): 10**-4,
    ("predatorprey", "comm LFD"): 10**-3,
    ("predatorprey", "ours"): 10**-3,
    ("predatorcapture", "RL"): 10**-3.5,
    ("predatorcapture", "no-comm LFD"): 10**-4,
    ("predatorcapture", "comm LFD"): 10**-3,
    ("predatorcapture", "ours"): 10**-3,
    ("firecommander", "RL"): 10**-4,
    ("firecommander", "no-comm LFD"): 10**-4,
    ("firecommander", "comm LFD"): 10**-3,
    ("firecommander", "ours"): 10**-3,
}


def main():
    tuner = Tuner(
        spec={
            "env": "task",
            "method": "science",
            "lr": "nuisance",
            "discrim_lr": "nuisance",
            "trial_idx": "id",
        },
        trial_fn=trial,
        metric="episode_len",
        mode="min",
        plot_dir="easy_plots_again",
        ckpt_filename="easy_again.ckpt",
    )

    for env in ("predatorprey", "predatorcapture", "firecommander"):
        for trial_idx in range(1):
            # uncomment for reproducing hyperparameter sweep
            # for lr in range(10**-4, 10**-3.5, 10**-3):
            # for discrim_lr in range(10**-5.5, 10**-5, 10**-4.5):
            for method in ("RL", "no-comm LFD", "comm LFD", "ours"):
                lr = BEST_LRS[(env, method)]

                tuner.add(
                    {
                        "env": env,
                        "method": method,
                        "lr": lr,
                        "discrim_lr": 1e-5,
                        "trial_idx": trial_idx,
                    },
                )
    tuner.run()


if __name__ == "__main__":
    main()
