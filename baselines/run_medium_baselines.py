import os
import random

import numpy as np
import torch

from ablations.simultaneous import CombinedTrainer, ExposedBCTrainer, ExposedPPOTrainer
from agents import FullAttentionMIMAgents, RecurrentAgents
from envs import FireCommander, PredatorCapture, PredatorPrey
from envs.comm_wrapper import add_uniform_comms
from ez_tuning import Tuner
from reward_signals import ExplicitReward, FullyLocalGAILReward, MixedGAILReward


def trial(config):
    random.seed(config["trial_idx"])
    np.random.seed(random.randint(0, 2**32))
    torch.random.manual_seed(random.randint(0, 2**32))

    envs = {
        "predatorprey": (
            "Predator-Prey",
            PredatorPrey(10, 6, vision=1, max_timesteps=80),
        ),
        "predatorcapture": (
            "Predator-Capture-Prey",
            PredatorCapture(10, 3, 3, vision=1, max_timesteps=80),
        ),
        "firecommander": (
            "FireCommander",
            FireCommander(10, 3, 3, vision=1, max_timesteps=80),
        ),
    }

    lr = config["lr"]
    env_name, env = envs[config["env"]]
    comm_dim = 26

    if config["method"] == "RL":
        agents = RecurrentAgents(lr, lr, fc_dim=256)
        demo_filename = f"../demos/{config['env']}_10x10_comm.pickle"
        env = add_uniform_comms(env, comm_dim)
        reward = ExplicitReward()
    elif config["method"] == "no-comm LFD":
        agents = RecurrentAgents(lr, lr, fc_dim=256)
        demo_filename = f"../demos/{config['env']}_10x10.pickle"
        reward = FullyLocalGAILReward(
            demo_filename,
            lr=config["discrim_lr"],
        )
    elif config["method"] == "comm LFD":
        agents = RecurrentAgents(lr, lr, fc_dim=256)
        demo_filename = f"../demos/{config['env']}_10x10_comm.pickle"
        env = add_uniform_comms(env, comm_dim)
        reward = FullyLocalGAILReward(
            demo_filename,
            lr=config["discrim_lr"],
        )
    else:
        assert config["method"] == "ours"
        agents = FullAttentionMIMAgents(lr, lr, mim_coeff=0.01, fc_dim=256)
        demo_filename = f"../demos/{config['env']}_10x10.pickle"
        reward = MixedGAILReward(
            demo_filename,
            lr=config["discrim_lr"],
        )

    trainer = CombinedTrainer(
        lr=lr,
        bc_trainer=ExposedBCTrainer(env, agents, demo_filename, minibatch_size=32),
        ppo_trainer=ExposedPPOTrainer(
            env, agents, reward.normalized(), gae_lambda=0.5, minibatch_size=32
        ),
        bc_weight=config["bc_weight"],
    )

    os.makedirs(f"saved_agents/{config['env']}_10x10/", exist_ok=True)

    best_score = float("inf")
    trainer.evaluate()
    for i in range(200):
        trainer.run()
        if i % 10 == 9:
            trainer.evaluate()
            score = trainer.logger.data["episode_len"][-1]
            if score < best_score:
                best_score = score
                torch.save(
                    trainer.ppo_trainer.agents.state_dict(),
                    f"saved_agents/{config['env']}_10x10/"
                    f"{config['method']}_best_{config['trial_idx']}.pth",
                )
            yield trainer.logger


BEST_LRS = {
    ("predatorprey", "RL"): 10**-3,
    ("predatorprey", "no-comm LFD"): 10**-3.5,
    ("predatorprey", "comm LFD"): 10**-3,
    ("predatorprey", "ours"): 10**-3,
    ("predatorcapture", "RL"): 10**-3,
    ("predatorcapture", "no-comm LFD"): 10**-3,
    ("predatorcapture", "comm LFD"): 10**-3,
    ("predatorcapture", "ours"): 10**-3.5,
    ("firecommander", "RL"): 10**-3,
    ("firecommander", "no-comm LFD"): 10**-3,
    ("firecommander", "comm LFD"): 10**-3,
    ("firecommander", "ours"): 10**-3,
}
BEST_BC_WEIGHTS = {
    ("predatorprey", "RL"): 10**-0.5,
    ("predatorprey", "no-comm LFD"): 10**-1,
    ("predatorprey", "comm LFD"): 10**-0.5,
    ("predatorprey", "ours"): 10**-0.5,
    ("predatorcapture", "RL"): 10**-1,
    ("predatorcapture", "no-comm LFD"): 10**-1,
    ("predatorcapture", "comm LFD"): 10**-1,
    ("predatorcapture", "ours"): 10**-0.5,
    ("firecommander", "RL"): 10**-1,
    ("firecommander", "no-comm LFD"): 10**-1,
    ("firecommander", "comm LFD"): 10**-1,
    ("firecommander", "ours"): 10**-0.5,
}


def main():
    # since environments are harder (5x5 -> 10x10, 3 agents -> 6 agents), we use:
    #   - larger fc_dim (64 -> 256)
    #   - larger minibatch size (8 segments / 64 steps -> 32 segments / 256 steps)
    #   - auxiliary BC loss (during online training, using offline data :)
    #   - our differentiable comm is now using the attention formulation

    tuner = Tuner(
        spec={
            "env": "task",
            "method": "science",
            "bc_weight": "nuisance",
            "lr": "nuisance",
            "discrim_lr": "nuisance",
            "trial_idx": "id",
        },
        trial_fn=trial,
        metric="episode_len",
        mode="min",
        plot_dir="medium_plots_again",
        ckpt_filename="medium_again.ckpt",
    )

    # Hyperparameter sweep
    # for env in ("predatorprey", "predatorcapture", "firecommander"):
    #     for lr in (10**-4, 10**-3.5, 10**-3):
    #         for bc_weight in (10**-1.5, 10**-1, 10**-0.5):
    #             for method in ("RL", "no-comm LFD", "comm LFD", "ours"):
    #
    #                 tuner.add(
    #                     {
    #                         "env": env,
    #                         "method": method,
    #                         "lr": lr,
    #                         "discrim_lr": 1e-5,
    #                         "bc_weight": bc_weight,
    #                         "trial_idx": 0,
    #                     },
    #                 )

    for env in ("predatorprey", "predatorcapture", "firecommander"):
        for trial_idx in range(1):
            for method in ("RL", "no-comm LFD", "comm LFD", "ours"):
                lr = BEST_LRS[(env, method)]
                bc_weight = BEST_BC_WEIGHTS[(env, method)]

                tuner.add(
                    {
                        "env": env,
                        "method": method,
                        "lr": lr,
                        "discrim_lr": 1e-5,
                        "bc_weight": bc_weight,
                        "trial_idx": trial_idx,
                    },
                )

    tuner.run()


if __name__ == "__main__":
    main()
