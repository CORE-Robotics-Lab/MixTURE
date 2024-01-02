import torch

from ablations.simultaneous import CombinedTrainer, ExposedBCTrainer, ExposedPPOTrainer
from agents import RecurrentAttentionCommAgents
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
    assert config["method"] in ("none", "offline", "online")

    match config["env_name"]:
        case "predatorprey_5x5":
            env = PredatorPrey5x5()
            fc_dim = 64
        case "predatorcapture_5x5":
            env = PredatorCapture5x5()
            fc_dim = 64
        case "firecommander_5x5":
            env = FireCommander5x5()
            fc_dim = 64
        case "predatorprey_10x10":
            env = PredatorPrey10x10()
            fc_dim = 256
        case "predatorcapture_10x10":
            env = PredatorCapture10x10()
            fc_dim = 256
        case "firecommander_10x10":
            env = FireCommander10x10(n_fires=1)
            fc_dim = 256
        case _:
            raise ValueError(f"unexpected env: {config['env_name']}")

    agents = RecurrentAttentionCommAgents(config["lr"], config["lr"], fc_dim=fc_dim)
    demo_filename = f"../../demos/{config['env_name']}.pickle"

    if config["method"] == "online":
        trainer = CombinedTrainer(
            config["lr"],
            bc_trainer=ExposedBCTrainer(
                env,
                agents,
                demo_filename,
                minibatch_size=32,
            ),
            ppo_trainer=ExposedPPOTrainer(
                env,
                agents,
                MixedGAILReward(demo_filename, 1e-5).normalized(),
                gae_lambda=0.5,
                minibatch_size=32,
            ),
            bc_weight=config["bc_weight"],
        )
    else:
        assert config["method"] in ("none", "offline")
        trainer = PPOTrainer(
            env,
            agents,
            MixedGAILReward(demo_filename, 1e-5).normalized(),
            gae_lambda=0.5,
            minibatch_size=32,
        )
        if config["method"] == "offline":
            agents.load_state_dict(
                torch.load(
                    f"../../baselines/pretrained_policies/"
                    f"continuous_bc_{config['env_name']}_best.pth"
                )
            )

    trainer.evaluate()
    for i in range(200):
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
            "bc_weight": "nuisance",
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
        ):
            for method in ("none", "offline", "online"):
                for lr in (10**-3,):
                    if method == "online":
                        for bc_weight in (0.1, 1):
                            tuner.add(
                                {
                                    "env_name": env_name,
                                    "method": method,
                                    "lr": lr,
                                    "bc_weight": bc_weight,
                                    "trial_idx": trial_idx,
                                }
                            )
                    else:
                        tuner.add(
                            {
                                "env_name": env_name,
                                "method": method,
                                "lr": lr,
                                "bc_weight": 0,
                                "trial_idx": trial_idx,
                            }
                        )

    tuner.run()


if __name__ == "__main__":
    main()
