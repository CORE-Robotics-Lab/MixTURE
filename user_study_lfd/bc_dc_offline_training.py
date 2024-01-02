import ray
import torch

from agents import RecurrentAttentionCommAgents
from behavioral_cloning import BCTrainer
from envs import SimpleFireCommander
from ez_tuning import Tuner


def trial(config):
    difficulty = config["difficulty"]

    if difficulty == "easy":
        map_size = 8
        n_perception = 3
        n_action = 2
        n_fires = 1
        vision = 1

        demo_difficulty = "easy"
    elif difficulty.startswith("medium"):
        assert any(difficulty == f"medium_{i}" for i in range(1, 6))

        map_size = 10
        n_perception = 2
        n_action = 2
        n_fires = int(difficulty.split("_")[-1])
        vision = 1

        demo_difficulty = "moderate"
    else:
        assert any(difficulty == f"hard_{i}" for i in range(1, 11))

        map_size = 20
        n_perception = 4
        n_action = 5
        n_fires = int(difficulty.split("_")[-1])
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
    trainer = BCTrainer(
        env,
        RecurrentAttentionCommAgents(1e-3, 0, fc_dim=256),
        f"../demos/user_study/"
        f"augmented_{demo_difficulty}_no_comm_min_rating=0.pickle",
    )

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
                    trainer.agents.state_dict(), f"pretrained_{difficulty}_best.pth"
                )

            yield trainer.logger


def main():
    tuner = Tuner(
        {"difficulty": "task"},
        trial,
        metric="episode_len",
        mode="min",
        plot_dir="bc_pretraining_plots",
        ckpt_filename="bc_pretraining.ckpt",
        throw_on_exception=True,
    )
    tuner.add({"difficulty": "easy"})
    tuner.add({"difficulty": "medium_1"})
    tuner.add({"difficulty": "hard_1"})
    tuner.run()


if __name__ == "__main__":
    main()

