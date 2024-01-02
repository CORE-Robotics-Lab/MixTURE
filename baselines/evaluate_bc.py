import torch

from agents import RecurrentAttentionCommAgents
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
from marl import PPOTrainer
from reward_signals import ExplicitReward

ENVS = {
    (5, "predatorprey"): PredatorPrey5x5(),
    (10, "predatorprey"): PredatorPrey10x10(),
    (20, "predatorprey"): PredatorPrey20x20(),
    (5, "predatorcapture"): PredatorCapture5x5(),
    (10, "predatorcapture"): PredatorCapture10x10(),
    (20, "predatorcapture"): PredatorCapture20x20(),
    (5, "firecommander"): FireCommander5x5(),
    (10, "firecommander"): FireCommander10x10(n_fires=1),
    (20, "firecommander"): FireCommander20x20(n_fires=1),
}


for size in (5, 10, 20):
    for task in ("predatorprey", "predatorcapture", "firecommander"):
        agents = RecurrentAttentionCommAgents(0, 0, fc_dim=256 if size >= 10 else 64)
        trainer = PPOTrainer(
            ENVS[size, task],
            agents=agents,
            reward_signal=ExplicitReward(),
        )
        agents.load_state_dict(
            torch.load(
                f"pretrained_policies/continuous_bc_{task}_{size}x{size}_best.pth"
            )
        )
        trainer.evaluate()
        print(f"({size}, {task}): {trainer.logger.data['episode_len']},")
