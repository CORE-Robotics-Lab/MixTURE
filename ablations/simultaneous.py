import math

import numpy as np
import torch
import torch.nn.functional as F

from agents import RecurrentAttentionCommAgents
from behavioral_cloning import BCTrainer
from envs import FireCommander
from ez_tuning import Tuner
from marl import PPOTrainer
from reward_signals import MixedGAILReward
from utils import Logger, deep_idx


# like a BCTrainer, but it yields loss and leaves the optimizer step up to you
class ExposedBCTrainer(BCTrainer):
    def run(self):
        curr = self.trajectory_starts.copy()
        while (mask := np.less(curr, self.trajectory_ends)).sum() > self.minibatch_size:
            traj_idxs = np.flatnonzero(mask)
            selected_traj_idx = np.random.choice(
                traj_idxs, size=self.minibatch_size, replace=False
            )
            yield from self.train_step(curr, selected_traj_idx)
        self.logger.step()

    def train_step(self, curr, selected_traj_idx):
        new_curr = np.minimum(
            curr[selected_traj_idx] + self.segment_length,
            self.trajectory_ends[selected_traj_idx],
        )

        batch_idx = np.concatenate(
            [
                np.arange(curr[selected_traj_idx][i], new_curr[i])
                for i in range(len(new_curr))
            ]
        )
        logps = self.agents.train_forward(self.rollout, batch_idx)

        loss = torch.tensor(0.0)
        for agent_id in logps.keys():
            loss += -(logps[agent_id].mean())
        yield loss

        if hasattr(self.agents, "hidden_states"):
            seg_idxs = []
            for idx in curr[selected_traj_idx]:
                ii = np.searchsorted(self.rollout["segment_starts"], idx)
                assert ii < len(self.rollout["segment_starts"])
                assert self.rollout["segment_starts"][ii] == idx
                seg_idxs.append(ii)
            seg_idxs = np.array(seg_idxs)
            assert len(seg_idxs) == len(selected_traj_idx)

            has_next_mask = new_curr < self.trajectory_ends[selected_traj_idx]
            for agent_id in logps.keys():
                self.rollout["initial_hidden"][agent_id][
                    :, seg_idxs[has_next_mask] + 1
                ] = (
                    self.agents.hidden_states[agent_id][:, has_next_mask]
                    .detach()
                    .clone()
                )

        curr[selected_traj_idx] = new_curr


# like a PPOTrainer, but it yields loss and leaves the optimizer step up to you
class ExposedPPOTrainer(PPOTrainer):
    def run(self):
        rollout = self.collect_rollout()

        self.reward_signal.update(rollout)

        all_segments = np.arange(rollout["segment_starts"].shape[0])

        for epoch in range(self.num_epochs):
            np.random.shuffle(all_segments)

            if self.recompute_advantages:
                self.compute_advantages_and_returns(rollout)

            for i in range(math.floor(all_segments.shape[0] / self.minibatch_size)):
                idx = np.concatenate(
                    [
                        np.arange(
                            rollout["segment_starts"][seg],
                            rollout["segment_ends"][seg],
                        )
                        for seg in all_segments[
                            i * self.minibatch_size : (i + 1) * self.minibatch_size
                        ]
                    ]
                )
                yield self.train_loss(rollout, idx)

        self.iteration += 1
        self.logger.step()

    def train_loss(self, rollout, idx):
        action_logps = self.agents.train_forward(rollout, idx)

        tot_loss = torch.tensor(0.0, device=self.device)
        for agent_id in self.agent_ids:
            adv = rollout["advantages"][agent_id][idx]
            advantages = torch.as_tensor(
                (adv - adv.mean()) / (adv.std() + 1e-8),
                device=self.device,
                dtype=torch.float32,
            )

            action_logps_old = torch.as_tensor(
                rollout["action_logp"][agent_id][idx],
                device=self.device,
                dtype=torch.float32,
            )
            ratio = torch.exp(action_logps[agent_id] - action_logps_old)

            with torch.no_grad():
                clip_fraction = torch.mean(
                    torch.gt(torch.abs(ratio - 1), self.ppo_clip_coeff).float()
                ).item()
                self.logger.push({f"ppo_clip_fraction_{agent_id}": clip_fraction})

            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(
                ratio, 1 - self.ppo_clip_coeff, 1 + self.ppo_clip_coeff
            )

            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            self.logger.push({f"policy_loss_{agent_id}": policy_loss.item()})

            tot_loss += policy_loss

        values = self.agents.predict_values(
            deep_idx(rollout["obs"], idx),
            deep_idx(rollout["actions"], idx),
        )

        for agent_id in self.agent_ids:
            value_loss = F.mse_loss(
                values[agent_id],
                torch.as_tensor(
                    rollout["value_targets"][agent_id][idx],
                    device=self.device,
                    dtype=torch.float32,
                ),
            )

            self.logger.push({f"critic_loss_{agent_id}": value_loss.item()})

            tot_loss += value_loss

        if custom_loss := self.agents.custom_loss(
            loss=tot_loss,
            rollout=rollout,
            batch_idx=idx,
            iteration=self.iteration,
        ):
            tot_loss = custom_loss
        return tot_loss


class CombinedTrainer:
    def __init__(
        self,
        lr,
        bc_trainer: ExposedBCTrainer,
        ppo_trainer: ExposedPPOTrainer,
        bc_weight=1.0,
    ):
        ppo_trainer.agents = bc_trainer.agents
        ppo_trainer.logger = bc_trainer.logger
        self.logger: Logger = ppo_trainer.logger

        self.optim = torch.optim.Adam(ppo_trainer.agents.parameters(), lr=lr)
        self.bc_trainer = bc_trainer
        self.ppo_trainer = ppo_trainer
        self.bc_weight = bc_weight

        self.curr_bc_iter = iter(bc_trainer.run())

    def run(self):
        for ppo_loss in self.ppo_trainer.run():
            self.optim.zero_grad(set_to_none=True)

            try:
                bc_loss = next(self.curr_bc_iter)
            except StopIteration:
                self.curr_bc_iter = iter(self.bc_trainer.run())
                bc_loss = next(self.curr_bc_iter)

            (ppo_loss + self.bc_weight * bc_loss).backward()

            self.optim.step()

    def evaluate(self):
        self.ppo_trainer.evaluate()


def trial(config):
    env = FireCommander(20, 6, 4, vision=2, max_timesteps=80)

    bc_trainer = ExposedBCTrainer(
        env,
        RecurrentAttentionCommAgents(0, 0, fc_dim=256),
        "../demos/firecommander_20x20.pickle",
        minibatch_size=32,
    )
    ppo_trainer = ExposedPPOTrainer(
        env,
        RecurrentAttentionCommAgents(0, 0, fc_dim=256),
        reward_signal=MixedGAILReward(
            "../demos/firecommander_20x20.pickle", 1e-5
        ).normalized(),
        gae_lambda=0.5,
        minibatch_size=32,
    )
    trainer = CombinedTrainer(
        config["lr"], bc_trainer, ppo_trainer, bc_weight=config["bc_weight"]
    )

    trainer.evaluate()
    for i in range(100):
        trainer.run()
        if i % 10 == 9:
            trainer.evaluate()
            yield ppo_trainer.logger


def main():
    tuner = Tuner(
        {"lr": "nuisance", "bc_weight": "science", "trial_idx": "id"},
        trial,
        metric="episode_len",
        mode="min",
    )
    for lr in (10**-3.5, 10**-3):
        for bc_weight in (10**-1, 10**-0.5, 1, 10**0.5):
            for trial_idx in range(1):
                tuner.add({"lr": lr, "bc_weight": bc_weight, "trial_idx": trial_idx})
    tuner.run()


if __name__ == "__main__":
    main()
