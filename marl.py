import math

import numpy as np
import torch
import torch.nn.functional as F

from agents import AgentGroup
from envs.base_multi_agent import MultiAgentEnv
from reward_signals import RewardSignal
from utils import Logger, batchify, deep_idx, encode_samples, stack_dicts


class PPOTrainer:
    def __init__(
        self,
        env: MultiAgentEnv,
        agents: AgentGroup,
        reward_signal: RewardSignal,
        gamma=0.99,
        gae_lambda=0.9,
        rollout_steps=4096,
        segment_length=8,
        num_epochs=3,
        minibatch_size=8,
        ppo_clip_coeff=0.2,
        recompute_advantages=True,
        gradient_clipping=5.0,
        device=torch.device("cpu"),
    ):
        self.logger: Logger = Logger()

        self.env = env
        self.env_obs = self.env.reset()
        self.env_terminal = False
        self.env.logger = self.logger

        self.agent_ids = self.env.observation_spaces.keys()
        self.agents: AgentGroup = agents.configured_with(
            self.env.observation_spaces, self.env.action_spaces, self.logger, device
        )
        self.agents.reset()
        self.reward_signal = reward_signal.configured_with(
            self.env.observation_spaces, self.env.action_spaces, self.logger, device
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_steps = rollout_steps
        self.segment_length = segment_length
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.ppo_clip_coeff = ppo_clip_coeff
        self.recompute_advantages = recompute_advantages
        self.gradient_clipping = gradient_clipping

        self.device = device

        self.iteration = 0

    def collect_rollout(self) -> dict:
        self.agents.to(torch.device("cpu"))

        rollout_entries = []
        num_steps = 0
        num_episodes = 0

        segment_starts = []
        hidden_states = []

        while num_steps < self.rollout_steps or not self.env_terminal:
            num_steps += 1
            if self.env_terminal:
                self.env_obs = self.env.reset()
                self.agents.reset()
                self.env_terminal = False
                segment_starts.append(num_steps - 1)
                num_episodes += 1
            elif num_steps == 1 or num_steps > segment_starts[-1] + self.segment_length:
                segment_starts.append(num_steps - 1)

            with torch.no_grad():
                if (
                    hasattr(self.agents, "hidden_states")
                    and num_steps - 1 == segment_starts[-1]
                ):
                    hidden_states.append(self.agents.hidden_states.copy())
                    for agent_id in self.agent_ids:
                        if hidden_states[-1][agent_id] is None:
                            hidden_states[-1][agent_id] = torch.zeros(
                                (1, 1, self.agents.fc_dim)
                            )
                actions, action_logp = self.agents.rollout_forward(
                    {
                        agent_id: encode_samples(
                            batchify(agent_obs),
                            self.env.observation_spaces[agent_id],
                        )
                        for agent_id, agent_obs in self.env_obs.items()
                    }
                )

            next_obs, env_rewards, self.env_terminal, _ = self.env.step(actions)

            rollout_entries.append(
                {
                    "obs": self.env_obs,
                    "actions": actions,
                    "action_logp": action_logp,
                    "env_rewards": env_rewards,
                    "terminal": self.env_terminal,
                }
            )
            self.env_obs = next_obs

        rollout = stack_dicts(rollout_entries)

        rollout["obs"] = {
            agent_id: encode_samples(
                agent_obs, self.env.observation_spaces[agent_id], device=self.device
            )
            for agent_id, agent_obs in rollout["obs"].items()
        }
        rollout["actions"] = {
            agent_id: encode_samples(
                agent_action, self.env.action_spaces[agent_id], device=self.device
            )
            for agent_id, agent_action in rollout["actions"].items()
        }

        rollout["num_steps"] = num_steps

        rollout["segment_starts"] = np.array(segment_starts)
        rollout["segment_ends"] = np.concatenate([segment_starts[1:], [num_steps]])

        if len(hidden_states) > 0:
            rollout["initial_hidden"] = {
                agent_id: torch.cat(
                    [hidden_states[i][agent_id] for i in range(len(hidden_states))],
                    dim=1,
                ).to(self.device)
                for agent_id in self.agent_ids
            }

        rollout["rewards"] = self.reward_signal.get_rewards(rollout)
        self.logger.log(
            {
                f"reward_signal_mean_{agent_id}": np.mean(rollout["rewards"][agent_id])
                for agent_id in self.agent_ids
            }
        )

        if not self.recompute_advantages:
            self.compute_advantages_and_returns(rollout)

        self.agents.to(self.device)

        self.logger.log(num_steps=num_steps)

        return rollout

    def compute_advantages_and_returns(self, rollout):
        with torch.no_grad():
            value_preds = {
                agent_id: value_estimate.cpu().numpy()
                for agent_id, value_estimate in self.agents.predict_values(
                    rollout["obs"], rollout["actions"]
                ).items()
            }

        advantages = {
            agent_id: np.zeros(rollout["num_steps"]) for agent_id in self.agent_ids
        }
        for agent_id in self.agent_ids:
            last_gae_lambda = 0
            for step in reversed(range(rollout["num_steps"])):
                if step == rollout["num_steps"] - 1:
                    next_value = 0
                else:
                    next_value = value_preds[agent_id][step + 1]

                non_terminal = 1 - rollout["terminal"][step]

                delta = (
                    rollout["rewards"][agent_id][step]
                    + self.gamma * next_value * non_terminal
                    - value_preds[agent_id][step]
                )
                last_gae_lambda = delta + (
                    self.gamma * self.gae_lambda * non_terminal * last_gae_lambda
                )
                advantages[agent_id][step] = last_gae_lambda

        rollout["advantages"] = advantages
        rollout["value_targets"] = {
            agent_id: advantages[agent_id] + value_preds[agent_id]
            for agent_id in self.agent_ids
        }

        for agent_id in self.agent_ids:
            symlog_advantage = np.sign(advantages[agent_id]) * np.log(
                1 + np.abs(advantages[agent_id])
            )
            self.logger.log(
                {
                    f"symlog_advantage_{agent_id}_mean": np.mean(symlog_advantage),
                    f"symlog_advantage_{agent_id}_std": np.std(symlog_advantage),
                }
            )

        self.logger.log(
            {
                f"explained_variance_{agent_id}": 1
                - np.var(rollout["value_targets"][agent_id] - value_preds[agent_id])
                / np.var(rollout["value_targets"][agent_id])
                for agent_id in self.agent_ids
            }
        )

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
                        self.logger.push(
                            {f"ppo_clip_fraction_{agent_id}": clip_fraction}
                        )

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

                for optim in self.agents.optims:
                    optim.zero_grad(set_to_none=True)
                tot_loss.backward()

                if self.gradient_clipping is not None:
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.agents.parameters(), self.gradient_clipping
                    )
                    self.logger.push(
                        grad_norm=norm.item(),
                        grad_clip_fraction=float(norm > self.gradient_clipping),
                    )

                for optim in self.agents.optims:
                    optim.step()

        self.iteration += 1
        self.logger.step()

    def evaluate(self, num_episodes=100):
        self.agents.to(torch.device("cpu"))

        wins = 0
        for _ in range(num_episodes):
            self.agents.reset()
            obs = self.env.reset()

            curr_episode_len = 0
            done = False
            while not done:
                curr_episode_len += 1
                with torch.no_grad():
                    actions, action_logps = self.agents.rollout_forward(
                        {
                            agent_id: encode_samples(
                                batchify(agent_obs),
                                space=self.env.observation_spaces[agent_id],
                            )
                            for agent_id, agent_obs in obs.items()
                        }
                    )

                for agent_id in action_logps.keys():
                    self.logger.push(
                        {
                            f"entropy_{agent_id}": -(
                                torch.exp(action_logps[agent_id])
                                * action_logps[agent_id]
                            )
                            .sum()
                            .item()
                        }
                    )
                obs, rewards, done, _ = self.env.step(actions)

            self.logger.push(episode_len=curr_episode_len)

            if (
                hasattr(self.env, "metrics")
                and "won" in self.env.metrics.keys()
                and self.env.metrics["won"]
            ):
                wins += 1
            if (
                hasattr(self.env, "metrics")
                and "fires_extinguished" in self.env.metrics
            ):
                self.logger.push(
                    fires_extinguished=self.env.metrics["fires_extinguished"]
                )

        self.agents.to(self.device)

        if hasattr(self.env, "metrics") and "won" in self.env.metrics:
            self.logger.log(winrate=wins / num_episodes)
        self.logger.step()


def main():
    import tqdm

    from agents import RecurrentCommChannelAgents
    from envs import PredatorPrey
    from reward_signals import MixedGAILReward

    trainer = PPOTrainer(
        PredatorPrey(10, n_predator=6, vision=1, max_timesteps=80),
        RecurrentCommChannelAgents(policy_lr=1e-3, critic_lr=1e-3),
        MixedGAILReward(
            demo_filename="demos/predatorprey_10x10.pickle",
            lr=1e-5,
        ).normalized(),
        gae_lambda=0.5,
    )

    for i in tqdm.trange(500):
        trainer.run()
        if i % 10 == 9:
            trainer.evaluate()
            trainer.logger.generate_plots()


if __name__ == "__main__":
    main()
