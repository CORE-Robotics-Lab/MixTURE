import pickle

import numpy as np
import torch

from agents import AgentGroup
from envs.base_multi_agent import MultiAgentEnv
from utils import PROJECT_DIR, Logger, batchify, deep_idx, encode_samples


class BCTrainer:
    def __init__(
        self,
        env: MultiAgentEnv,
        agents: AgentGroup,
        demo_filename: str,
        segment_length=8,
        minibatch_size=8,  # num segments per minibatch
        demo_limit=None,
    ):
        self.env = env
        self.segment_length = segment_length
        self.minibatch_size = minibatch_size
        self.logger: Logger = Logger()

        self.agents = agents.configured_with(env.observation_spaces, env.action_spaces)
        self.agents.reset()

        with open(demo_filename, "rb") as f:
            data = pickle.load(f)
        if demo_limit is None:
            demo_data = data
        else:
            demo_data = deep_idx(data, slice(demo_limit), copy=True)

        self.trajectory_ends = np.flatnonzero(demo_data["terminal"]) + 1
        if not demo_data["terminal"][-1]:
            self.trajectory_ends = np.concatenate(
                [self.trajectory_ends, [len(demo_data["terminal"])]]
            )
        self.trajectory_starts = np.concatenate([[0], self.trajectory_ends[:-1]])

        assert self.trajectory_starts.shape[0] >= self.minibatch_size

        segment_starts = np.concatenate(
            [
                np.arange(start, end, self.segment_length)
                for start, end in zip(self.trajectory_starts, self.trajectory_ends)
            ]
        )

        self.rollout = {
            "obs": {
                agent_id: encode_samples(agent_obs, env.observation_spaces[agent_id])
                for agent_id, agent_obs in demo_data["obs"].items()
                if agent_id != "global_state"
            },
            "actions": {
                agent_id: encode_samples(agent_action, env.action_spaces[agent_id])
                for agent_id, agent_action in demo_data["actions"].items()
            },
            "segment_starts": segment_starts,
            "segment_ends": np.concatenate(
                [segment_starts[1:], [len(demo_data["terminal"])]]
            ),
            "initial_hidden": {
                agent_id: torch.zeros((1, len(segment_starts), agents.fc_dim))
                for agent_id in demo_data["obs"].keys()
                if agent_id != "global_state"
            },
        }

    def run(self):
        # rollout keys: segment starts, segment ends, initial hidden, obs, actions
        curr = self.trajectory_starts.copy()
        while (mask := np.less(curr, self.trajectory_ends)).sum() > self.minibatch_size:
            traj_idxs = np.flatnonzero(mask)
            selected_traj_idx = np.random.choice(
                traj_idxs, size=self.minibatch_size, replace=False
            )
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

            self.logger.push(loss=loss.item())
            for optim in self.agents.optims:
                optim.zero_grad(set_to_none=True)
            loss.backward()
            for optim in self.agents.optims:
                optim.step()

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

        self.logger.step()

    def evaluate(self, num_episodes=100):
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

        if hasattr(self.env, "metrics") and "won" in self.env.metrics.keys():
            self.logger.log(winrate=wins / num_episodes)
        self.logger.step()


def main():
    from agents import RecurrentMIMAgents2
    from envs import FireCommander

    agents = RecurrentMIMAgents2(1e-3, 1e-3, 0.1)
    trainer = BCTrainer(
        FireCommander(),
        agents,
        demo_filename="demos/firecommander.pickle",
        demo_limit=2000,
    )
    for i in range(500):
        trainer.run()
        if i % 5 == 4:
            # trainer.evaluate()
            trainer.logger.generate_plots(f"{PROJECT_DIR}/baselines/bc_plotgen")
            print(
                sum(
                    trainer.rollout["initial_hidden"][agent_id].var().item()
                    for agent_id in agents.agent_ids
                )
                / 3
            )


if __name__ == "__main__":
    main()
