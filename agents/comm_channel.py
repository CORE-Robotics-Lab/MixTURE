from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence

from utils import deep_idx, flat_size, shuffled_within_classes

from .recurrent import RecurrentAgents


class RecurrentCommChannelAgents(RecurrentAgents):
    def __init__(
        self,
        policy_lr: float,
        critic_lr: float,
        fc_dim=64,
        reduce_head_weights=True,
        share_parameters=True,
        permute_within_classes=True,
    ):
        super().__init__(
            policy_lr=policy_lr,
            critic_lr=critic_lr,
            fc_dim=fc_dim,
            reduce_head_weights=reduce_head_weights,
            share_parameters=share_parameters,
            permute_within_classes=permute_within_classes,
        )

        self.preprocessors = None
        self.grus = None
        self.action_heads = None
        self.hidden_states = None
        self.embeddings = None
        self.prev_logits = None

    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        super().configured_with(obs_spaces, action_spaces, logger, device)
        self.action_heads = nn.ModuleDict(
            {
                policy_id: nn.Linear(
                    len(self.agent_ids) * self.fc_dim,
                    flat_size(self.action_spaces[policy_id]),
                ).to(self.device)
                for policy_id in self.policy_ids
            }
        )
        with torch.no_grad():
            if self.reduce_head_weights:
                for action_head in self.action_heads.values():
                    action_head.weight *= 0.01

        self.optims.clear()
        for policy_id in self.policy_ids:
            self.optims.append(
                torch.optim.Adam(
                    list(self.preprocessors[policy_id].parameters())
                    + list(self.grus[policy_id].parameters())
                    + list(self.action_heads[policy_id].parameters()),
                    lr=self.policy_lr,
                )
            )

        self._setup_critics()
        return self

    def reset(self) -> None:
        super().reset()
        self.embeddings = {}

    def rollout_forward(
        self, all_obs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        preprocessed = {
            agent_id: self.preprocessors[self.pid[agent_id]](all_obs[agent_id])
            for agent_id in self.agent_ids
        }

        self.embeddings = {}
        for agent_id in self.agent_ids:
            output, self.hidden_states[agent_id] = self.grus[self.pid[agent_id]](
                preprocessed[agent_id].unsqueeze(0), self.hidden_states[agent_id]
            )
            self.embeddings[agent_id] = output.squeeze(0)

        actions, logps = {}, {}
        for agent_id in self.agent_ids:
            head_inputs = []
            for other_id in (
                shuffled_within_classes(self.agent_ids, self.pid)
                if self.permute_within_classes
                else self.agent_ids
            ):
                if other_id == agent_id:
                    continue
                head_inputs.append(self.embeddings[other_id])
            head_inputs.append(self.embeddings[agent_id])
            logits = self.action_heads[self.pid[agent_id]](
                torch.cat(head_inputs, dim=1)
            )
            actions[agent_id], logps[agent_id] = self.policy_heads[
                self.pid[agent_id]
            ].step(logits)

        return actions, logps

    def train_forward(self, rollout, batch_idx):
        seg_idxs = {}
        for idx in batch_idx:
            ii = np.searchsorted(rollout["segment_starts"], idx)
            if (
                ii < len(rollout["segment_starts"])
                and rollout["segment_starts"][ii] == idx
            ):
                seg_idxs[idx] = ii

        gru_outputs = {}
        for agent_id in self.agent_ids:
            sequences = [
                rollout["obs"][agent_id][
                    rollout["segment_starts"][seg_idxs[idx]] : rollout["segment_ends"][
                        seg_idxs[idx]
                    ]
                ]
                for idx in batch_idx
                if idx in seg_idxs
            ]
            sequences = [
                self.preprocessors[self.pid[agent_id]](sequence)
                for sequence in sequences
            ]
            h0 = torch.stack(
                [
                    rollout["initial_hidden"][agent_id][:, seg_idxs[idx]]
                    for idx in batch_idx
                    if idx in seg_idxs
                ],
                dim=1,
            )
            gru_outputs[agent_id], self.hidden_states[agent_id] = self.grus[
                self.pid[agent_id]
            ](
                pack_sequence(sequences, enforce_sorted=False),
                h0,
            )

        logits = {}
        ret = {}
        self.embeddings = {}
        for agent_id in self.agent_ids:
            head_inputs = []
            for other_id in (
                shuffled_within_classes(self.agent_ids, self.pid)
                if self.permute_within_classes
                else self.agent_ids
            ):
                if other_id == agent_id:
                    continue
                head_inputs.append(gru_outputs[other_id].data)
            head_inputs.append(gru_outputs[agent_id].data)
            padded, seq_lens = pad_packed_sequence(
                PackedSequence(
                    self.action_heads[self.pid[agent_id]](
                        torch.cat(head_inputs, dim=-1)
                    ),
                    batch_sizes=gru_outputs[agent_id].batch_sizes,
                    sorted_indices=gru_outputs[agent_id].sorted_indices,
                    unsorted_indices=gru_outputs[agent_id].unsorted_indices,
                ),
                batch_first=True,
            )
            padded_embeddings, _ = pad_packed_sequence(
                gru_outputs[agent_id], batch_first=True
            )
            self.embeddings[agent_id] = torch.cat(
                [
                    padded_embeddings[i][: seq_lens[i]]
                    for i in range(len(padded_embeddings))
                ],
                dim=0,
            )

            logits[agent_id] = torch.cat(
                [padded[i][: seq_lens[i]] for i in range(len(padded))], dim=0
            )
            ret[agent_id] = self.policy_heads[self.pid[agent_id]].logp(
                logits[agent_id], deep_idx(rollout["actions"][agent_id], batch_idx)
            )
        self.prev_logits = logits

        return ret

    def custom_loss(self, loss, rollout, batch_idx, iteration) -> torch.Tensor:
        self.prev_logits = None
