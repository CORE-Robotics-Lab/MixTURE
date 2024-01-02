from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence

from utils import deep_idx, flat_size, shuffled_within_classes

from .recurrent import RecurrentAgents


class RecurrentAttentionCommAgents(RecurrentAgents):
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
        self.attention = None
        self.layer_norm = None
        self.hidden_states = None
        self.embeddings = None
        self.prev_logits = None

    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        super().configured_with(obs_spaces, action_spaces, logger, device)
        self.attention = nn.MultiheadAttention(
            self.fc_dim, num_heads=2, batch_first=True
        )
        self.action_heads = nn.ModuleDict(
            {
                policy_id: nn.Linear(
                    self.fc_dim,
                    flat_size(self.action_spaces[policy_id]),
                ).to(self.device)
                for policy_id in self.policy_ids
            }
        )
        self.layer_norm = LayerNorm(self.fc_dim)
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

        xx = torch.cat(
            [self.embeddings[agent_id] for agent_id in self.agent_ids], dim=0
        )
        all_attn_out, _ = self.attention(xx, xx, xx, need_weights=False)
        all_attn_out = self.layer_norm(xx + F.relu(all_attn_out))

        actions, logps = {}, {}

        for agent_id, attn_out in zip(self.agent_ids, all_attn_out):
            actions[agent_id], logps[agent_id] = self.policy_heads[
                self.pid[agent_id]
            ].step(self.action_heads[self.pid[agent_id]](attn_out))
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

        # treat packed sequence length / segment dimension as a batch dimension, and
        # stack agents along what the MultiheadAttention thinks the sequence dimension
        # is supposed to be
        xx = torch.stack(
            [gru_outputs[agent_id].data for agent_id in self.agent_ids], dim=1
        )
        all_attn_out, _ = self.attention(xx, xx, xx, need_weights=False)
        all_attn_out = self.layer_norm(xx + F.relu(all_attn_out))

        self.embeddings = {}
        logits = {}
        logps = {}

        for i, agent_id in enumerate(self.agent_ids):
            padded, seq_lens = pad_packed_sequence(
                PackedSequence(
                    self.action_heads[self.pid[agent_id]](all_attn_out[:, i]),
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
            logps[agent_id] = self.policy_heads[self.pid[agent_id]].logp(
                logits[agent_id], deep_idx(rollout["actions"][agent_id], batch_idx)
            )
        self.prev_logits = logits

        return logps

    def custom_loss(self, loss, rollout, batch_idx, iteration) -> torch.Tensor:
        self.prev_logits = None


class MLPAttentionMIMAgents(RecurrentAttentionCommAgents):
    def __init__(
        self,
        policy_lr: float,
        critic_lr: float,
        mim_coeff: Union[float, Callable[[int], float]],
        fc_dim=64,
        reduce_head_weights=True,
        share_parameters=True,
        permute_within_classes=True,
        loss_fn="mse",
    ):
        super().__init__(
            policy_lr=policy_lr,
            critic_lr=critic_lr,
            fc_dim=fc_dim,
            reduce_head_weights=reduce_head_weights,
            share_parameters=share_parameters,
            permute_within_classes=permute_within_classes,
        )
        self.mim_coeff = mim_coeff
        self.loss_fn = loss_fn
        self.reverse_models = None

    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        super().configured_with(obs_spaces, action_spaces, logger, device)
        self.reverse_models = nn.ModuleDict(
            {
                policy_id: nn.Sequential(
                    nn.Linear(
                        sum(
                            flat_size(self.action_spaces[self.pid[agent_id]])
                            for agent_id in self.agent_ids
                        )
                        + self.fc_dim * (len(self.agent_ids) - 1),
                        self.fc_dim,
                    ),
                    nn.ReLU(),
                    nn.Linear(self.fc_dim, self.fc_dim),
                    nn.ReLU(),
                    nn.Linear(self.fc_dim, self.fc_dim),
                ).to(self.device)
                for policy_id in self.policy_ids
            }
        )
        return self

    def reverse_predict_embeddings(self, agent_id, all_embeddings, all_logits):
        reverse_model_inputs = []
        for other_id in shuffled_within_classes(self.agent_ids, self.pid):
            if other_id == agent_id:
                continue
            reverse_model_inputs.append(all_logits[other_id])
            reverse_model_inputs.append(all_embeddings[other_id])
        reverse_model_inputs.append(all_logits[agent_id])
        return self.reverse_models[self.pid[agent_id]](
            torch.cat(reverse_model_inputs, dim=1)
        )

    def custom_loss(self, loss, rollout, batch_idx, iteration) -> torch.Tensor:
        curr_mim_coeff = (
            self.mim_coeff(iteration) if callable(self.mim_coeff) else self.mim_coeff
        )
        if curr_mim_coeff == 0:
            return loss

        for agent_id in self.agent_ids:
            predicted_embeddings = self.reverse_predict_embeddings(
                agent_id, self.embeddings, self.prev_logits
            )

            if self.loss_fn == "mse":
                additional_loss = F.mse_loss(
                    predicted_embeddings, self.embeddings[agent_id]
                )
            elif self.loss_fn == "cos":
                additional_loss = F.cosine_embedding_loss(
                    predicted_embeddings,
                    self.embeddings[agent_id],
                    torch.ones(predicted_embeddings.shape[0], device=self.device),
                )
            else:
                raise ValueError("expected loss_fn to be mse or cos")
            if self.logger is not None:
                self.logger.push(mim_loss=additional_loss.item())
            loss += curr_mim_coeff * additional_loss

        self.prev_logits = None
        return loss


class FullAttentionMIMAgents(RecurrentAttentionCommAgents):
    def __init__(
        self,
        policy_lr: float,
        critic_lr: float,
        mim_coeff: Union[float, Callable[[int], float]],
        fc_dim=64,
        reduce_head_weights=True,
        share_parameters=True,
        permute_within_classes=True,
        loss_fn="mse",
    ):
        super().__init__(
            policy_lr=policy_lr,
            critic_lr=critic_lr,
            fc_dim=fc_dim,
            reduce_head_weights=reduce_head_weights,
            share_parameters=share_parameters,
            permute_within_classes=permute_within_classes,
        )
        self.mim_coeff = mim_coeff
        self.loss_fn = loss_fn
        self.reverse_embed_decoders = None
        self.reverse_action_decoders = None
        self.reverse_attention = None
        self.reverse_linear = None

    def configured_with(
        self, obs_spaces, action_spaces, logger=None, device=torch.device("cpu")
    ):
        super().configured_with(obs_spaces, action_spaces, logger, device)
        self.reverse_embed_decoders = nn.ModuleDict(
            {
                policy_id: nn.Linear(
                    flat_size(self.action_spaces[policy_id]) + self.fc_dim, self.fc_dim
                )
                for policy_id in self.policy_ids
            }
        )
        self.reverse_action_decoders = nn.ModuleDict(
            {
                policy_id: nn.Linear(
                    flat_size(self.action_spaces[policy_id]), self.fc_dim
                )
                for policy_id in self.policy_ids
            }
        )
        self.reverse_attention = nn.ModuleDict(
            {
                policy_id: nn.MultiheadAttention(
                    embed_dim=self.fc_dim, num_heads=2, batch_first=True
                )
                for policy_id in self.policy_ids
            }
        )
        self.reverse_linear = nn.Linear(self.fc_dim, self.fc_dim)
        return self

    def reverse_predict_embeddings(self, agent_id, all_embeddings, all_logits):
        decoded_embeds = [
            self.reverse_embed_decoders[self.pid[other_id]](
                torch.cat([all_logits[other_id], all_embeddings[other_id]], dim=-1)
            )
            for other_id in self.agent_ids
        ] + [self.reverse_action_decoders[self.pid[agent_id]](all_logits[agent_id])]
        x = torch.stack(decoded_embeds, dim=-2)
        x = F.relu(x)
        x, _ = self.reverse_attention[self.pid[agent_id]](x, x, x, need_weights=False)
        x = x[..., -1, :]
        x = F.relu(x)
        x = self.reverse_linear(x)
        return x

    def custom_loss(self, loss, rollout, batch_idx, iteration) -> torch.Tensor:
        curr_mim_coeff = (
            self.mim_coeff(iteration) if callable(self.mim_coeff) else self.mim_coeff
        )
        if curr_mim_coeff == 0:
            return loss

        for agent_id in self.agent_ids:
            predicted_embeddings = self.reverse_predict_embeddings(
                agent_id, self.embeddings, self.prev_logits
            )

            if self.loss_fn == "mse":
                additional_loss = F.mse_loss(
                    predicted_embeddings, self.embeddings[agent_id]
                )
            elif self.loss_fn == "cos":
                additional_loss = F.cosine_embedding_loss(
                    predicted_embeddings,
                    self.embeddings[agent_id],
                    torch.ones(predicted_embeddings.shape[0], device=self.device),
                )
            else:
                raise ValueError("expected loss_fn to be mse or cos")
            if self.logger is not None:
                self.logger.push(mim_loss=additional_loss.item())
            loss += curr_mim_coeff * additional_loss

        self.prev_logits = None
        return loss
