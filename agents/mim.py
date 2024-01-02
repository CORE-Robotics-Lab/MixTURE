from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import flat_size, shuffled_within_classes

from .comm_channel import RecurrentCommChannelAgents


# Original reverse model formulation
# (one agent's observation) + (one agent's action) -> (all OTHER embeddings)
class RecurrentMIMAgents(RecurrentCommChannelAgents):
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
                        flat_size(self.obs_spaces[policy_id])
                        + flat_size(self.action_spaces[policy_id]),
                        self.fc_dim,
                    ),
                    nn.ReLU(),
                    nn.Linear(self.fc_dim, self.fc_dim),
                    nn.ReLU(),
                    nn.Linear(self.fc_dim, (len(self.agent_ids) - 1) * self.fc_dim),
                ).to(self.device)
                for policy_id in self.policy_ids
            }
        )
        return self

    def reverse_predict_embeddings(self, agent_id, own_obs, own_logits):
        reverse_model_inputs = [
            own_obs,
            own_logits,
        ]
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
                agent_id,
                rollout["obs"][agent_id][batch_idx],
                self.prev_logits[agent_id],
            )

            actual_embeddings = []
            for other_id in shuffled_within_classes(self.agent_ids, self.pid):
                if other_id != agent_id:
                    actual_embeddings.append(self.embeddings[other_id])
            actual_embeddings = torch.cat(actual_embeddings, dim=1)

            if self.loss_fn == "mse":
                loss += curr_mim_coeff * F.mse_loss(
                    predicted_embeddings, actual_embeddings
                )
            elif self.loss_fn == "cos":
                loss += curr_mim_coeff * F.cosine_embedding_loss(
                    predicted_embeddings,
                    actual_embeddings,
                    torch.ones(predicted_embeddings.shape[0], device=self.device),
                )

        self.prev_logits = None
        return loss


# Alternative reverse model formulation
# (all actions) + (all OTHER embeddings) --> (one embedding)
class RecurrentMIMAgents2(RecurrentCommChannelAgents):
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
