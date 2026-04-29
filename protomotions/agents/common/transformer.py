# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from copy import copy

import torch
from torch import nn
import numpy as np

from protomotions.utils.model_utils import get_activation_func

from hydra.utils import instantiate

import torch.nn.functional as F
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], : x.shape[2]]
        return x


class Transformer(nn.Module):
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config

        self.mask_keys = {}
        input_models = {}

        for input_key, input_config in config.input_models.items():
            input_models[input_key] = instantiate(input_config)
            self.mask_keys[input_key] = input_config.config.get("mask_key", None)

        self.input_models = nn.ModuleDict(input_models)
        self.feature_size = self.config.transformer_token_size * len(input_models)

        # Transformer layers
        self.sequence_pos_encoder = PositionalEncoding(config.latent_dim)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_size,
            dropout=config.dropout,
            activation=get_activation_func(config.activation, return_type="functional"),
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=config.num_layers
        )

        if config.get("output_model", None) is not None:
            self.output_model = instantiate(config.output_model)

    def get_extracted_features(self, input_dict):
        batch_size = next(iter(input_dict.values())).shape[0]
        device = next(iter(input_dict.values())).device
        cat_obs = []
        cat_mask = []

        for model_name, input_model in self.input_models.items():
            input_key = input_model.config.obs_key
            if input_key not in input_dict:
                print(f"Transformer expected to see key {input_key} in input_dict.")
                # Transformer token will not be created for this key
                # This acts similar to masking out the token
                continue

            key_obs = input_dict[input_key]

            if input_model.config.get("operations", None) is not None:
                for operation in input_model.config.get("operations", []):
                    if operation.type == "permute":
                        key_obs = key_obs.permute(*operation.new_order)
                    elif operation.type == "reshape":
                        new_shape = copy(operation.new_shape)
                        if new_shape[0] == "batch_size":
                            new_shape[0] = batch_size
                        key_obs = key_obs.reshape(*new_shape)
                    elif operation.type == "squeeze":
                        key_obs = key_obs.squeeze(dim=operation.squeeze_dim)
                    elif operation.type == "unsqueeze":
                        key_obs = key_obs.unsqueeze(dim=operation.unsqueeze_dim)
                    elif operation.type == "expand":
                        key_obs = key_obs.expand(*operation.expand_shape)
                    elif operation.type == "positional_encoding":
                        key_obs = self.sequence_pos_encoder(key_obs)
                    elif operation.type == "encode":
                        key_obs = {input_key: key_obs}
                        key_obs = input_model(key_obs)
                    elif operation.type == "mask_multiply":
                        num_mask_dims = len(
                            input_dict[self.mask_keys[model_name]].shape
                        )
                        num_obs_dims = len(key_obs.shape)
                        extra_needed_dims = num_obs_dims - num_mask_dims
                        key_obs = key_obs * input_dict[self.mask_keys[model_name]].view(
                            *input_dict[self.mask_keys[model_name]].shape,
                            *((1,) * extra_needed_dims),
                        )
                    elif operation.type == "mask_multiply_concat":
                        num_mask_dims = len(
                            input_dict[self.mask_keys[model_name]].shape
                        )
                        num_obs_dims = len(key_obs.shape)
                        extra_needed_dims = num_obs_dims - num_mask_dims
                        key_obs = key_obs * input_dict[self.mask_keys[model_name]].view(
                            *input_dict[self.mask_keys[model_name]].shape,
                            *((1,) * extra_needed_dims),
                        )
                        key_obs = torch.cat(
                            [
                                key_obs,
                                input_dict[self.mask_keys[model_name]].view(
                                    *input_dict[self.mask_keys[model_name]].shape,
                                    *((1,) * extra_needed_dims),
                                ),
                            ],
                            dim=-1,
                        )
                    elif operation.type == "concat_obs":
                        to_add_obs = input_dict[operation.obs_key]
                        if len(to_add_obs.shape) != len(key_obs.shape):
                            to_add_obs = to_add_obs.unsqueeze(1).expand(
                                to_add_obs.shape[0],
                                key_obs.shape[1],
                                to_add_obs.shape[-1],
                            )
                        key_obs = torch.cat([key_obs, to_add_obs], dim=-1)
                    else:
                        raise NotImplementedError(
                            f"Operation {operation} not implemented"
                        )
            else:
                key_obs = {input_key: key_obs}
                key_obs = input_model(key_obs)

            if len(key_obs.shape) == 2:
                # Add a sequence dimension
                key_obs = key_obs.unsqueeze(1)

            cat_obs.append(key_obs)

            if self.mask_keys[model_name] is not None:
                key_mask = input_dict[self.mask_keys[model_name]]
                # Our mask is 1 for valid and 0 for invalid
                # The transformer expects the mask to be 0 for valid and 1 for invalid
                key_mask = key_mask.logical_not()
            else:
                key_mask = torch.zeros(
                    batch_size,
                    key_obs.shape[1],
                    dtype=torch.bool,
                    device=device,
                )
            cat_mask.append(key_mask)

        # Concatenate all the features
        cat_obs = torch.cat(cat_obs, dim=1)
        cat_mask = torch.cat(cat_mask, dim=1)

        # obs creation works in batch_first but transformer expects seq_len first
        cat_obs = cat_obs.permute(1, 0, 2).contiguous()  # [seq_len, bs, d]

        cur_mask = cat_mask.unsqueeze(1).expand(-1, cat_obs.shape[0], -1)
        cur_mask = torch.repeat_interleave(cur_mask, self.config.num_heads, dim=0)

        output = self.seqTransEncoder(cat_obs, mask=cur_mask)[0]  # [bs, d]

        return output
    
    def input_model_forward(self, input_dict):
        batch_size = next(iter(input_dict.values())).shape[0]
        device = next(iter(input_dict.values())).device
        cat_obs = []
        cat_mask = []

        for model_name, input_model in self.input_models.items():
            input_key = input_model.config.obs_key
            if input_key not in input_dict:
                print(f"Transformer expected to see key {input_key} in input_dict.")
                # Transformer token will not be created for this key
                # This acts similar to masking out the token
                continue

            key_obs = input_dict[input_key]

            if input_model.config.get("operations", None) is not None:
                for operation in input_model.config.get("operations", []):
                    if operation.type == "permute":
                        key_obs = key_obs.permute(*operation.new_order)
                    elif operation.type == "reshape":
                        new_shape = copy(operation.new_shape)
                        if new_shape[0] == "batch_size":
                            new_shape[0] = batch_size
                        key_obs = key_obs.reshape(*new_shape)
                    elif operation.type == "squeeze":
                        key_obs = key_obs.squeeze(dim=operation.squeeze_dim)
                    elif operation.type == "unsqueeze":
                        key_obs = key_obs.unsqueeze(dim=operation.unsqueeze_dim)
                    elif operation.type == "expand":
                        key_obs = key_obs.expand(*operation.expand_shape)
                    elif operation.type == "positional_encoding":
                        key_obs = self.sequence_pos_encoder(key_obs)
                    elif operation.type == "encode":
                        key_obs = {input_key: key_obs}
                        key_obs = input_model(key_obs)
                    elif operation.type == "mask_multiply":
                        num_mask_dims = len(
                            input_dict[self.mask_keys[model_name]].shape
                        )
                        num_obs_dims = len(key_obs.shape)
                        extra_needed_dims = num_obs_dims - num_mask_dims
                        key_obs = key_obs * input_dict[self.mask_keys[model_name]].view(
                            *input_dict[self.mask_keys[model_name]].shape,
                            *((1,) * extra_needed_dims),
                        )
                    elif operation.type == "mask_multiply_concat":
                        num_mask_dims = len(
                            input_dict[self.mask_keys[model_name]].shape
                        )
                        num_obs_dims = len(key_obs.shape)
                        extra_needed_dims = num_obs_dims - num_mask_dims
                        key_obs = key_obs * input_dict[self.mask_keys[model_name]].view(
                            *input_dict[self.mask_keys[model_name]].shape,
                            *((1,) * extra_needed_dims),
                        )
                        key_obs = torch.cat(
                            [
                                key_obs,
                                input_dict[self.mask_keys[model_name]].view(
                                    *input_dict[self.mask_keys[model_name]].shape,
                                    *((1,) * extra_needed_dims),
                                ),
                            ],
                            dim=-1,
                        )
                    elif operation.type == "concat_obs":
                        to_add_obs = input_dict[operation.obs_key]
                        if len(to_add_obs.shape) != len(key_obs.shape):
                            to_add_obs = to_add_obs.unsqueeze(1).expand(
                                to_add_obs.shape[0],
                                key_obs.shape[1],
                                to_add_obs.shape[-1],
                            )
                        key_obs = torch.cat([key_obs, to_add_obs], dim=-1)
                    else:
                        raise NotImplementedError(
                            f"Operation {operation} not implemented"
                        )
            else:
                key_obs = {input_key: key_obs}
                key_obs = input_model(key_obs)

            if len(key_obs.shape) == 2:
                # Add a sequence dimension
                key_obs = key_obs.unsqueeze(1)
            
            # print(f"Key {model_name} has shape {key_obs.shape} and dtype {key_obs.dtype}")
            cat_obs.append(key_obs)

            if self.mask_keys[model_name] is not None:
                key_mask = input_dict[self.mask_keys[model_name]]
                # Our mask is 1 for valid and 0 for invalid
                # The transformer expects the mask to be 0 for valid and 1 for invalid
                key_mask = key_mask.logical_not()
            else:
                key_mask = torch.zeros(
                    batch_size,
                    key_obs.shape[1],
                    dtype=torch.bool,
                    device=device,
                )
            cat_mask.append(key_mask)

        # Concatenate all the features
        cat_obs = torch.cat(cat_obs, dim=1)
        cat_mask = torch.cat(cat_mask, dim=1)

        # obs creation works in batch_first but transformer expects seq_len first
        cat_obs = cat_obs.permute(1, 0, 2).contiguous()  # [seq_len, bs, d]

        cur_mask = cat_mask.unsqueeze(1).expand(-1, cat_obs.shape[0], -1)
        cur_mask = torch.repeat_interleave(cur_mask, self.config.num_heads, dim=0)
        return cat_obs, cur_mask
    
    def transformer_forward(self, cat_obs):
        output = self.seqTransEncoder(cat_obs)[0]
        return output

    def output_model_forward(self, output):
        if self.config.get("output_model", None) is not None:
            output = self.output_model(output)
        return output


    def forward(self, input_dict):
        output = self.get_extracted_features(input_dict)

        if self.config.get("output_model", None) is not None:
            output = self.output_model(output)

        return output, torch.tensor(0.0)
    
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Transformer layers
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_size,
            dropout=config.dropout,
            activation=get_activation_func(config.activation, return_type="functional"),
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=config.num_layers
        )
    def forward(self, x):
        x = self.seqTransEncoder(x)[0]
        return x


class MultiBranchControlNet(nn.Module):
    def __init__(self, 
                base_cfg: dict,
                trainable_cfg: dict,
                zero_linear_cfg: dict,
                residual: bool,
                spare_gate: bool,
    ):
        super().__init__()
        self.residual_moe = residual
        self.spare_gate = spare_gate
        # Translated comment.
        self.base = instantiate(base_cfg)
        for p in self.base.input_models.parameters():
            p.requires_grad = False
        for p in self.base.seqTransEncoder.parameters():
            p.requires_grad = False

        # Translated comment.
        transformer_layers = []
        zero_linear_projs = []
        for _ in range(trainable_cfg.config.branch_num):
            transformer_layers.append(instantiate(trainable_cfg))
            zero_linear_projs.append(instantiate(zero_linear_cfg))
        self.adapters = nn.ModuleList(transformer_layers)
        self.adapters_zero = nn.ModuleList(zero_linear_projs)


        # Translated comment.
        with torch.no_grad():
            for zero_linear_proj in self.adapters_zero:
                zero_linear_proj.weight.zero_()
                if zero_linear_proj.bias is not None:
                    zero_linear_proj.bias.zero_()
        if self.residual_moe:
            # Translated comment.
            self.gate_net = nn.Sequential(
                nn.Linear(base_cfg.config.transformer_token_size, 256),
                nn.ReLU(),
                nn.Linear(256, trainable_cfg.config.branch_num),  # translated comment
                )

        else:
            # Translated comment.
            self.gate_net = nn.Sequential(
                nn.Linear(base_cfg.config.transformer_token_size, 256),
                nn.ReLU(),
                nn.Linear(256, trainable_cfg.config.branch_num+1),  # translated comment
                )

        # Translated comment.
        for layer in self.gate_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, input_dict):
        """
        Returns: action or predicted value.
        """
        # Translated comment.
        # with torch.no_grad():
        cat_obs, _ = self.base.input_model_forward(input_dict) # # [seq_len, bs, d]
        _cat_obs = cat_obs.permute(1, 0, 2)  # [bs, seq_len, d]
        logits = self.gate_net(_cat_obs).mean(dim=1)            # [B, 3]
        if self.residual_moe:
            embedding = self.base.transformer_forward(cat_obs)
            if self.spare_gate:
                # spare gate
                # Translated comment.
                probs = torch.sigmoid(logits)          # [B, 3]
                # Translated comment.
                gate_hard = (probs > 0.5).float()      # translated comment
                gate_w = gate_hard.detach() - probs.detach() + probs

                self.expert_selection = gate_hard.detach()  # translated comment
                # Translated comment.
                for i, adapter in enumerate(self.adapters):

                    # Translated comment.
                    if gate_hard[:, i].any().item():
                        out = self.adapters_zero[i](adapter(cat_obs))  # [B, D]
                        # Translated comment.
                        embedding = embedding + out * gate_w[:, i].unsqueeze(-1)
            else:
                # dense gate
                weight = torch.softmax(logits, dim=-1)          # [B, 3]
                # Translated comment.
                for i, adapter in enumerate(self.adapters):
                    out = self.adapters_zero[i](adapter(cat_obs))
                    # Translated comment.
                    embedding = embedding + out * weight[:, i].unsqueeze(-1)
        else:
            if self.spare_gate:
                # spare gate
                probs = torch.sigmoid(logits)
                gate_hard = (probs > 0.5).float()
                gate_w = gate_hard.detach() - probs.detach() + probs
                # Translated comment.
                # if gate_hard[:, 0].any().item():
                # Translated comment.
                # Translated comment.
                embedding = self.base.transformer_forward(cat_obs) * gate_w[:, 0]
                for i, adapter in enumerate(self.adapters):
                    # Translated comment.
                    if gate_hard[:, i+1].any().item():
                        out = self.adapters_zero[i](adapter(cat_obs))
                        # Translated comment.
                        embedding = embedding + out * gate_w[:, i+1].unsqueeze(-1)
            else:
                # dense gate
                weight = torch.softmax(logits, dim=-1)
                # Translated comment.
                embedding = self.base.transformer_forward(cat_obs) * weight[:, 0]
                for i, adapter in enumerate(self.adapters):
                    out = self.adapters_zero[i](adapter(cat_obs))
                    # Translated comment.
                    embedding = embedding + out * weight[:, i+1].unsqueeze(-1)
        return self.base.output_model_forward(embedding)


class Residual_Moe(torch.nn.Module):
    def __init__(self, 
                base_cfg: dict,
                trainable_cfg: dict,
                zero_linear_cfg: dict,
                residual: bool,
                spare_gate: bool,
    ):
        super().__init__()
        self.residual_moe = residual
        self.spare_gate = spare_gate
        # Translated comment.
        self.base = instantiate(base_cfg)
        for p in self.base.input_models.parameters():
            p.requires_grad = False
        for p in self.base.seqTransEncoder.parameters():
            p.requires_grad = False
        # for p in self.base.output_model.parameters():
        #     p.requires_grad = False

        # Translated comment.
        self.expert_num = trainable_cfg.config.branch_num
        self.idxs       = torch.arange(0, self.expert_num+1)
        transformer_layers = []
        zero_linear_projs = []
        for _ in range(self.expert_num):
            transformer_layers.append(instantiate(trainable_cfg))
            zero_linear_projs.append(instantiate(zero_linear_cfg))
        self.adapters = nn.ModuleList(transformer_layers)
        self.adapters_zero = nn.ModuleList(zero_linear_projs)

        # Translated comment.
        with torch.no_grad():
            for zero_linear_proj in self.adapters_zero:
                zero_linear_proj.weight.zero_()
                if zero_linear_proj.bias is not None:
                    zero_linear_proj.bias.zero_()
        # if self.residual_moe:
        self.gate_net = nn.Sequential(
            nn.Linear(base_cfg.config.transformer_token_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.expert_num+1),  # translated comment
            )

        # Translated comment.
        for layer in self.gate_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # self.all_embeddings = []
        # self.all_expert_ids = []


    def forward(self, input_dict):
        # Translated comment.
        cat_obs, _ = self.base.input_model_forward(input_dict)  # [seq_len, B, d]
        prop = cat_obs.permute(1, 0, 2)[:,0,:]
        refe = cat_obs.permute(1, 0, 2)[:,1,:]            # translated comment
        # Translated comment.
        gate_obs = torch.cat([prop, refe], dim=-1)  # [B, 2*d]


        # 2) Gate logits → p_gate
        logits = self.gate_net(gate_obs)                         # [B, n+1]
        p_gate = F.softmax(logits, dim=-1)                       # [B, n+1]

        # Translated comment.
        velocity = input_dict["mimic_target_poses_velocity"]  # [B]

        quantile = torch.linspace(0, 1, self.expert_num+2, device=cat_obs.device)[1:-1]  # [n]
        # print("quantile = ", quantile)
        bins = torch.quantile(velocity, quantile)  # [n]
        # print("bins = ", bins)

        k_star = torch.bucketize(velocity, bins)  # translated comment
        # print("k_star = ", k_star)


        loss_gate = F.cross_entropy(logits, k_star) 

        # 6) 
        emb = self.base.transformer_forward(cat_obs)         # [B, D], no grad


        # Translated comment.
        if self.residual_moe:
            ks       = logits.argmax(dim=-1)                    # translated comment
            self.expert_selection = ks.detach()

            # print("expert_selection = ", self.expert_selection)
            residual = torch.zeros_like(emb)                  # [B, D]
            w        = p_gate.flip(1).cumsum(1).flip(1)[:, 1:]   # [B, n]

            for i, adapter in enumerate(self.adapters, start=1):
                mask = ks >= i                                  # [B] bool
                if not mask.any():
                    continue
                # Translated comment.
                x_i   = cat_obs.permute(1, 0, 2)[mask].permute(1, 0, 2)                       # [seq_len, B, d]
                out_i = self.adapters_zero[i-1](adapter(x_i))                         # [b_i, D]

                # Translated comment.
                residual[mask] += out_i * w[mask, i-1].unsqueeze(-1)  # [b_i, D] * [b_i, 1] -> [b_i, D]

            emb = emb + residual
            # self.all_expert_ids.append(self.expert_selection.cpu())
            # Translated comment.
        # Translated comment.
        output = self.base.output_model_forward(emb)
        return output, loss_gate

class Residual_Moe_No_SAR(torch.nn.Module):
    def __init__(self, 
                base_cfg: dict,
                trainable_cfg: dict,
                zero_linear_cfg: dict,
                residual: bool,
                spare_gate: bool,
    ):
        super().__init__()
        self.residual_moe = residual
        self.spare_gate = spare_gate
        # Translated comment.
        self.base = instantiate(base_cfg)
        for p in self.base.input_models.parameters():
            p.requires_grad = False
        for p in self.base.seqTransEncoder.parameters():
            p.requires_grad = False
        # for p in self.base.output_model.parameters():
        #     p.requires_grad = False

        # Translated comment.
        self.expert_num = trainable_cfg.config.branch_num
        self.idxs       = torch.arange(0, self.expert_num+1)
        transformer_layers = []
        zero_linear_projs = []
        for _ in range(self.expert_num):
            transformer_layers.append(instantiate(trainable_cfg))
            zero_linear_projs.append(instantiate(zero_linear_cfg))
        self.adapters = nn.ModuleList(transformer_layers)
        self.adapters_zero = nn.ModuleList(zero_linear_projs)

        # Translated comment.
        with torch.no_grad():
            for zero_linear_proj in self.adapters_zero:
                zero_linear_proj.weight.zero_()
                if zero_linear_proj.bias is not None:
                    zero_linear_proj.bias.zero_()
        # if self.residual_moe:
        self.gate_net = nn.Sequential(
            nn.Linear(base_cfg.config.transformer_token_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.expert_num+1),  # translated comment
            )

        # Translated comment.
        for layer in self.gate_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    def forward(self, input_dict):
        # Translated comment.
        cat_obs, _ = self.base.input_model_forward(input_dict)  # [seq_len, B, d]
        prop = cat_obs.permute(1, 0, 2)[:,0,:]
        refe = cat_obs.permute(1, 0, 2)[:,1,:]            # translated comment
        # Translated comment.
        gate_obs = torch.cat([prop, refe], dim=-1)  # [B, 2*d]


        # 2) Gate logits → p_gate
        logits = self.gate_net(gate_obs)                         # [B, n+1]
        p_gate = F.softmax(logits, dim=-1)                       # [B, n+1]

        # Translated comment.
        # velocity = input_dict["mimic_target_poses_velocity"]  # [B]

        # quantile = torch.linspace(0, 1, self.expert_num+2, device=cat_obs.device)[1:-1]  # [n]
        # # print("quantile = ", quantile)
        # bins = torch.quantile(velocity, quantile)  # [n]
        # # print("bins = ", bins)

        # Translated comment.
        # # print("k_star = ", k_star)

        # loss_gate = F.cross_entropy(logits, k_star) 
        loss_gate = torch.tensor(0.0)  # translated comment
        # 6) 
        emb = self.base.transformer_forward(cat_obs)         # [B, D], no grad


        # Translated comment.
        if self.residual_moe:
            ks       = logits.argmax(dim=-1)                    # translated comment
            self.expert_selection = ks.detach()

            residual = torch.zeros_like(emb)                  # [B, D]
            w        = p_gate.flip(1).cumsum(1).flip(1)[:, 1:]   # [B, n]

            for i, adapter in enumerate(self.adapters, start=1):
                mask = ks >= i                                  # [B] bool
                if not mask.any():
                    continue
                # Translated comment.
                x_i   = cat_obs.permute(1, 0, 2)[mask].permute(1, 0, 2)                       # [seq_len, B, d]
                out_i = self.adapters_zero[i-1](adapter(x_i))                         # [b_i, D]

                # Translated comment.
                residual[mask] += out_i * w[mask, i-1].unsqueeze(-1)  # [b_i, D] * [b_i, 1] -> [b_i, D]

            emb = emb + residual

        # Translated comment.
        output = self.base.output_model_forward(emb)
        return output, loss_gate
    
class Full_Moe(torch.nn.Module):
    def __init__(self, 
                base_cfg: dict,
                trainable_cfg: dict,
                zero_linear_cfg: dict,
                residual: bool,
                spare_gate: bool,
    ):
        super().__init__()
        self.spare_gate = spare_gate
        # Translated comment.
        self.base = instantiate(base_cfg)
        for p in self.base.input_models.parameters():
            p.requires_grad = False
        # for p in self.base.seqTransEncoder.parameters():
        #     p.requires_grad = False
        # for p in self.base.output_model.parameters():
        #     p.requires_grad = False

        # Translated comment.
        self.expert_num = trainable_cfg.config.branch_num
        self.idxs       = torch.arange(0, self.expert_num+1)
        transformer_layers = []
        zero_linear_projs = []
        for _ in range(self.expert_num):
            transformer_layers.append(instantiate(trainable_cfg))
            zero_linear_projs.append(instantiate(zero_linear_cfg))
        self.adapters = nn.ModuleList(transformer_layers)
        self.adapters_zero = nn.ModuleList(zero_linear_projs)

        # Translated comment.
        with torch.no_grad():
            for zero_linear_proj in self.adapters_zero:
                zero_linear_proj.weight.zero_()
                if zero_linear_proj.bias is not None:
                    zero_linear_proj.bias.zero_()
        # if self.residual_moe:
        self.gate_net = nn.Sequential(
            nn.Linear(base_cfg.config.transformer_token_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.expert_num+1),  # translated comment
            )

        # Translated comment.
        for layer in self.gate_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    def forward(self, input_dict):
        # Translated comment.
        cat_obs, _ = self.base.input_model_forward(input_dict)  # [seq_len, B, d]
        prop = cat_obs.permute(1, 0, 2)[:,0,:]
        refe = cat_obs.permute(1, 0, 2)[:,1,:]            # translated comment
        # Translated comment.
        gate_obs = torch.cat([prop, refe], dim=-1)  # [B, 2*d]


        # 2) Gate logits → p_gate
        logits = self.gate_net(gate_obs)                         # [B, n+1]
        p_gate = F.softmax(logits, dim=-1)                       # [B, n+1]

        # Translated comment.
        velocity = input_dict["mimic_target_poses_velocity"]  # [B]

        quantile = torch.linspace(0, 1, self.expert_num+2, device=cat_obs.device)[1:-1]  # [n]
        # print("quantile = ", quantile)
        bins = torch.quantile(velocity, quantile)  # [n]
        # print("bins = ", bins)

        k_star = torch.bucketize(velocity, bins)  # translated comment
        # print("k_star = ", k_star)


        loss_gate = F.cross_entropy(logits, k_star) 

        # 6) 
        emb = self.base.transformer_forward(cat_obs)         # [B, D], no grad


        # Translated comment.
        ks       = logits.argmax(dim=-1)                    # translated comment
        self.expert_selection = ks.detach()

        residual = torch.zeros_like(emb)                  # [B, D]
        w        = p_gate.flip(1).cumsum(1).flip(1)[:, 1:]   # [B, n]
        emb *= p_gate.flip(1).cumsum(1).flip(1)[:, 0].unsqueeze(-1)

        for i, adapter in enumerate(self.adapters, start=1):
            mask = ks >= i                                  # [B] bool
            if not mask.any():
                continue
            # Translated comment.
            x_i   = cat_obs.permute(1, 0, 2)[mask].permute(1, 0, 2)                       # [seq_len, B, d]
            out_i = self.adapters_zero[i-1](adapter(x_i))                         # [b_i, D]

            # Translated comment.
            residual[mask] += out_i * w[mask, i-1].unsqueeze(-1)  # [b_i, D] * [b_i, 1] -> [b_i, D]

            emb = emb + residual

        # Translated comment.
        output = self.base.output_model_forward(emb)
        return output, loss_gate

class No_Moe(torch.nn.Module):
    def __init__(self, 
                base_cfg: dict,
                trainable_cfg: dict,
                zero_linear_cfg: dict,
                residual: bool,
                spare_gate: bool,
    ):
        super().__init__()
        # self.spare_gate = spare_gate
        # Translated comment.
        self.base = instantiate(base_cfg)
        for p in self.base.input_models.parameters():
            p.requires_grad = False
        

    def forward(self, input_dict):
        cat_obs, _ = self.base.input_model_forward(input_dict)  # [seq_len, B, d]
        loss_gate = torch.tensor(0.0)
        emb = self.base.transformer_forward(cat_obs)         # [B, D]
        output = self.base.output_model_forward(emb)
        self.expert_selection = torch.zeros(cat_obs.shape[1], dtype=torch.int64, device=cat_obs.device)  # translated comment
        return output, loss_gate

    
class Residual_Moe_TOP1(torch.nn.Module):
    def __init__(self, 
                base_cfg: dict,
                trainable_cfg: dict,
                zero_linear_cfg: dict,
                residual: bool,
                spare_gate: bool,
    ):
        super().__init__()
        self.residual_moe = residual
        self.spare_gate = spare_gate
        # Translated comment.
        self.base = instantiate(base_cfg)
        for p in self.base.input_models.parameters():
            p.requires_grad = False
        for p in self.base.seqTransEncoder.parameters():
            p.requires_grad = False
        # for p in self.base.output_model.parameters():
        #     p.requires_grad = False

        # Translated comment.
        self.expert_num = trainable_cfg.config.branch_num
        transformer_layers = []
        zero_linear_projs = []
        for _ in range(self.expert_num):
            transformer_layers.append(instantiate(trainable_cfg))
            zero_linear_projs.append(instantiate(zero_linear_cfg))
        self.adapters = nn.ModuleList(transformer_layers)
        self.adapters_zero = nn.ModuleList(zero_linear_projs)

        # Translated comment.
        with torch.no_grad():
            for zero_linear_proj in self.adapters_zero:
                zero_linear_proj.weight.zero_()
                if zero_linear_proj.bias is not None:
                    zero_linear_proj.bias.zero_()
        # if self.residual_moe:
        self.gate_net = nn.Sequential(
            nn.Linear(base_cfg.config.transformer_token_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.expert_num),  # translated comment
            )

        # Translated comment.
        for layer in self.gate_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    def forward(self, input_dict):
        # Translated comment.
        cat_obs, _ = self.base.input_model_forward(input_dict)  # [seq_len, B, d]
        prop = cat_obs.permute(1, 0, 2)[:,0,:]
        refe = cat_obs.permute(1, 0, 2)[:,1,:]            # translated comment
        # Translated comment.
        gate_obs = torch.cat([prop, refe], dim=-1)  # [B, 2*d]


        # 2) Gate logits → p_gate
        logits = self.gate_net(gate_obs)                         # [B, n+1]
        p_gate = F.softmax(logits, dim=-1)                       # [B, n+1]

        # Translated comment.
        velocity = input_dict["mimic_target_poses_velocity"]  # [B]

        quantile = torch.linspace(0, 1, self.expert_num+1, device=cat_obs.device)[1:-1]  # [n]
        # print("quantile = ", quantile)
        bins = torch.quantile(velocity, quantile)  # [n]
        # print("bins = ", bins)

        k_star = torch.bucketize(velocity, bins)  # translated comment
        # print("k_star = ", k_star)


        loss_gate = F.cross_entropy(logits, k_star) 

        # 6) 
        emb = self.base.transformer_forward(cat_obs)         # [B, D], no grad


        # Translated comment.
        if self.residual_moe:
            # Translated comment.
            ks = logits.argmax(dim=-1)                     # [B]
            self.expert_selection = ks.detach()            # translated comment
            # Translated comment.
            # Translated comment.
            prob_top1 = p_gate[torch.arange(p_gate.size(0)), ks]  # [B]

            # Translated comment.
            residual = torch.zeros_like(emb)               # [B, D]

            # Translated comment.
            for expert_idx, adapter in enumerate(self.adapters):
                mask = ks == expert_idx                  # translated comment
                if not mask.any():

                    continue
                # Translated comment.
                # Translated comment.
                x_i = cat_obs[:, mask, :]                 # [seq_len, b_i, d]
                # Translated comment.
                out_i = self.adapters_zero[expert_idx](adapter(x_i))  # [b_i, D]
                # Translated comment.
                # Translated comment.
                residual[mask] = out_i * prob_top1[mask].unsqueeze(-1)  # [b_i, D]

            # Translated comment.
            emb = emb + residual

        # if self.residual_moe:
        # Translated comment.
        #     self.expert_selection = ks.detach()
        #     residual = torch.zeros_like(emb)                  # [B, D]
        #     w        = p_gate.flip(1).cumsum(1).flip(1)[:, 1:]   # [B, n]

        #     for i, adapter in enumerate(self.adapters, start=1):
        #         mask = ks >= i                                  # [B] bool
        #         if not mask.any():
        #             continue
        # Translated comment.
        #         x_i   = cat_obs.permute(1, 0, 2)[mask].permute(1, 0, 2)                       # [seq_len, B, d]
        #         out_i = self.adapters_zero[i-1](adapter(x_i))                         # [b_i, D]

        # Translated comment.
        #         residual[mask] += out_i * w[mask, i-1].unsqueeze(-1)  # [b_i, D] * [b_i, 1] -> [b_i, D]

        #     emb = emb + residual

        # Translated comment.
        output = self.base.output_model_forward(emb)
        return output, loss_gate

    # def forward(self, input_dict):
    # Translated comment.
    #     cat_obs, _ = self.base.input_model_forward(input_dict)  # [seq_len, B, d]
    #     _cat_obs   = cat_obs.permute(1, 0, 2)                    # [B, seq_len, d]

    # Translated comment.
    #     logits = self.gate_net(_cat_obs).mean(dim=1)             # [B, n+1]

    # Translated comment.
    #     if self.residual_moe:
    # Translated comment.
    #         embedding = self.base.transformer_forward(cat_obs)  # e.g. [B, D]

    #         p_soft = F.softmax(logits, dim=-1)                  

    # Translated comment.
    # Translated comment.
    # Translated comment.
    # Translated comment.
    #         residual = embedding.new_zeros_like(embedding)     # [B, D]

    # Translated comment.
    # Translated comment.
    #         for i, adapter in enumerate(self.adapters, start=1):
    # Translated comment.
    #             mask = ks >= i                              # [B] bool
    #             if not mask.any():
    # Translated comment.
    # Translated comment.
    #             x_i   = _cat_obs[mask].permute(1, 0, 2)                       # [seq_len, B, d]
    #             out_i = self.adapters_zero[i](adapter(x_i))                         # [b_i, D]

    # Translated comment.
    #             residual[mask] += out_i * w[mask, i-1].unsqueeze(-1)  # [b_i, D] * [b_i, 1] -> [b_i, D]

    # Translated comment.
    #         embedding = embedding + residual

    #     return self.base.output_model_forward(embedding)


# class MultiBranchControlNetSpeed(nn.Module):
#     def __init__(self, 
#                 base_cfg: dict,
#                 trainable_cfg: dict,
#                 zero_linear_cfg: dict,
#                 speed_predictor_cfg: dict,
#     ):
#         super().__init__()
# Translated comment.
#         self.base = instantiate(base_cfg)
#         for p in self.base.parameters():
#             p.requires_grad = False

# Translated comment.
#         # self.trainable_transformer_layer = instantiate(trainable_cfg)
#         # self.zero_linear_proj = instantiate(zero_linear_cfg)

# Translated comment.
#         self.transformer_layer_1 = instantiate(trainable_cfg)
#         self.transformer_layer_2 = instantiate(trainable_cfg)
#         self.transformer_layer_3 = instantiate(trainable_cfg)
#         self.adapters = nn.ModuleList([
#             self.transformer_layer_1,
#             self.transformer_layer_2,
#             self.transformer_layer_3,
#         ])

# Translated comment.
#         self.zero_linear_proj_1 = instantiate(zero_linear_cfg)
#         self.zero_linear_proj_2 = instantiate(zero_linear_cfg)
#         self.zero_linear_proj_3 = instantiate(zero_linear_cfg)
#         with torch.no_grad():
#             self.zero_linear_proj_1.weight.zero_()
#             if self.zero_linear_proj_1.bias is not None:
#                 self.zero_linear_proj_1.bias.zero_()
#             self.zero_linear_proj_2.weight.zero_()
#             if self.zero_linear_proj_2.bias is not None:
#                 self.zero_linear_proj_2.bias.zero_()
#             self.zero_linear_proj_3.weight.zero_()
#             if self.zero_linear_proj_3.bias is not None:
#                 self.zero_linear_proj_3.bias.zero_()

#         self.adapters_zero = nn.ModuleList([
#             self.zero_linear_proj_1,
#             self.zero_linear_proj_2,
#             self.zero_linear_proj_3,
#         ])

# Translated comment.
#         self.gate_net = nn.Sequential(
#             nn.Linear(base_cfg.config.transformer_token_size, 256),
#             nn.ReLU(),
# Translated comment.
#         )
# Translated comment.
#         self.manual_gate: Optional[int] = None

#         # self.speed_predictor_cfg = speed_predictor_cfg
#         # if self.speed_predictor_cfg.config.enabled:
# Translated comment.
#         #     self.speed_predictor = instantiate(speed_predictor_cfg)

#     def set_manual_gate(self, n: Optional[int]):
# Translated comment.
# Translated comment.
#         self.manual_gate = n

#     def forward(self, input_dict):
#         """
# Translated comment.
#         """
#         speed_pred = torch.tensor(0.0)
#         speed_true = torch.tensor(0.0)
# Translated comment.
#         # with torch.no_grad():
#         cat_obs, _ = self.base.input_model_forward(input_dict) # # [seq_len, bs, d]
#         e0 = self.base.transformer_forward(cat_obs)
#         e1 = e0 + self.adapters_zero[0](self.adapters[0](cat_obs)) # # [bs, d]

#         one_speed_feature = self.one_speed_forward(input_dict)
#         action_1 = self.base.output_model_forward(one_speed_feature)

# Translated comment.
#         if self.manual_gate is not None:
#             e = e1
#             if self.manual_gate > 1:
#                 high_dynamic_feature_delta = self.adapters_zero[self.manual_gate-1](self.adapters[self.manual_gate-1](cat_obs))
#                 e = e + high_dynamic_feature_delta
#                 # if self.speed_predictor_cfg.config.enabled:
# Translated comment.
#                     # speed_true = input_dict["mimic_target_speed"]
#                     # one_speed_feature = self.one_speed_forward(input_dict)
# Translated comment.
#                     # concat_feature = torch.cat([e, one_speed_feature], dim=-1)  # [B, D + D]
# Translated comment.
#                     # speed_pred = 1 + self.speed_predictor(concat_feature)  # [B, 1]
#             action_2 = self.base.output_model_forward(e)
# Translated comment.
# Translated comment.
#             action = torch.cat([action_1, action_2], dim=-1)  # [B, D + D]
#             return action, speed_pred, speed_true

# Translated comment.
# Translated comment.
#         logits = self.gate_net(e1)             # [B, 3]
#         probs = torch.sigmoid(logits)          # [B, 3]
# Translated comment.
# Translated comment.
# Translated comment.
# Translated comment.
#         gate_w = gate_hard.detach() - probs.detach() + probs

# Translated comment.
#         e = e1
#         for i, adapter in enumerate(self.adapters[1:]):
# Translated comment.
#             if gate_hard[:, i].any().item():
#                 out = self.adapters_zero[i+1](adapter(cat_obs))           # [B, D]
# Translated comment.
#                 e = e + out * gate_w[:, i:i+1]
#         return self.base.output_model_forward(e), torch.tensor(0.0), torch.tensor(0.0)
    
#     def one_speed_forward(self, input_dict):
# Translated comment.
#         input_dict["mimic_target_poses"] = input_dict["mimic_target_poses_1"]
#         with torch.no_grad():
#             cat_obs, _ = self.base.input_model_forward(input_dict) # # [seq_len, bs, d]
#             e0 = self.base.transformer_forward(cat_obs)
#             one_speed_feature = e0 + self.adapters_zero[0](self.adapters[0](cat_obs)) # # [bs, d]
#         return one_speed_feature


# class ControlNetTransformer(nn.Module):
#     def __init__(self, *,
#                 base_cfg: dict,
#                 trainable_cfg: dict,
#                 zero_linear_cfg: dict,
#                 predictor_cfg: dict,
#                 ):
#         super().__init__()
# Translated comment.
#         self.base = instantiate(base_cfg)
# Translated comment.
#         # for p in self.base.parameters():
#         #     p.requires_grad = False

# Translated comment.
#         self.trainable_transformer_layer = instantiate(trainable_cfg)

# Translated comment.
#         self.zero_linear_proj = instantiate(zero_linear_cfg)
#         with torch.no_grad():
#             self.zero_linear_proj.weight.zero_()
#             if self.zero_linear_proj.bias is not None:
#                 self.zero_linear_proj.bias.zero_()

#         self.predictor_cfg = predictor_cfg
#         if self.predictor_cfg.config.enabled:
# Translated comment.
#             self.predictor = instantiate(predictor_cfg)

#     def forward(self, input_dict):
#         # a) base output（no grad）
#         with torch.no_grad():
#             cat_obs, _ = self.base.input_model_forward(input_dict) # # [seq_len, bs, d]
#             embedding = self.base.transformer_forward(cat_obs)

#         # b) trainable output
#         prior = self.trainable_transformer_layer(cat_obs) # # [bs, d]
#         delta_embedding = self.zero_linear_proj(prior) # # [bs, d]

#         fused = embedding + delta_embedding # # [bs, d]

#         output = self.base.output_model_forward(fused)

#         if self.predictor_cfg.config.enabled:
# Translated comment.
#             embedding_prior, embedding_1 = self.predictor_forward(input_dict, prior)
#             return output, embedding_prior, embedding_1

#         return output, torch.tensor(0.0), torch.tensor(0.0)
    
#     def predictor_forward(self, input_dict, prior):
# Translated comment.
#         input_dict["mimic_target_poses"] = input_dict["mimic_target_poses_1"]
#         with torch.no_grad():
#             cat_obs_1, _ = self.base.input_model_forward(input_dict)
#             embedding_1 = self.base.transformer_forward(cat_obs_1)
# Translated comment.
#         embedding_prior = self.predictor(prior)
#         return embedding_prior, embedding_1

# class MultiRateTransformer(nn.Module):
#     def __init__(self, *,
#                 base_cfg: dict,
#                 trainable_cfg: dict,
#                 zero_linear_cfg: dict,
#                 embedding_loss: bool,
#                 ):
#         super().__init__()
# Translated comment.
#         self.base = instantiate(base_cfg)

# Translated comment.
#         self.transformer_layer_1 = instantiate(trainable_cfg)
#         self.transformer_layer_2 = instantiate(trainable_cfg)

# Translated comment.
#         self.zero_linear_proj_1 = instantiate(zero_linear_cfg)
#         self.zero_linear_proj_2 = instantiate(zero_linear_cfg)
#         with torch.no_grad():
#             self.zero_linear_proj_1.weight.zero_()
#             if self.zero_linear_proj_1.bias is not None:
#                 self.zero_linear_proj_1.bias.zero_()
#             self.zero_linear_proj_2.weight.zero_()
#             if self.zero_linear_proj_2.bias is not None:
#                 self.zero_linear_proj_2.bias.zero_()

#         self.embedding_loss = embedding_loss

#     def forward(self, input_dict):
#         # a) base output（no grad）
#         with torch.no_grad():
#             cat_obs, _ = self.base.input_model_forward(input_dict) # # [seq_len, bs, d]
        
#         embedding_1_prime = self.zero_linear_proj_1(self.transformer_layer_1(cat_obs)) # # [bs, d]
#         delta_embedding_2_prime = self.zero_linear_proj_2(self.transformer_layer_2(cat_obs)) # # [bs, d]
#         delta_embedding_3_prime = self.base.transformer_forward(cat_obs)

#         fused = embedding_1_prime + delta_embedding_2_prime + delta_embedding_3_prime # # [bs, d]
#         output = self.base.output_model_forward(fused)

#         if self.embedding_loss:
#             # b) get real embedding
#             embedding_3 = self.embedding_inference(input_dict) # # [bs, d]
#             input_dict["mimic_target_poses"] = input_dict["mimic_target_poses_2"]
#             embedding_2 = self.embedding_inference(input_dict)
#             input_dict["mimic_target_poses"] = input_dict["mimic_target_poses_1"]
#             embedding_1 = self.embedding_inference(input_dict)

#             # c) get embedding loss
#             embedding_loss_1 = torch.mean(torch.square(embedding_1 - embedding_1_prime))
#             embedding_loss_2 = torch.mean(torch.square((embedding_2 - embedding_1) - delta_embedding_2_prime))
#             embedding_loss_3 = torch.mean(torch.square((embedding_3 - embedding_2) - delta_embedding_3_prime))
#             embedding_loss = (embedding_loss_1 + embedding_loss_2 + embedding_loss_3) 

#             return output, embedding_loss, torch.tensor(0.0)
#         return output, torch.tensor(0.0), torch.tensor(0.0)
    
#     def embedding_inference(self, input_dict):
#         # a) base output（no grad）
#         with torch.no_grad():
#             token, _ = self.base.input_model_forward(input_dict) # # [seq_len, bs, d]
#             embedding_1 = self.zero_linear_proj_1(self.transformer_layer_1(token)) # # [bs, d]
#             delta_embedding_2= self.zero_linear_proj_2(self.transformer_layer_2(token)) # # [bs, d]
#             delta_embedding_3 = self.base.transformer_forward(token)
#             embedding = embedding_1 + delta_embedding_2 + delta_embedding_3
#         return embedding

# class MultiRateTransformer(nn.Module):
#     def __init__(self, *,
#                 base_cfg: dict,
#                 trainable_cfg: dict,
#                 zero_linear_cfg: dict,
#                 embedding_loss: bool,
#                 ):
#         super().__init__()
# Translated comment.
#         self.base = instantiate(base_cfg)

# Translated comment.
#         self.transformer_layer_1 = instantiate(trainable_cfg)
#         self.transformer_layer_2 = instantiate(trainable_cfg)

# Translated comment.
#         self.zero_linear_proj_1 = instantiate(zero_linear_cfg)
#         self.zero_linear_proj_2 = instantiate(zero_linear_cfg)
#         with torch.no_grad():
#             for m in self.zero_linear_proj_1.modules():
#                 if isinstance(m, nn.Linear):
#                     m.weight.zero_()
#                     if m.bias is not None:
#                         m.bias.zero_()
#             for m in self.zero_linear_proj_2.modules():
#                 if isinstance(m, nn.Linear):
#                     m.weight.zero_()
#                     if m.bias is not None:
#                         m.bias.zero_()
#             # self.zero_linear_proj_1.weight.zero_()
#             # if self.zero_linear_proj_1.bias is not None:
#             #     self.zero_linear_proj_1.bias.zero_()
#             # self.zero_linear_proj_2.weight.zero_()
#             # if self.zero_linear_proj_2.bias is not None:
#             #     self.zero_linear_proj_2.bias.zero_()

#         self.embedding_loss = embedding_loss

#     def forward(self, input_dict):
#         # a) base output（no grad）
#         with torch.no_grad():
#             cat_obs, _ = self.base.input_model_forward(input_dict) # # [seq_len, bs, d]
#             embedding_3 = self.base.transformer_forward(cat_obs)
        
#         embedding_1_prime = self.transformer_layer_1(cat_obs)
#         delta_embedding_1 = self.zero_linear_proj_1(embedding_1_prime.detach()) # # [bs, d]
#         embedding_2_prime = self.transformer_layer_2(cat_obs)
#         delta_embedding_2 = self.zero_linear_proj_2(embedding_2_prime.detach())
        

#         fused = delta_embedding_1 + delta_embedding_2 + embedding_3 # # [bs, d]
#         output = self.base.output_model_forward(fused)

#         if self.embedding_loss:
#             # b) get real embedding
#             input_dict["mimic_target_poses"] = input_dict["mimic_target_poses_2"]
#             embedding_2 = self.embedding_inference(input_dict)
#             input_dict["mimic_target_poses"] = input_dict["mimic_target_poses_1"]
#             embedding_1 = self.embedding_inference(input_dict)

#             # c) get embedding loss
#             embedding_loss_1 = torch.mean(torch.square(embedding_1 - embedding_1_prime))
#             embedding_loss_2 = torch.mean(torch.square(embedding_2 - embedding_2_prime))
#             embedding_loss = embedding_loss_1 + embedding_loss_2

#             return output, embedding_loss, torch.tensor(0.0)
#         return output, torch.tensor(0.0), torch.tensor(0.0)
    
#     def embedding_inference(self, input_dict):
#         # a) base output（no grad）
#         with torch.no_grad():
#             token, _ = self.base.input_model_forward(input_dict) # # [seq_len, bs, d]
#             delta_embedding_1 = self.zero_linear_proj_1(self.transformer_layer_1(token)) # # [bs, d]
#             delta_embedding_2= self.zero_linear_proj_2(self.transformer_layer_2(token)) # # [bs, d]
#             embedding_3 = self.base.transformer_forward(token)
#             embedding = delta_embedding_1 + delta_embedding_2 + embedding_3
#         return embedding
#         # return embedding_3
