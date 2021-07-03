# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)

from fairseq.lrk_utils import LrkLinear, weight_decomposition

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        args,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        export: bool = False,
        # new added
        encoder_normalize_before: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters

        self.args = args
        self.alpha = args.alpha
        #print('build encoder layer for roberta, alpha value: %.3f'%self.alpha)

        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            args=self.args,
            dropout=attention_dropout,
            bias=True
        )

        # new added
        self.normalize_before = encoder_normalize_before
        

        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)


        self.fc1_right = LrkLinear(self.embedding_dim, args.rank)
        self.fc1_left = LrkLinear(args.rank, ffn_embedding_dim)

        self.fc2_right = LrkLinear(ffn_embedding_dim, args.rank)
        self.fc2_left = LrkLinear(args.rank, self.embedding_dim)
        
        self.is_training = None

        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.need_decompose = True

    def replace_param(self, fc1_left, fc1_right, fc1_residual, fc2_left, fc2_right, fc2_residual):
        self.fc1_left.weight.data = fc1_left
        self.fc1_right.weight.data = fc1_right
        self.fc1.weight.data = fc1_residual

        self.fc2_left.weight.data = fc2_left
        self.fc2_right.weight.data = fc2_right
        self.fc2.weight.data = fc2_residual

    def restore_param(self):
        self.fc1.weight.data = self.fc1_cached
        self.fc2.weight.data = self.fc2_cached
        self.need_decompose = True
            
    def _assign_full_grad(self, left, right, host):
        left_w, left_g = left.data, left.grad
        right_w, right_g = right.data, right.grad

        left_g_right_w = torch.matmul(left_g, right_w)
        m1 = left_g_right_w + torch.matmul(left_w, right_g)
        m2 = torch.matmul(left_w, torch.matmul(left_w.T, left_g_right_w))

        host.grad = m1 + m2

    def assign_full_grad(self):
        ## assign grad for
        # left_g_right_w = torch.matmul(self.left_layer.weight.grad, self.right_layer.weight.data)

        # m1 = left_g_right_w + torch.matmul(self.left_layer.weight.data, self.right_layer.weight.grad)
        # mg = torch.matmul(self.left_layer.weight.grad, self.right_layer.weight.grad)
        # m2 = torch.matmul(self.left_layer.weight.data, torch.matmul(self.left_layer.weight.data.T, left_g_right_w))

        # grad = m1 - m2
        self._assign_full_grad(self.fc1_left.weight, self.fc1_right.weight, self.fc1.weight)
        self._assign_full_grad(self.fc2_left.weight, self.fc2_right.weight, self.fc2.weight)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        rel_pos_bias: torch.Tensor = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        # new added
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            rel_pos_bias=rel_pos_bias,
        )

        if(self.is_training and self.need_decompose):
            self.fc1_cached, self.fc2_cached = self.fc1.weight.data, self.fc2.weight.data
            fc1_left_data, fc1_right_data, fc1_residual = weight_decomposition(self.fc1.weight.data, self.args.rank)
            fc2_left_data, fc2_right_data, fc2_residual = weight_decomposition(self.fc2.weight.data, self.args.rank)
            self.need_decompose = False

            self.replace_param(fc1_left_data, fc1_right_data, fc1_residual, fc2_left_data, fc2_right_data, fc2_residual)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        # change 
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        # x = self.self_attn_layer_norm(x)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)

        #normal_x = F.linear(x, self.fc1_cached, self.fc1.bias.data)
        #normal_x = self.fc1(x)
        if(self.is_training):
            lrk_x = self.fc1_right(x)
            lrk_x = self.fc1_left(lrk_x)
            residual_x = self.fc1(x)
            x = lrk_x + residual_x
            #print('fc1, normal_x norm: ', normal_x.norm().item(), 'lrk reparam norm: ', x.norm().item())
        else:
            x = self.fc1(x)

        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)

        #normal_x = F.linear(x, self.fc2_cached, self.fc2.bias.data)
        #normal_x = self.fc2(x)
        if(self.is_training):
            lrk_x = self.fc2_right(x)
            lrk_x = self.fc2_left(lrk_x)
            residual_x = self.fc2(x)
            x = lrk_x + residual_x
            #print('fc2, normal_x norm: ', normal_x.norm().item(), 'lrk reparam norm: ', x.norm().item())
        else:
            x = self.fc2(x)


        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        # x = self.final_layer_norm(x)
        
        return x
    
    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        if(self.args.noln):
            return x
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
