# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils
from fairseq.lrk_utils import LrkLinear, weight_decomposition, process_batch_grad


    #def normal_forward(self, x):
    #    return self.full_linear(x)




def multi_head_attention_forward(query,                           # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 attn_embed_dim,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 bias_k,                          # type: Optional[Tensor]
                                 bias_v,                          # type: Optional[Tensor]
                                 ma_module,                       # multihead_attention module
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 rel_pos_bias=None                # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    tgt_len, bsz, embed_dim = query.size()
    assert key.size() == value.size()

    head_dim = attn_embed_dim // num_heads

    scaling = float(head_dim) ** -0.5

    args = ma_module.args


    if(ma_module.is_training):
        lrk_in_left, lrk_in_right, lrk_out_left, lrk_out_right = ma_module.in_proj_left, ma_module.in_proj_right, ma_module.out_proj_left, ma_module.out_proj_right

        if(ma_module.need_decompose):
            ma_module.in_cached, ma_module.out_cached = in_proj_weight.data, out_proj_weight.data
            in_left_data, in_right_data, in_residual_data = weight_decomposition(in_proj_weight.data, rank=args.rank)
            out_left_data, out_right_data, out_residual_data = weight_decomposition(out_proj_weight.data, rank=args.rank)
            ma_module.replace_param(in_left_data, in_right_data, in_residual_data, out_left_data, out_right_data, out_residual_data)
            ma_module.need_decompose = False

        residual_acti0 = F.linear(query, in_proj_weight, in_proj_bias)
        lrk_acti0 = lrk_in_right(query)
        lrk_acti0 = lrk_in_left(lrk_acti0)
        acti = residual_acti0+lrk_acti0
    else:
        acti = F.linear(query, in_proj_weight, in_proj_bias)

    q, k, v = acti.chunk(3, dim=-1)

    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask

    if rel_pos_bias is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights += rel_pos_bias
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, attn_embed_dim)

    if(ma_module.is_training):
        residual_acti1 = F.linear(attn_output, out_proj_weight, out_proj_bias)
        lrk_acti1 = lrk_out_right(attn_output)
        lrk_acti1 = lrk_out_left(lrk_acti1)

        attn_output = residual_acti1+lrk_acti1
    else:
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

class MultiheadAttention(nn.Module):
    """MultiHeadAttention
    """

    def __init__(self, embed_dim, num_heads, args, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        
        self.head_dim = embed_dim // num_heads
        self.attn_embed_dim = embed_dim

        
        self.in_proj_weight = Parameter(torch.Tensor(3 * self.attn_embed_dim, embed_dim))

        self.in_proj_left = LrkLinear(args.rank, 3 * self.attn_embed_dim)
        self.in_proj_right = LrkLinear(embed_dim, args.rank)

        self.args = args
        
        

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * self.attn_embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(self.attn_embed_dim, embed_dim, bias=bias)

        self.out_proj_left =  LrkLinear(args.rank, embed_dim)
        self.out_proj_right = LrkLinear(self.attn_embed_dim, args.rank)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, self.attn_embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.attn_embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.is_training = None
        self.need_decompose = True

    def reset_parameters(self):
        # Note: these initilaztion will be overrided in `init_bert_params`, if using BERT
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def replace_param(self, in_left, in_right, in_residual, out_left, out_right, out_residual):
        self.in_proj_left.weight.data = in_left
        self.in_proj_right.weight.data = in_right
        self.in_proj_weight.data = in_residual


        self.out_proj_left.weight.data = out_left
        self.out_proj_right.weight.data = out_right
        self.out_proj.weight.data = out_residual

    def use_batch_grad(self, scale=None):
        if(self.in_proj_left.weight.grad == None):
            self.in_proj_left.weight.grad = process_batch_grad(self.in_proj_left.weight.batch_grad, scale=scale)
            self.in_proj_right.weight.grad = process_batch_grad(self.in_proj_right.weight.batch_grad, scale=scale)
            self.out_proj_left.weight.grad = process_batch_grad(self.out_proj_left.weight.batch_grad, scale=scale)
            self.out_proj_right.weight.grad = process_batch_grad(self.out_proj_right.weight.batch_grad, scale=scale)

        else:
            self.in_proj_left.weight.grad += process_batch_grad(self.in_proj_left.weight.batch_grad, scale=scale)
            self.in_proj_right.weight.grad += process_batch_grad(self.in_proj_right.weight.batch_grad, scale=scale)
            self.out_proj_left.weight.grad += process_batch_grad(self.out_proj_left.weight.batch_grad, scale=scale)
            self.out_proj_right.weight.grad += process_batch_grad(self.out_proj_right.weight.batch_grad, scale=scale)            

    def _assign_full_grad(self, left, right, host):
        left_w, left_g = left.data, left.grad
        right_w, right_g = right.data, right.grad

        left_g_right_w = torch.matmul(left_g, right_w)
        m1 = left_g_right_w + torch.matmul(left_w, right_g)
        m2 = torch.matmul(left_w, torch.matmul(left_w.T, left_g_right_w))

        host.grad = m1 + m2


    def assign_full_grad(self):


        self._assign_full_grad(self.in_proj_left.weight, self.in_proj_right.weight, self.in_proj_weight)
        self._assign_full_grad(self.out_proj_left.weight, self.out_proj_right.weight, self.out_proj.weight)

        self.in_proj_left.weight.grad = None
        self.in_proj_right.weight.grad = None
        self.out_proj_left.weight.grad = None
        self.out_proj_right.weight.grad = None

    def restore_param(self):
        self.in_proj_weight.data = self.in_cached
        self.out_proj.weight.data = self.out_cached

        self.need_decompose = True

    def forward(
        self,
        query, key, value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        rel_pos_bias=None,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        # rel_pos_bias = None
        return multi_head_attention_forward(query, key, value, 
                                            attn_embed_dim=self.attn_embed_dim,
                                            num_heads=self.num_heads,
                                            in_proj_weight=self.in_proj_weight,
                                            in_proj_bias=self.in_proj_bias, 
                                            bias_k=self.bias_k, 
                                            bias_v=self.bias_v,
                                            ma_module=self,
                                            add_zero_attn=self.add_zero_attn,
                                            dropout_p=self.dropout,
                                            out_proj_weight=self.out_proj.weight,
                                            out_proj_bias=self.out_proj.bias,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            need_weights=need_weights,
                                            attn_mask=attn_mask,
                                            rel_pos_bias=rel_pos_bias)
