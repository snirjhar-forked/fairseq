# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder

from .utils import shuffled_indices


class GTAttention(FairseqIncrementalDecoder):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        max_positions=1024,
        num_windows=1,
        shuffle_type='none',
        shuffle_size=0.25,
        keep_ratio=1.0,
        self_attention=False,
        encoder_decoder_attention=False,
        dictionary=None,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__(dictionary)

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.max_positions_ = max_positions
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.num_windows = num_windows
        self.shuffle_type = shuffle_type
        self.shuffle_size = shuffle_size
        self.keep_ratio = keep_ratio
        assert not self.encoder_decoder_attention, (
            "Currently only self-attention is supported"
        )
        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def _pad_masks(
        self,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if attn_mask is not None:
            shape = attn_mask.size()[:-1] + torch.Size([1])
            attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(shape)], dim=-1)
        if key_padding_mask is not None:
            shape = key_padding_mask.size()[:-1] + torch.Size([1])
            key_padding_mask = torch.cat(
                [
                    key_padding_mask,
                    key_padding_mask.new_zeros(shape),
                ],
                dim=-1,
            )
        return key_padding_mask, attn_mask
        
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
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
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        assert not static_kv, "Currently not supported."
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        tgt_len0 = tgt_len # save for later
        src_len = tgt_len
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]
        
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None
        if self.self_attention:
            key = value = query
        else:
            assert key is not None and value is not None
        
        if self.shuffle_type != "none" and self.training:
            assert not torch.jit.is_scripting(), "Shuffle is not supported in scripting"
            if src_len % self.num_windows != 0 or tgt_len % self.num_windows != 0:
                tgt_padding = (0, 0, 0, 0, 0, self.num_windows - tgt_len % self.num_windows)
                src_paddings = (0, 0, 0, 0, 0, self.num_windows - src_len % self.num_windows)
                if (query is key) and (query is value):
                    query = key = value = F.pad(query, src_paddings)
                elif key is value:
                    query = F.pad(query, tgt_padding)
                    value = key = F.pad(key, src_paddings)
                else:
                    query = F.pad(query, tgt_padding)
                    key = F.pad(key, src_paddings)
                    value = F.pad(value, src_paddings)
                if attn_mask is not None:
                    mask_paddings = (0, self.num_windows - src_len % self.num_windows,
                                     0, self.num_windows - tgt_len % self.num_windows)
                    attn_mask = F.pad(attn_mask, mask_paddings,
                                      value=-float("inf"))
                if key_padding_mask is not None:
                    key_mask_paddings = (0, self.num_windows - src_len % self.num_windows)
                    key_padding_mask = F.pad(key_padding_mask, key_mask_paddings,
                                             value=True)
                tgt_len = query.size(0)
                src_len = key.size(0)
            # indices = torch.randperm(src_len, device=key.device)
            # indices = torch.arange(src_len, device=key.device, dtype=torch.long)
            indices = shuffled_indices(
                shuffle_type=self.shuffle_type,
                shuffle_size=self.shuffle_size,
                input_dim=src_len,
                num_blocks=self.num_windows,
                keep_ratio=self.keep_ratio,
            ).to(key.device)
            if key is value:
                key = value = key.index_select(0, indices)
            else:
                key = key.index_select(0, indices)
                value = value.index_select(0, indices)
            
            if attn_mask is not None:
                assert attn_mask.size(-1) == src_len, "attn_mask size must match input size"
                attn_mask = attn_mask.index_select(-1, indices)
            if key_padding_mask is not None:
                assert key_padding_mask.size(-1) == src_len, "key_padding_mask size must match input size"
                key_padding_mask = key_padding_mask.index_select(-1, indices)
            src_len = key.size(0)
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                assert bsz == _prev_key.size(0)
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                assert bsz == _prev_value.size(0)
                prev_value = _prev_value.view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            key_padding_mask = type(self)._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )    
            
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.num_windows>1 and self.training:
            q = q.view(bsz, self.num_heads, self.num_windows, -1, self.head_dim)
            k = k.view(bsz, self.num_heads, self.num_windows, -1, self.head_dim)
            v = v.view(bsz, self.num_heads, self.num_windows, -1, self.head_dim)
        else:
            q = q.view(bsz, self.num_heads, -1, self.head_dim)
            k = k.view(bsz, self.num_heads, -1, self.head_dim)
            v = v.view(bsz, self.num_heads, -1, self.head_dim)
        attn_weights = q @ k.transpose(-1, -2)

        if attn_mask is not None:
            if self.num_windows>1 and self.training:
                mask_shape = attn_mask.size()
                attn_mask = attn_mask.view(*mask_shape[:-2], self.num_windows,
                                           mask_shape[-2]//self.num_windows,
                                           self.num_windows,
                                           mask_shape[-1]//self.num_windows).transpose(-3,-2)\
                                        .contiguous().view(*mask_shape[:-2], 
                                                            self.num_windows*self.num_windows,
                                                            mask_shape[-2]//self.num_windows,
                                                            mask_shape[-1]//self.num_windows)
                attn_mask = attn_mask[...,::self.num_windows+1,:,:]
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            if self.num_windows>1 and self.training:
                key_padding_mask = key_padding_mask.view(bsz, 1, self.num_windows, 1, -1)
            else:
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            key_padding_mask = key_padding_mask.to(torch.bool)
            attn_weights = attn_weights.masked_fill(key_padding_mask,
                                                    float("-inf"))

        if before_softmax:
            return attn_weights, v

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = attn_probs @ v
        attn = attn.view(bsz*self.num_heads, tgt_len, self.head_dim)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        
        if tgt_len0 != tgt_len:
            attn = attn.narrow(0, 0, tgt_len0)
        
        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)
    
    def extra_repr(self) -> str:
        return f'attention_dropout={self.dropout}, ' \
               f'num_windows={self.num_windows}, shuffle_type={self.shuffle_type}, ' \
               f'shuffle_size={self.shuffle_size}, keep_ratio={self.keep_ratio}'

