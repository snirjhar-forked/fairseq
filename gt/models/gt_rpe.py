# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor


class ContinuousRPE(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dist_embedding = nn.Linear(1, embedding_dim, bias=True)

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):  
        seq_len = input.size(1)
        dtype = torch.long
        if incremental_state is not None:
            pairwise_ids = torch.arange(seq_len-1, -1, -1,
                                device=input.device, dtype=dtype).unsqueeze(0)
        else:
            positions = torch.arange(seq_len, device=input.device, dtype=dtype)
            pairwise_ids = (positions.unsqueeze(-1) - positions).clamp(min=0)
        table_input = torch.log2(1+torch.arange(seq_len,
                                                device=input.device,
                                                dtype=torch.float)).unsqueeze(-1)
        table_input = table_input.to(self.dist_embedding.weight)
        pairwise_table = F.relu(self.dist_embedding(table_input))
        return pairwise_ids, pairwise_table
        

def make_pairwise_ids(inputs, padding_idx: int):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    mask = inputs.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx