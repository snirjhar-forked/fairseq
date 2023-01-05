import numpy as np
import torch
from typing import Optional, Union, Tuple

from .sort_1d import gindex_1d, hgindex_1d, uindex_1d

def index_rearrange(coords, shift):
    if shift == 0:
        return coords
    coords = np.roll(coords, shift, axis=0)
    coords = (coords + shift) % coords.shape[0]
    return coords


def shuffled_indices(shuffle_type: str,
                     shuffle_size: float,
                     input_dim: int,
                     num_blocks: int,
                     shift: int = 0,
                     keep_ratio: float = 1.0,
                     ) -> torch.Tensor:
    shuffle_length = round(input_dim * shuffle_size)
    if shuffle_type == 'randperm':
        indices = np.random.permutation(input_dim)
    elif shuffle_type == 'gaussian':
        indices = gindex_1d(input_dim, shuffle_length)
        indices = index_rearrange(indices, shift)
    elif shuffle_type == 'half_gaussian':
        indices = hgindex_1d(input_dim, num_blocks, shuffle_length)
        indices = index_rearrange(indices, shift)
    elif shuffle_type == 'uniform':
        indices = uindex_1d(input_dim, shuffle_length)
        indices = index_rearrange(indices, shift)
    else:
        raise ValueError(f'Unknown shuffle_type: {shuffle_type}')
    
    if keep_ratio >= 1.0:
        return torch.from_numpy(indices)
    
    if num_blocks == 1:
        return torch.from_numpy(indices[:round(keep_ratio*input_dim)])
    
    block_shape = (num_blocks, input_dim // num_blocks)
    indices = np.reshape(indices, block_shape)
    
    reshuffle_indices = np.argsort(np.random.rand(*block_shape), axis=1)
    reshuffle_indices = reshuffle_indices[:,:round(keep_ratio*
                                                    block_shape[1])]
    indices = np.take_along_axis(indices, reshuffle_indices, axis=1)
    indices = indices.ravel()
    return torch.from_numpy(indices)



