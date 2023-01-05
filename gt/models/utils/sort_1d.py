import numpy as np
import numba as nb


@nb.njit
def uindex_1d(length, half_window):
    indices = np.arange(length, dtype=np.int64)
    noise = np.random.rand(length) * half_window * 2
    shuffled_indices = np.argsort(indices + noise)
    return shuffled_indices
    
    
@nb.njit
def gindex_1d(length, sigma):
    indices = np.arange(length, dtype=np.int64)
    noise = np.random.randn(length) * sigma
    shuffled_indices = np.argsort(indices + noise)
    return shuffled_indices
    

@nb.njit
def hgindex_1d(length, num_blocks, sigma):
    block_length = length // num_blocks
    indices = np.arange(length, dtype=np.int64)
    shuffled_indices = np.empty(length, dtype=np.int64)
    for i in range(num_blocks):
        upto = min(length, (i+1)*block_length)
        frm = max(upto-block_length,0)
        noise = np.random.randn(upto) * sigma
        sinds = np.argsort(indices[:upto] + noise)
        shuffled_indices[frm:upto] = sinds[frm:]
    return shuffled_indices



if __name__ == '__main__':
    print(hgindex_1d(32, 4, 4))
    print(gindex_1d(32, 4))