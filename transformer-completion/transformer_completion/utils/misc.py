from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def pad_stack(tensor_list: List[Tensor]):
    """
    pad each tensor on the input to the max value in shape[1] and 
    concatenate them in a single tensor.
    Input:
        list of tensors [Ni,Pi]
    Output:
        tensor [sum(Ni),max(Pi)]
    """
    _max = max([t.shape[1] for t in tensor_list])
    # valid = [F.pad(torch.ones_like(t), (0, _max - t.shape[1])) for t in tensor_list]
    batched = torch.cat([F.pad(t, (0, _max - t.shape[1])) for t in tensor_list])
    return batched


def sample_points(masks, masks_ids, n_pts, n_samples):
    # select n_pts per mask to focus on instances
    # plus random points up to n_samples
    sampled = []
    for ids, mm in zip(masks_ids, masks):
        m_idx = torch.cat(
            [
                id[torch.randperm(n_pts)[:n_pts]] if id.shape[0] > n_pts else id
                for id in ids
            ]
        )
        r_idx = torch.randint(mm.shape[1], [n_samples - m_idx.shape[0]]).to(m_idx)
        idx = torch.cat((m_idx, r_idx))
        sampled.append(idx)
    return sampled


def get_points(n_masks, idx, source):
    _masks = n_masks.copy()
    _masks.insert(0, 0)
    _masks = np.cumsum(np.array(_masks))
    points = []
    for i, p in enumerate(idx):
        pts = source[_masks[i] : _masks[i + 1]][:, p]
        points.append(pts)
    return torch.cat(points)
