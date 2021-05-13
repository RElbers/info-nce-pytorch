import torch
import torch.nn.functional as F
from torch import nn


class InfoNCE(nn.Module):
    """
    PyTorch Module for the InfoNCE loss.
    """

    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, query, positive_keys, negative_keys):
        return info_nce(query, positive_keys, negative_keys, tau=self.tau)


def info_nce(query, positive_keys, negative_keys, tau=1.0):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar samples to be close and those of different samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        query: NxD Tensor with query samples (e.g. embeddings of the input).
        positive_keys: NxD Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys: MxD Tensor with negative samples (e.g. embeddings of other inputs).
        tau: Temperature term. Logits are divided by tau.

    Returns:
         Value of the InfoNCE Loss.
    """

    # Inputs should all have 2 dimensions.
    assert 2 == query.dim() == positive_keys.dim() == negative_keys.dim()

    # Each query sample is paired with exactly one positive key sample.
    assert len(query) == len(positive_keys)

    # Embedding vectors should have same dimensionality.
    assert query.shape[1] == positive_keys.shape[1] == negative_keys.shape[1]

    # Normalize to unit vectors
    query = F.normalize(query, dim=1)
    positive_keys = F.normalize(positive_keys, dim=1)
    negative_keys = F.normalize(negative_keys, dim=1)

    # Dot product between positive pairs
    positive_logits = torch.sum(query * positive_keys, dim=1, keepdim=True)

    # Dot product between all positive-negative combinations
    negative_logits = query @ negative_keys.transpose(-2, -1)

    # First index in last dimension is for the positive samples
    logits = torch.cat([positive_logits, negative_logits], dim=1) / tau
    labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

    return F.cross_entropy(logits, labels, reduction='mean')
