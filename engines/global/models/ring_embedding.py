"""Ring Feature Embedding for Cryo-EM density injection.

This module provides the RingFeatureEmbedding layer that maps per-atom/residue
ring features (max EM density at radii [4,5,6,7] Å) to the model's hidden dimension.

The embedding is designed for additive injection into pretrained DiffDock models:
- Final layer is zero-initialized, so output is near-zero at initialization
- This preserves pretrained model behavior at the start of finetuning
- Ring signal is gradually learned during training
"""

import torch
import torch.nn as nn


def normalize_ring_per_complex(ring_features, ring_vectors, batch):
    """Per-complex normalization of ring features and vectors.

    Ring scalars: (x - mean) / (std + eps) across all atoms and radii per complex.
    Ring vectors: x / (norm_std + eps) per complex, preserving direction.

    Args:
        ring_features: (N, ring_dim) scalar ring features, or None
        ring_vectors: (N, ring_dim, 3) vector ring features, or None
        batch: (N,) complex index per node (from PyG batching)

    Returns:
        Normalized (ring_features, ring_vectors) — same shapes, None passthrough.
    """
    if ring_features is not None:
        rf = ring_features.clone()
        for i in batch.unique():
            mask = batch == i
            rf_i = rf[mask]
            std = rf_i.std()
            if std > 1e-6:
                rf[mask] = (rf_i - rf_i.mean()) / std
        ring_features = rf

    if ring_vectors is not None:
        rv = ring_vectors.clone()
        for i in batch.unique():
            mask = batch == i
            rv_i = rv[mask]
            norms = rv_i.norm(dim=-1)  # (N_i, ring_dim)
            std = norms.std()
            if std > 1e-6:
                rv[mask] = rv_i / std
        ring_vectors = rv

    return ring_features, ring_vectors


class RingFeatureEmbedding(nn.Module):
    """Embed ring features (per-atom max EM density at multiple radii) into model dimension.

    Architecture: Linear(ring_dim, emb_dim) -> ReLU -> Linear(emb_dim, emb_dim)

    The final layer is initialized to near-zero so that at initialization,
    the embedding output is approximately zero. This allows additive injection
    into pretrained models without disrupting their initial behavior.

    Args:
        ring_dim: Number of ring feature channels (default: 4 for radii [4,5,6,7] Å)
        emb_dim: Output embedding dimension (should match model's ns parameter)
        zero_init_std: Standard deviation for final layer weight initialization (default: 1e-4)
    """

    def __init__(self, ring_dim: int = 4, emb_dim: int = 48, zero_init_std: float = 1e-4):
        super().__init__()
        self.ring_dim = ring_dim
        self.emb_dim = emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(ring_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Zero-initialize the final layer for additive injection
        # This ensures the model output is identical to pretrained at initialization
        final_layer = self.mlp[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=zero_init_std)
        nn.init.zeros_(final_layer.bias)

    def forward(self, ring_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ring_features: Tensor of shape (N, ring_dim) where N is number of atoms/residues
                          Values are max EM density sampled at each radius

        Returns:
            Tensor of shape (N, emb_dim) - embedding to add to node features
        """
        return self.mlp(ring_features)

    def extra_repr(self) -> str:
        return f'ring_dim={self.ring_dim}, emb_dim={self.emb_dim}'


class RingVectorEmbedding(nn.Module):
    """Equivariant l=1 embedding for directional ring features.

    Applies scalar linear weights to combine radius channels into nv output vectors.
    No bias (would break equivariance). Zero-initialized for additive injection.

    Args:
        ring_dim: Number of ring feature channels (default: 4 for radii [4,5,6,7] Å)
        nv: Number of output vector channels (should match model's nv parameter)
        zero_init_std: Standard deviation for weight initialization (default: 1e-4)
    """

    def __init__(self, ring_dim: int = 4, nv: int = 10, zero_init_std: float = 1e-4):
        super().__init__()
        self.ring_dim = ring_dim
        self.nv = nv
        # Linear on radius dimension (scalar weights, equivariant)
        self.linear = nn.Linear(ring_dim, nv, bias=False)
        nn.init.normal_(self.linear.weight, mean=0.0, std=zero_init_std)

    def forward(self, ring_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ring_vectors: (N, ring_dim, 3) direction vectors per radius

        Returns:
            (N, nv*3) flattened l=1 embedding
        """
        # Apply linear on radius dim, preserving vector structure
        x = ring_vectors.transpose(1, 2)  # (N, 3, ring_dim)
        x = self.linear(x)                # (N, 3, nv)
        x = x.transpose(1, 2)             # (N, nv, 3)
        return x.reshape(x.shape[0], -1)  # (N, nv*3)

    def extra_repr(self) -> str:
        return f'ring_dim={self.ring_dim}, nv={self.nv}'
