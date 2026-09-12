from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from cryodiffdock_pocket.density_guidance import DensityCrop, gaussian_ncc_batch


def effective_sample_size(weights: torch.Tensor) -> float:
    """Return ESS for normalized, non-negative one-dimensional weights."""
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("weights must be a non-empty one-dimensional tensor")
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")
    total = weights.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        raise ValueError("weights must have a positive finite sum")
    normalized = weights / total
    return float(1.0 / normalized.square().sum())


def systematic_resample(
    weights: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw systematic-resampling ancestor indices on CPU.

    The supplied generator is intentionally separate from the reverse-diffusion
    RNG, so selection does not shift the model's subsequent noise stream.
    """
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("weights must be a non-empty one-dimensional tensor")
    normalized = weights.detach().to(dtype=torch.float64, device="cpu")
    normalized = normalized / normalized.sum()
    count = int(normalized.numel())
    offset = float(torch.rand((), generator=generator, dtype=torch.float64)) / count
    positions = offset + torch.arange(count, dtype=torch.float64) / count
    cumulative = torch.cumsum(normalized, dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions, right=False).clamp_max(count - 1)


def _family_statistics(lineages: np.ndarray) -> Tuple[int, float]:
    if lineages.size == 0:
        return 0, 0.0
    _, counts = np.unique(lineages, return_counts=True)
    return int(counts.size), float(counts.max() / counts.sum())


@dataclass
class DensityParticleResampler:
    """Derivative-free Gaussian-NCC particle selection during reverse diffusion.

    This is deliberately a conservative particle-selection experiment, not a
    claim of an exact Bayesian SMC sampler. It never edits coordinates. Leading
    particle slots are protected, while active slots are systematically
    resampled at configured late checkpoints. Descendants diversify through the
    sampler's native independent reverse-SDE noise.
    """

    crop: DensityCrop
    checkpoint_fractions: Tuple[float, ...] = (0.8, 0.9)
    protected_fraction: float = 0.25
    target_ess_fraction: float = 0.7
    resample_ess_threshold_fraction: float = 0.8
    max_beta: float = 10.0
    ncc_sigma_A: float = 1.5
    ncc_mask_radius_A: float = 10.0
    ncc_chunk_size: int = 512
    seed: Optional[int] = None
    _device_cache: dict[str, Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )
    _events: list[Mapping[str, Any]] = field(default_factory=list, init=False, repr=False)
    _lineages: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _resampling_counts: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _protected_count: Optional[int] = field(default=None, init=False, repr=False)
    _num_particles: Optional[int] = field(default=None, init=False, repr=False)
    _generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        checkpoints = tuple(float(value) for value in self.checkpoint_fractions)
        if not checkpoints or any(not 0.0 < value < 1.0 for value in checkpoints):
            raise ValueError("checkpoint fractions must be non-empty and strictly between 0 and 1")
        if len(set(checkpoints)) != len(checkpoints):
            raise ValueError("checkpoint fractions must be unique")
        self.checkpoint_fractions = tuple(sorted(checkpoints))
        if not 0.0 <= self.protected_fraction < 1.0:
            raise ValueError("protected_fraction must be in [0,1)")
        if not 0.0 < self.target_ess_fraction <= 1.0:
            raise ValueError("target_ess_fraction must be in (0,1]")
        if not self.target_ess_fraction <= self.resample_ess_threshold_fraction <= 1.0:
            raise ValueError(
                "resample_ess_threshold_fraction must be between target ESS and 1"
            )
        if self.max_beta < 0.0:
            raise ValueError("max_beta must be non-negative")
        if self.ncc_sigma_A <= 0 or self.ncc_mask_radius_A <= 0 or self.ncc_chunk_size <= 0:
            raise ValueError("NCC sigma, mask radius, and chunk size must be positive")
        self._generator = torch.Generator(device="cpu")
        if self.seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(int(self.seed))

    def _ensure_state(self, count: int) -> None:
        if count < 2:
            raise ValueError("particle resampling requires at least two particles")
        if self._num_particles is None:
            self._num_particles = int(count)
            self._protected_count = int(math.floor(count * self.protected_fraction))
            if count - self._protected_count < 2:
                raise ValueError("protected pool must leave at least two active particles")
            self._lineages = np.arange(count, dtype=np.int64)
            self._resampling_counts = np.zeros(count, dtype=np.int64)
        elif count != self._num_particles:
            raise ValueError(
                f"particle count changed from {self._num_particles} to {count}"
            )

    def checkpoint_steps(self, inference_steps: int) -> Tuple[int, ...]:
        if inference_steps < 2:
            raise ValueError("particle resampling requires at least two inference steps")
        steps = {
            max(1, min(inference_steps - 1, int(math.ceil(fraction * inference_steps - 1e-12))))
            for fraction in self.checkpoint_fractions
        }
        if len(steps) != len(self.checkpoint_fractions):
            raise ValueError(
                "checkpoint fractions collapse to the same discrete diffusion step; "
                "use fewer or more separated checkpoints"
            )
        return tuple(sorted(steps))

    def _ncc_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        if key not in self._device_cache:
            density = torch.as_tensor(self.crop.density, dtype=torch.float32, device=device)
            origin = torch.as_tensor(self.crop.origin, dtype=torch.float32, device=device)
            spacing = torch.as_tensor(self.crop.spacing, dtype=torch.float32, device=device)
            axes = [
                origin[axis]
                + torch.arange(density.shape[axis], dtype=density.dtype, device=device)
                * spacing[axis]
                for axis in range(3)
            ]
            xx, yy, zz = torch.meshgrid(*axes, indexing="ij")
            points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
            center = origin + 0.5 * (
                torch.as_tensor(density.shape, dtype=density.dtype, device=device) - 1.0
            ) * spacing
            mask = torch.linalg.vector_norm(points - center.reshape(1, 3), dim=-1) \
                <= float(self.ncc_mask_radius_A)
            if int(mask.sum()) < 64:
                raise ValueError(f"NCC mask contains only {int(mask.sum())} voxels")
            target = (
                (density - float(self.crop.norm_center)) / float(self.crop.norm_scale)
            ).reshape(-1)[mask]
            self._device_cache[key] = points[mask], target
        return self._device_cache[key]

    def _score_particles(
        self,
        data_list: Sequence[Any],
        device: torch.device,
    ) -> torch.Tensor:
        points, target = self._ncc_tensors(device)
        first_indices = data_list[0]["ligand"].x[:, 0].detach().to(device).long()
        heavy = first_indices != 0
        if not bool(heavy.any()):
            raise ValueError("particle resampling requires at least one ligand heavy atom")
        atomic_weights = torch.where(
            first_indices[heavy] < 118,
            first_indices[heavy] + 1,
            torch.full_like(first_indices[heavy], 6),
        ).to(torch.float32)
        poses = []
        for graph in data_list:
            indices = graph["ligand"].x[:, 0].detach().to(device).long()
            if not torch.equal(indices, first_indices):
                raise ValueError("all particles must have identical ligand atom ordering")
            poses.append(
                graph["ligand"].pos.detach().to(device)[heavy]
                + graph.original_center.detach().to(device).reshape(1, 3)
            )
        with torch.no_grad():
            scores = gaussian_ncc_batch(
                points,
                torch.stack(poses, dim=0),
                atomic_weights,
                target,
                sigma_A=self.ncc_sigma_A,
                chunk_size=self.ncc_chunk_size,
            )
        return scores.detach().to(dtype=torch.float64, device="cpu")

    @staticmethod
    def _weights_for_beta(z_scores: torch.Tensor, beta: float) -> torch.Tensor:
        return torch.softmax(z_scores * float(beta), dim=0)

    def _calibrate_beta(self, z_scores: torch.Tensor) -> Tuple[float, torch.Tensor, float]:
        count = int(z_scores.numel())
        uniform = torch.full((count,), 1.0 / count, dtype=torch.float64)
        if self.max_beta == 0.0:
            return 0.0, uniform, float(count)
        target = self.target_ess_fraction * count
        upper_weights = self._weights_for_beta(z_scores, self.max_beta)
        upper_ess = effective_sample_size(upper_weights)
        if upper_ess > target:
            return float(self.max_beta), upper_weights, upper_ess
        lower = 0.0
        upper = float(self.max_beta)
        for _ in range(50):
            middle = 0.5 * (lower + upper)
            middle_weights = self._weights_for_beta(z_scores, middle)
            if effective_sample_size(middle_weights) > target:
                lower = middle
            else:
                upper = middle
        beta = upper
        weights = self._weights_for_beta(z_scores, beta)
        return beta, weights, effective_sample_size(weights)

    def apply(
        self,
        data_list: Sequence[Any],
        step_index: int,
        inference_steps: int,
        device: torch.device,
    ) -> Tuple[list[Any], Optional[list[int]]]:
        """Optionally resample and return ``(particles, parent_slots)``.

        ``parent_slots`` is ``None`` for an exact no-op. Otherwise it maps each
        new slot to its parent slot immediately before this checkpoint.
        """
        self._ensure_state(len(data_list))
        step_number = int(step_index) + 1
        if step_number not in self.checkpoint_steps(inference_steps):
            return list(data_list), None

        assert self._protected_count is not None
        assert self._lineages is not None
        assert self._resampling_counts is not None
        protected = self._protected_count
        active_count = len(data_list) - protected
        unique_before, max_family_before = _family_statistics(self._lineages[protected:])
        if self.max_beta == 0.0:
            # A genuine execution-path no-op: do not launch NCC kernels or
            # allocate additional GPU tensors. Even RNG-free CUDA work can
            # perturb a numerically sensitive downstream trajectory.
            self._events.append({
                "step": step_number,
                "progress": float(step_number / inference_steps),
                "num_particles": len(data_list),
                "num_protected": protected,
                "num_active": active_count,
                "active_ncc_mean": None,
                "active_ncc_std": None,
                "active_ncc_min": None,
                "active_ncc_max": None,
                "unique_active_ancestors_before": unique_before,
                "max_active_family_fraction_before": max_family_before,
                "resampled": False,
                "reason": "max_beta_zero_exact_execution_noop",
                "beta": 0.0,
                "ess": float(active_count),
                "ess_fraction": 1.0,
            })
            return list(data_list), None

        scores = self._score_particles(data_list, device)
        active_scores = scores[protected:]
        if not bool(torch.isfinite(active_scores).all()):
            raise ValueError("particle NCC scores contain non-finite values")
        score_std = float(active_scores.std(unbiased=False))
        base_event = {
            "step": step_number,
            "progress": float(step_number / inference_steps),
            "num_particles": len(data_list),
            "num_protected": protected,
            "num_active": active_count,
            "active_ncc_mean": float(active_scores.mean()),
            "active_ncc_std": score_std,
            "active_ncc_min": float(active_scores.min()),
            "active_ncc_max": float(active_scores.max()),
            "unique_active_ancestors_before": unique_before,
            "max_active_family_fraction_before": max_family_before,
        }
        if not math.isfinite(score_std) or score_std <= 1e-12:
            self._events.append({
                **base_event,
                "resampled": False,
                "reason": "ncc_scores_indistinguishable",
                "beta": 0.0,
                "ess": float(active_count),
                "ess_fraction": 1.0,
            })
            return list(data_list), None

        z_scores = (active_scores - active_scores.mean()) / active_scores.std(unbiased=False)
        beta, weights, ess = self._calibrate_beta(z_scores)
        ess_fraction = ess / active_count
        if ess_fraction > self.resample_ess_threshold_fraction:
            self._events.append({
                **base_event,
                "resampled": False,
                "reason": "ess_above_resampling_threshold",
                "beta": beta,
                "ess": ess,
                "ess_fraction": ess_fraction,
                "max_weight": float(weights.max()),
            })
            return list(data_list), None

        local_parents = systematic_resample(weights, self._generator).numpy()
        parent_slots = list(range(protected)) + [
            protected + int(parent) for parent in local_parents
        ]
        new_data = list(data_list[:protected]) + [
            copy.deepcopy(data_list[parent]) for parent in parent_slots[protected:]
        ]
        old_lineages = self._lineages.copy()
        old_counts = self._resampling_counts.copy()
        self._lineages = old_lineages[np.asarray(parent_slots, dtype=np.int64)]
        self._resampling_counts = old_counts[np.asarray(parent_slots, dtype=np.int64)]
        self._resampling_counts[protected:] += 1
        unique_after, max_family_after = _family_statistics(self._lineages[protected:])
        selected_counts = np.bincount(local_parents, minlength=active_count)
        self._events.append({
            **base_event,
            "resampled": True,
            "reason": "ess_triggered",
            "beta": beta,
            "ess": ess,
            "ess_fraction": ess_fraction,
            "max_weight": float(weights.max()),
            "unique_active_ancestors_after": unique_after,
            "max_active_family_fraction_after": max_family_after,
            "num_active_parents_selected": int(np.count_nonzero(selected_counts)),
            "max_parent_offspring": int(selected_counts.max()),
            "parent_slots": parent_slots,
            "active_parent_offspring_counts": selected_counts.astype(int).tolist(),
        })
        return new_data, parent_slots

    def final_particle_metadata(self) -> list[Mapping[str, Any]]:
        if self._lineages is None or self._resampling_counts is None:
            return []
        protected = int(self._protected_count or 0)
        return [
            {
                "particle_index": int(slot),
                "initial_ancestor_index": int(self._lineages[slot]),
                "protected": bool(slot < protected),
                "resampling_events_in_lineage": int(self._resampling_counts[slot]),
            }
            for slot in range(len(self._lineages))
        ]

    def summary(self) -> Mapping[str, Any]:
        final = self.final_particle_metadata()
        lineages = np.asarray(
            [row["initial_ancestor_index"] for row in final], dtype=np.int64
        )
        protected = int(self._protected_count or 0)
        unique_all, max_family_all = _family_statistics(lineages)
        unique_active, max_family_active = _family_statistics(lineages[protected:])
        return {
            "enabled": True,
            "mechanism": "coordinate_preserving_density_weighted_systematic_resampling",
            "crop_path": self.crop.path,
            "checkpoint_fractions": list(self.checkpoint_fractions),
            "protected_fraction": float(self.protected_fraction),
            "target_ess_fraction": float(self.target_ess_fraction),
            "resample_ess_threshold_fraction": float(
                self.resample_ess_threshold_fraction
            ),
            "max_beta": float(self.max_beta),
            "ncc_sigma_A": float(self.ncc_sigma_A),
            "ncc_mask_radius_A": float(self.ncc_mask_radius_A),
            "ncc_chunk_size": int(self.ncc_chunk_size),
            "num_particles": int(self._num_particles or 0),
            "num_protected": protected,
            "num_active": int((self._num_particles or 0) - protected),
            "num_checkpoint_events": len(self._events),
            "num_resampling_events": int(sum(bool(row["resampled"]) for row in self._events)),
            "final_unique_ancestors": unique_all,
            "final_max_family_fraction": max_family_all,
            "final_unique_active_ancestors": unique_active,
            "final_max_active_family_fraction": max_family_active,
            "events": list(self._events),
            "final_particles": final,
        }

    def write_audit(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.summary(), indent=2) + "\n")
