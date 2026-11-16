"""Run Local diffusion with a restrained ligand fragment and free-atom map guidance."""
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from scipy.spatial import cKDTree

import inference
from utils import sampling
from cryodiffdock_pocket.physics import atom_radii, clash_metrics, read_protein_heavy_atoms
from cryodiffdock_pocket.density_ranking import load_cartesian_density_crop, score_gaussian_density
from cryodiffdock_pocket.density_guidance import (
    compose_score, guidance_weight, project_torsion_scores, trilinear_density,
)


def torsion_graph(graph):
    ligand = graph["ligand"]
    edges = graph["ligand", "ligand"].edge_index.T[ligand.edge_mask]
    masks = torch.as_tensor(ligand.mask_rotate, dtype=torch.bool).reshape(
        len(edges), ligand.pos.shape[0]
    )
    return edges, masks


def allowed_torsions(edges, masks, anchors):
    # An anchor on the rotation axis stays fixed. A torsion may also rotate
    # the entire anchor as a rigid unit, which the global projection undoes.
    on_axis = (edges[:, :, None] == anchors).any(dim=1)
    moving = masks[:, anchors]
    return ((moving | on_axis).all(dim=1) | (~moving | on_axis).all(dim=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-config", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    args = inference._get_parser().parse_args(remaining)
    config = json.loads(options.anchor_config.read_text())
    molecule = Chem.MolFromMolFile(config["ligand_path"])
    reference = Chem.MolFromMolFile(config["reference_path"])
    anchors = torch.as_tensor(config["indices"], dtype=torch.long)
    fixed = reference.GetConformer().GetPositions()
    fixed_center = fixed.mean(axis=0)
    elements = np.array([atom.GetAtomicNum() - 1 for atom in molecule.GetAtoms()])
    bonds = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
             for bond in molecule.GetBonds()}
    protein, protein_elements = read_protein_heavy_atoms(config["protein_path"])
    protein_tree = cKDTree(protein)
    receptor_radii = atom_radii(protein_elements)
    ligand_radii = atom_radii([atom.GetSymbol() for atom in molecule.GetAtoms()])
    native_randomize = inference.randomize_position
    native_modify = sampling.modify_conformer

    def heavy_atoms(graph):
        return torch.where(graph["ligand"].x[:, 0].detach().cpu() != 0)[0]

    def project(graph):
        positions = graph["ligand"].pos
        origin = graph.original_center.detach().cpu().numpy().reshape(3)
        xyz = positions.detach().cpu().numpy().astype(np.float64) + origin
        anchor_xyz = xyz[heavy_atoms(graph)[anchors].numpy()]
        center = anchor_xyz.mean(axis=0)
        u, _, vt = np.linalg.svd((anchor_xyz - center).T @ (fixed - fixed_center))
        u[:, -1] *= np.linalg.det(u @ vt)
        aligned = (xyz - center) @ (u @ vt) + fixed_center - origin
        graph["ligand"].pos = torch.as_tensor(aligned, device=positions.device, dtype=positions.dtype)
        return graph

    def modify(graph, translation, rotation, torsion, pivot=None):
        if torsion is not None:
            edges, masks = torsion_graph(graph)
            allowed = allowed_torsions(edges.cpu(), masks.cpu(), heavy_atoms(graph)[anchors]).numpy()
            torsion = np.where(allowed, torsion, 0.0)
        return project(native_modify(graph, torch.zeros_like(translation),
                                     torch.zeros_like(rotation), torsion, pivot=pivot))

    def randomize(graphs, no_torsion, *rest, **kwargs):
        native_randomize(graphs, True, *rest, **kwargs)
        for graph in graphs:
            heavy = heavy_atoms(graph).tolist()
            index = {atom: i for i, atom in enumerate(heavy)}
            actual = {tuple(sorted((index[u], index[v])))
                      for u, v in graph["ligand", "ligand"].edge_index.T.tolist()
                      if u in index and v in index}
            if not np.array_equal(graph["ligand"].x[heavy, 0].cpu().numpy(), elements) or actual != bonds:
                raise ValueError("The model's heavy-atom order or connectivity differs from the ligand SDF.")
            edges, masks = torsion_graph(graph)
            allowed = allowed_torsions(edges.cpu(), masks.cpu(), heavy_atoms(graph)[anchors])
            if no_torsion or not allowed.any():
                raise ValueError("There are no rotatable bonds that preserve the selected fragment.")
            pos = graph["ligand"].pos
            modify(graph, pos.new_zeros((1, 3)), pos.new_zeros(3),
                   np.random.uniform(-np.pi, np.pi, size=len(edges)))

    class FreeAtomGuidance(inference.LigandDensityGuidance):
        def apply(self, data_list, translation_score, rotation_score, torsion_score,
                  tr_sigma, rot_sigma, tor_sigma, step_index, inference_steps, device):
            ramp = guidance_weight((step_index + 1) / inference_steps, self.start_fraction)
            if not ramp or torsion_score is None:
                return translation_score, rotation_score, torsion_score
            density, origin, spacing = self._tensors(device)
            outputs = []
            offset = 0
            map_norms = []
            steric_norms = []
            for graph in data_list:
                edges, masks = torsion_graph(graph)
                coordinates = (graph["ligand"].pos + graph.original_center).to(device)
                heavy = heavy_atoms(graph).to(device)
                anchored = heavy[anchors.to(device)]
                free = heavy[~torch.isin(heavy, anchored)]
                with torch.enable_grad():
                    coordinates = coordinates.detach().requires_grad_(True)
                    sampled, _ = trilinear_density(density, origin, spacing,
                                                  coordinates[free], 0.0)
                    gradient = torch.autograd.grad(sampled.mean(), coordinates)[0]
                # For bonds that rotate the anchor side, projection is
                # equivalent to rotating the opposite side in reverse.
                edges, masks = edges.to(device), masks.to(device)
                moving_anchor = (masks[:, anchored]
                                 & ~(edges[:, :, None] == anchored).any(dim=1)).any(dim=1)
                relative_masks = torch.where(moving_anchor[:, None], ~masks, masks)
                scores = project_torsion_scores(coordinates.detach(), gradient, edges, relative_masks)
                scores = torch.where(moving_anchor, -scores, scores)
                allowed = allowed_torsions(edges, masks, anchored)
                scores *= allowed
                positions = coordinates.detach().cpu().numpy()
                repulsion = np.zeros_like(positions)
                free_positions = free.cpu().tolist()
                radius_by_atom = dict(zip(heavy.cpu().tolist(), ligand_radii))
                for atom, neighbors in zip(free_positions, protein_tree.query_ball_point(
                    positions[free_positions], 4.5
                )):
                    if not neighbors:
                        continue
                    delta = positions[atom] - protein[neighbors]
                    distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-8)
                    overlap = np.maximum(radius_by_atom[atom] + receptor_radii[neighbors] - 0.8 - distance, 0)
                    repulsion[atom] = ((overlap / distance)[:, None] * delta).sum(axis=0)
                steric = project_torsion_scores(coordinates.detach(), torch.from_numpy(repulsion).to(device),
                                               edges, relative_masks)
                steric = torch.where(moving_anchor, -steric, steric) * allowed
                map_norms.append(torch.linalg.vector_norm(scores).item())
                steric_norms.append(torch.linalg.vector_norm(steric).item())
                learned = torsion_score[offset:offset + len(edges)]
                combined, _ = compose_score(learned[None, :], scores.to(learned)[None, :],
                                            ramp * self.lambda_torsion, self.clip_ratio)
                combined, _ = compose_score(combined, steric.to(learned)[None, :],
                                            ramp * 0.5, self.clip_ratio)
                outputs.append(combined.flatten())
                offset += len(edges)
            if offset != torsion_score.numel():
                raise ValueError("The predicted torsion count does not match the ligand graph.")
            self._audit_steps.append({
                "step": step_index + 1, "free_atom_map_gradient_norm": float(np.mean(map_norms)),
                "steric_torsion_gradient_norm": float(np.mean(steric_norms)),
            })
            return translation_score, rotation_score, torch.cat(outputs)

    inference.randomize_position = randomize
    sampling.modify_conformer = modify
    inference.LigandDensityGuidance = FreeAtomGuidance
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    with torch.no_grad():
        inference.main(args)
    rank_candidates(Path(args.out_dir), config, molecule, anchors.numpy(), fixed)



def rank_candidates(directory, config, ligand, anchors, fixed):
    crop = load_cartesian_density_crop(config["crop_path"])
    mask = np.ones(crop.density.shape, dtype=bool)
    axes = np.ogrid[tuple(slice(0, n) for n in mask.shape)]
    for position in fixed:
        distance_sq = sum((crop.origin[i] + crop.spacing[i] * axes[i] - position[i]) ** 2
                          for i in range(3))
        mask &= distance_sq > 4.0
    if not mask.any():
        raise ValueError("The density crop has no grid points outside the anchor region.")
    protein, protein_elements = read_protein_heavy_atoms(config["protein_path"])
    free = np.ones(ligand.GetNumAtoms(), dtype=bool)
    free[anchors] = False
    numbers = np.array([atom.GetAtomicNum() for atom in ligand.GetAtoms()])
    elements = [atom.GetSymbol() for atom in ligand.GetAtoms()]
    rows = []
    for path in sorted(directory.glob("index*/rank*_confidence*.sdf")):
        candidate = Chem.MolFromMolFile(str(path))
        if candidate is None or candidate.GetNumAtoms() != ligand.GetNumAtoms():
            raise ValueError(f"Could not read candidate in input atom order: {path.name}")
        xyz = candidate.GetConformer().GetPositions()
        deviations = np.linalg.norm(xyz[anchors] - fixed, axis=1)
        score = score_gaussian_density(crop, xyz[free], numbers[free], 1.5, mask)
        clashes = clash_metrics(protein, protein_elements, xyz, elements)["severe_clash_count"]
        stereo = candidate.GetSubstructMatch(ligand, useChirality=True) != ()
        reasons = []
        if deviations.max() > config["anchor_tolerance"]:
            reasons.append("anchor deviation exceeds tolerance")
        if score.ligand_in_bounds_fraction < 1.0 or not math.isfinite(score.ncc):
            reasons.append("free fragment lacks a valid density score inside the crop")
        if clashes:
            reasons.append("severe receptor clashes")
        if not stereo:
            reasons.append("stereochemistry mismatch")
        rows.append({
            "file": str(path.relative_to(directory.parent)),
            "model_rank": int(re.match(r"rank(\d+)", path.name)[1]),
            "anchor_rmsd_A": float(np.sqrt(np.mean(deviations ** 2))),
            "max_anchor_deviation_A": float(deviations.max()),
            "free_fragment_ncc": score.ncc,
            "free_atoms_in_map_fraction": score.ligand_in_bounds_fraction,
            "severe_receptor_clashes": clashes,
            "stereochemistry_matches": stereo,
            "review_reason": "; ".join(reasons),
        })
    if not rows:
        raise RuntimeError("Sampling produced no candidate SDF files. Check the inference output in run.log.")
    rows.sort(key=lambda row: (bool(row["review_reason"]),
                               -(row["free_fragment_ncc"] if math.isfinite(row["free_fragment_ncc"]) else -math.inf),
                               row["model_rank"]))
    with (directory.parent / "candidate_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if not rows[0]["review_reason"]:
        shutil.copy2(directory.parent / rows[0]["file"], directory.parent / "final_pose.sdf")


if __name__ == "__main__":
    main()
