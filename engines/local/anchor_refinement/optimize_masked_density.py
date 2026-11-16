#!/usr/bin/env python3
"""Optimize a fixed-head ligand tail against protein-masked experimental density.

This is reference-assisted candidate refinement, not an independent prediction.
No explicit tunnel segmentation or ligand-specific density is assumed. It uses
MMFF intramolecular energy, a harmonic head restraint, full-protein steric
penalty and a bounded smoothed map potential. The density weight 0 is a control.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryodiffdock_pocket.physics import atom_radii, read_protein_heavy_atoms
from cryodiffdock_pocket.density_ranking import (
    load_cartesian_density_crop,
    fixed_crop_mask,
    score_gaussian_density,
)


def sample_trilinear(grid, xyz):
    """Exact values/gradients of trilinear interpolation, zero outside."""
    shape = np.array(grid.shape)
    valid = ((xyz >= 0) & (xyz < shape - 1)).all(axis=1)
    base = np.clip(np.floor(xyz).astype(int), 0, shape - 2)
    frac = np.clip(xyz - base, 0, 1)
    values = np.zeros(len(xyz))
    grads = np.zeros_like(xyz)
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                bits = np.array((i, j, k))
                w = np.where(bits, frac, 1 - frac)
                v = grid[base[:, 0] + i, base[:, 1] + j, base[:, 2] + k]
                values += v * np.prod(w, axis=1)
                for dim in range(3):
                    grads[:, dim] += (
                        v
                        * (1 if bits[dim] else -1)
                        * w[:, (dim + 1) % 3]
                        * w[:, (dim + 2) % 3]
                    )
    values[~valid] = 0
    grads[~valid] = 0
    return values, grads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iterations", type=int, default=1800)
    p.add_argument("--weights", default="0,1,2,4")
    p.add_argument("--max-seeds", type=int, default=8)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    row = next(csv.DictReader(open(args.manifest)))
    seedrows = list(csv.DictReader(open(args.seeds)))[: args.max_seeds]
    weights = [float(x) for x in args.weights.split(",")]
    if not seedrows or any(x < 0 for x in weights):
        raise ValueError("No seeds or negative density weights")
    ref = Chem.SDMolSupplier(cfg["reference_path"])[0]
    reference = ref.GetConformer().GetPositions()
    q = Chem.SDMolSupplier(cfg["ligand_path"])[0]
    head = np.array(cfg["ligand_indices"])
    protein, el = read_protein_heavy_atoms(cfg["protein_path"])
    radii_p = atom_radii(el)
    tree = cKDTree(protein)
    crop = load_cartesian_density_crop(row["ligand_density_crop"])
    mask = fixed_crop_mask(crop, radius_A=36)
    # Values are in units of the crop standard deviation; capping limits attraction
    # to one unusually strong residual peak. Sigma is in voxels.
    density = np.clip((crop.density - crop.norm_center) / crop.norm_scale, 0, 4)
    density = gaussian_filter(density, sigma=1.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    clock = time.monotonic()
    for si, seedrow in enumerate(seedrows):
        source = seedrow["pose"]
        initial = Chem.SDMolSupplier(source)[0]
        if initial is None or tuple(
            initial.GetSubstructMatch(q, useChirality=True)
        ) != tuple(range(q.GetNumAtoms())):
            print("SKIP_SEED_STEREO_OR_ORDER", source, flush=True)
            continue
        for weight in weights:
            mol = Chem.Mol(initial)
            nh = mol.GetNumAtoms()
            tail = np.array([i for i in range(nh) if i not in set(head)])
            lr = atom_radii([x.GetSymbol() for x in mol.GetAtoms()])
            mh = Chem.AddHs(mol, addCoords=True)
            props = AllChem.MMFFGetMoleculeProperties(mh)
            if props is None:
                raise ValueError("MMFF parameters unavailable")
            ff = AllChem.MMFFGetMoleculeForceField(
                mh, props, ignoreInterfragInteractions=False
            )
            x0 = mh.GetConformer().GetPositions()

            def objective(flat):
                x = flat.reshape(-1, 3)
                energy = float(ff.CalcEnergy(flat.tolist()))
                grad = np.asarray(ff.CalcGrad(flat.tolist())).reshape(-1, 3)
                delta = x[head] - reference
                energy += 30.0 * np.sum(delta ** 2)
                np.add.at(grad, head, 60.0 * delta)
                ii = []
                jj = []
                for i, js in enumerate(tree.query_ball_point(x[:nh], 4.6)):
                    ii.extend([i] * len(js))
                    jj.extend(js)
                if ii:
                    ii = np.array(ii)
                    jj = np.array(jj)
                    d = x[ii] - protein[jj]
                    ds = np.maximum(np.linalg.norm(d, axis=1), 1e-8)
                    over = np.maximum(lr[ii] + radii_p[jj] - 0.6 - ds, 0)
                    energy += 30 * np.sum(over ** 2)
                    np.add.at(grad, ii, -60 * (over / ds)[:, None] * d)
                vals, dgrid = sample_trilinear(
                    density, (x[tail] - crop.origin) / crop.spacing
                )
                energy -= weight * float(vals.sum())
                np.add.at(grad, tail, -weight * dgrid / crop.spacing)
                # A crop boundary is a numerical condition, not a physical tunnel wall.
                lower = crop.origin
                upper = crop.origin + (np.array(crop.density.shape) - 1) * crop.spacing
                lo = np.minimum(x[tail] - lower, 0)
                hi = np.maximum(x[tail] - upper, 0)
                energy += 20 * np.sum(lo ** 2 + hi ** 2)
                np.add.at(grad, tail, 40 * (lo + hi))
                return energy, grad.ravel()

            res = minimize(
                objective,
                x0.ravel(),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": args.iterations, "ftol": 1e-9, "gtol": 1e-4},
            )
            x = res.x.reshape(-1, 3)[:nh]
            for i, pt in enumerate(x):
                mol.GetConformer().SetAtomPosition(i, pt.tolist())
            Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
            stereo = bool(mol.GetSubstructMatch(q, useChirality=True))
            rmsd = float(np.sqrt(np.mean(np.sum((x[head] - reference) ** 2, axis=1))))
            nearest = tree.query(x[tail])[0]
            clashes = 0
            for i, js in enumerate(tree.query_ball_point(x, 4.6)):
                if js:
                    clashes += int(
                        (
                            lr[i]
                            + radii_p[js]
                            - np.linalg.norm(protein[js] - x[i], axis=1)
                            > 0.8
                        ).sum()
                    )
            ncc = score_gaussian_density(
                crop, x, [z.GetAtomicNum() for z in mol.GetAtoms()], 1.5, mask
            )
            coordinates = ((x[tail] - crop.origin) / crop.spacing).T
            raw = (
                map_coordinates(
                    crop.density, coordinates, order=1, mode="constant", cval=0
                )
                - crop.norm_center
            ) / crop.norm_scale
            output = out / f"seed{si:02d}_density{weight:g}.sdf"
            w = Chem.SDWriter(str(output))
            w.write(mol)
            w.close()
            rec = {
                "seed": source,
                "density_weight": weight,
                "pose": str(output),
                "head_rmsd_A": rmsd,
                "stereo_matches": stereo,
                "severe_clash_pairs": clashes,
                "tail_max_protein_distance_A": float(nearest.max()),
                "tail_gt7_A": int((nearest > 7).sum()),
                "density_ncc": float(ncc.ncc),
                "tail_mean_residual_z": float(raw.mean()),
                "tail_gt1_residual_z_fraction": float((raw > 1).mean()),
                "crop_in_bounds_fraction": float(ncc.ligand_in_bounds_fraction),
                "optimizer_converged": bool(res.success),
                "optimizer_message": str(res.message),
                "iterations": int(res.nit),
            }
            records.append(rec)
            print("OPTIMIZED", json.dumps(rec), flush=True)
    if not records:
        raise RuntimeError("No eligible input seeds")
    with (out / "candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0])
        w.writeheader()
        w.writerows(records)
    eligible = [
        r
        for r in records
        if r["stereo_matches"]
        and r["head_rmsd_A"] <= 0.5
        and r["severe_clash_pairs"] == 0
        and r["crop_in_bounds_fraction"] >= 0.99
        and r["optimizer_converged"]
    ]
    eligible.sort(key=lambda r: -r["density_ncc"])
    report = {
        "candidate_count": len(records),
        "eligible_count": len(eligible),
        "selected": eligible[0] if eligible else None,
        "criteria": "converged, correct stereo, head RMSD <=0.5A, zero severe protein clashes, >=99% inside map crop; highest masked-map NCC",
        "caveat": "same map used for optimization and scoring; no independent validation; mask does not identify tunnel",
        "runtime_s": time.monotonic() - clock,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print("OPTIMIZATION_FINISHED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
