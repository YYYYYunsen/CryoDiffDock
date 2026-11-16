#!/usr/bin/env python3
"""Reference-guided exit-bond scan and density/clash refinement; research prototype.

Angles are supplied explicitly as four zero-based ligand SDF atom indices each.
Same input seeds/angles for strict and relaxed head. Reference data cannot serve
as independent validation. Protein proximity is not a tunnel segmentation.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryodiffdock_pocket.physics import atom_radii, read_protein_heavy_atoms
from cryodiffdock_pocket.density_ranking import (
    load_cartesian_density_crop,
    fixed_crop_mask,
    score_gaussian_density,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", required=True)
p.add_argument("--manifest", required=True)
p.add_argument("--seeds", required=True)
p.add_argument("--out", required=True)
p.add_argument(
    "--dihedral1", required=True, help="Four comma-separated zero-based atom indices"
)
p.add_argument(
    "--dihedral2", required=True, help="Four comma-separated zero-based atom indices"
)
p.add_argument("--nseeds", type=int, default=3)
p.add_argument("--angle-step", type=int, default=30)
p.add_argument("--second-angles", default="-120,0,120")
p.add_argument("--iterations", type=int, default=300)
p.add_argument("--density-weight", type=float, default=1.0)
p.add_argument("--threads", type=int, default=1)
a = p.parse_args()
start = time.monotonic()
if a.angle_step < 1 or 360 % a.angle_step or a.nseeds < 1 or a.iterations < 1:
    raise ValueError(
        "angle step must divide 360; seeds and iterations must be positive"
    )
cfg = json.loads(Path(a.config).read_text())
row = next(csv.DictReader(Path(a.manifest).open()))
lig = Chem.SDMolSupplier(cfg["ligand_path"])[0]
refmol = Chem.SDMolSupplier(cfg["reference_path"])[0]
ref = refmol.GetConformer().GetPositions()
head = np.array(cfg["ligand_indices"])
headset = set(head.tolist())


def parse_dihedral(text):
    values = tuple(int(i) for i in text.split(","))
    if (
        len(values) != 4
        or len(set(values)) != 4
        or min(values) < 0
        or max(values) >= lig.GetNumAtoms()
    ):
        raise ValueError("A dihedral requires four distinct valid atom indices")
    if any(lig.GetBondBetweenAtoms(values[i], values[i + 1]) is None for i in range(3)):
        raise ValueError("Dihedral atoms must be sequentially bonded")
    bond = lig.GetBondBetweenAtoms(values[1], values[2])
    if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
        raise ValueError(
            "Central dihedral bond must be a rotatable non-ring single bond"
        )
    return values


dih1 = parse_dihedral(a.dihedral1)
dih2 = parse_dihedral(a.dihedral2)
if dih1[1] not in headset or dih1[2] in headset:
    raise ValueError("The first dihedral must point from head to tail")
for dih in (dih1, dih2):
    cut = Chem.RWMol(lig)
    cut.RemoveBond(dih[1], dih[2])
    fragments = Chem.GetMolFrags(cut.GetMol())
    moving = next(set(f) for f in fragments if dih[2] in f)
    if moving & headset:
        raise ValueError("Dihedral rotation would move reference head atoms")
protein, pe = read_protein_heavy_atoms(cfg["protein_path"])
pr = atom_radii(pe)
tree = cKDTree(protein)
crop = load_cartesian_density_crop(row["ligand_density_crop"])
mask = fixed_crop_mask(crop, radius_A=36)
grid = gaussian_filter(
    np.clip((crop.density - crop.norm_center) / crop.norm_scale, -1, 4).astype(
        np.float32
    ),
    sigma=1,
)
grad = np.asarray(np.gradient(grid, *crop.spacing))
origin = crop.origin
spacing = crop.spacing
seeds = []
for f in sorted(Path(a.seeds).glob("index*/rank*_confidence*.sdf")):
    m = Chem.SDMolSupplier(str(f), removeHs=True)[0]
    if m is None or not m.GetSubstructMatch(lig, useChirality=True):
        continue
    # Demand exact atom order to preserve the semantics of torsion atom indices.
    if tuple(m.GetSubstructMatch(lig, useChirality=True)) != tuple(
        range(lig.GetNumAtoms())
    ):
        continue
    seeds.append((f, m))
if not seeds:
    raise ValueError("No stereo-consistent seeds with matching atom order")
seeds = seeds[: a.nseeds]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
angles = [
    (x, int(y))
    for x in range(-180, 180, a.angle_step)
    for y in a.second_angles.split(",")
]
jobs = [
    (i, x, y, mode)
    for i in range(len(seeds))
    for x, y in angles
    for mode in ("strict", "relaxed")
]
print(
    "SCAN_START",
    json.dumps(
        {
            "seed_files": [str(s[0]) for s in seeds],
            "bond_1": dih1,
            "bond_2": dih2,
            "grid_angles": len(angles),
            "tasks": len(jobs),
            "modes": {"strict_head_force": 2000, "relaxed_head_force": 5},
            "density_weight": a.density_weight,
            "reference_assisted": True,
            "tunnel_constraint": "nearest-protein proxy; no segmented tunnel mask",
        }
    ),
    flush=True,
)


def run(job):
    si, angle1, angle2, mode = job
    source, base = seeds[si]
    mol = Chem.Mol(base)
    conf = mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf, *dih1, float(angle1))
    rdMolTransforms.SetDihedralDeg(conf, *dih2, float(angle2))
    heavy = mol.GetNumAtoms()
    tail = np.array([i for i in range(heavy) if i not in headset])
    radii = atom_radii([x.GetSymbol() for x in mol.GetAtoms()])
    mh = Chem.AddHs(mol, addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(mh)
    if props is None:
        raise ValueError("MMFF properties unavailable")
    ff = AllChem.MMFFGetMoleculeForceField(mh, props, ignoreInterfragInteractions=False)
    # Hold each tested exit angle within a 15-degree window while relaxing distal torsions.
    ff.MMFFAddTorsionConstraint(
        *dih1, False, float(angle1 - 7.5), float(angle1 + 7.5), 100.0
    )
    ff.MMFFAddTorsionConstraint(
        *dih2, False, float(angle2 - 7.5), float(angle2 + 7.5), 100.0
    )
    x0 = mh.GetConformer().GetPositions()
    k = 2000.0 if mode == "strict" else 5.0

    def objective(flat):
        x = flat.reshape(-1, 3)
        energy = float(ff.CalcEnergy(flat.tolist()))
        g = np.asarray(ff.CalcGrad(flat.tolist())).reshape(-1, 3)
        delta = x[head] - ref
        energy += k * np.sum(delta * delta)
        np.add.at(g, head, 2 * k * delta)
        # Evaluate neighbours afresh: the long distal tail can travel many angstroms.
        ii = []
        jj = []
        for i, js in enumerate(tree.query_ball_point(x[:heavy], 4.6)):
            ii.extend([i] * len(js))
            jj.extend(js)
        if ii:
            ii = np.asarray(ii)
            jj = np.asarray(jj)
            d = x[ii] - protein[jj]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-8)
            over = np.maximum(radii[ii] + pr[jj] - 0.6 - dist, 0)
            energy += 25 * np.sum(over ** 2)
            np.add.at(g, ii, -50 * (over / dist)[:, None] * d)
        # Smooth, bounded protein-masked map sampled at tail atoms; background is not guaranteed to be ligand density.
        coord = ((x[tail] - origin) / spacing).T
        vals = map_coordinates(grid, coord, order=1, mode="constant", cval=0.0)
        der = np.stack(
            [
                map_coordinates(grad[j], coord, order=1, mode="constant", cval=0.0)
                for j in range(3)
            ],
            axis=1,
        )
        energy -= a.density_weight * vals.sum()
        np.add.at(g, tail, -a.density_weight * der)
        # Weak proximity penalty only; does not establish that a point is in a tunnel.
        near, nn = tree.query(x[tail])
        excess = np.maximum(near - 7.0, 0)
        energy += 0.5 * np.sum(excess ** 2)
        np.add.at(
            g,
            tail,
            (excess / np.maximum(near, 1e-8))[:, None] * (x[tail] - protein[nn]),
        )
        return energy, g.ravel()

    res = minimize(
        objective,
        x0.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": a.iterations, "ftol": 1e-7},
    )
    x = res.x.reshape(-1, 3)[:heavy]
    for i, v in enumerate(x):
        conf.SetAtomPosition(i, v.tolist())
    Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    stereo = bool(mol.GetSubstructMatch(lig, useChirality=True))
    near = tree.query(x[tail])[0]
    clashes = 0
    for i, js in enumerate(tree.query_ball_point(x, 4.6)):
        if js:
            clashes += int(
                (
                    radii[i] + pr[js] - np.linalg.norm(protein[js] - x[i], axis=1) > 0.8
                ).sum()
            )
    ncc = score_gaussian_density(
        crop, x, [z.GetAtomicNum() for z in mol.GetAtoms()], 1.5, mask
    )
    rmsd = float(np.sqrt(np.mean(np.sum((x[head] - ref) ** 2, axis=1))))
    outpath = (
        out / "candidates" / mode / f"seed{si:02d}_a{angle1:+04d}_b{angle2:+04d}.sdf"
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(outpath))
    w.write(mol)
    w.close()
    return {
        "mode": mode,
        "seed": str(source),
        "angle1_grid_deg": angle1,
        "angle2_grid_deg": angle2,
        "angle1_final_deg": rdMolTransforms.GetDihedralDeg(conf, *dih1),
        "angle2_final_deg": rdMolTransforms.GetDihedralDeg(conf, *dih2),
        "head_rmsd_A": rmsd,
        "head_gate_pass": rmsd <= (0.3 if mode == "strict" else 1.0),
        "stereo_matches": stereo,
        "severe_clash_pairs": clashes,
        "tail_gt_7A": int((near > 7).sum()),
        "tail_max_protein_dist_A": float(near.max()),
        "density_ncc": float(ncc.ncc),
        "crop_in_bounds_fraction": float(ncc.ligand_in_bounds_fraction),
        "optimizer_converged": bool(res.success),
        "optimizer_message": str(res.message),
        "iterations": int(res.nit),
        "pose": str(outpath),
    }


# Threading is optional. The force field/molecule is separately created for each job.
from concurrent.futures import ThreadPoolExecutor

results = []
with ThreadPoolExecutor(max_workers=a.threads) as pool:
    for i, r in enumerate(pool.map(run, jobs), 1):
        results.append(r)
        if i % 12 == 0 or i == len(jobs):
            print(
                f"SCAN_PROGRESS {i}/{len(jobs)} elapsed={time.monotonic()-start:.1f}s",
                flush=True,
            )
with (out / "candidates.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0])
    w.writeheader()
    w.writerows(results)
for mode in ("strict", "relaxed"):
    subset = [r for r in results if r["mode"] == mode]
    valid = [
        r
        for r in subset
        if r["stereo_matches"]
        and r["head_gate_pass"]
        and r["optimizer_converged"]
        and r["crop_in_bounds_fraction"] >= 0.99
    ]
    valid.sort(
        key=lambda r: (r["severe_clash_pairs"], r["tail_gt_7A"], -r["density_ncc"])
    )
    report = {
        "mode": mode,
        "total": len(subset),
        "eligible_converged_stereo_head_and_inbounds": len(valid),
        "eligible_zero_clash": sum(r["severe_clash_pairs"] == 0 for r in valid),
        "best": valid[0] if valid else None,
    }
    (out / f"{mode}_summary.json").write_text(json.dumps(report, indent=2))
    print("SCAN_SUMMARY", json.dumps(report), flush=True)
