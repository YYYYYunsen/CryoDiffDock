#!/usr/bin/env python3
"""Locally relax sampled poses with a VK3 head restraint and full-protein clashes."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import minimize
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryodiffdock_pocket.physics import atom_radii, read_protein_heavy_atoms

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--inference", required=True)
p.add_argument("--out", required=True)
p.add_argument("--iterations", type=int, default=2000)
a = p.parse_args()
cfg = json.loads(Path(a.config).read_text())
q = Chem.SDMolSupplier(cfg["ligand_path"])[0]
ref = Chem.SDMolSupplier(cfg["reference_path"])[0].GetConformer().GetPositions()
head = np.array(cfg["ligand_indices"])
protein, el = read_protein_heavy_atoms(cfg["protein_path"])
pr = atom_radii(el)
tree = cKDTree(protein)
dest = Path(a.out)
dest.mkdir(parents=True, exist_ok=True)
rows = []
for f in sorted(Path(a.inference).glob("index*/rank*_confidence*.sdf")):
    mol = Chem.SDMolSupplier(str(f), removeHs=True)[0]
    if mol is None:
        continue
    matches = mol.GetSubstructMatches(
        q, useChirality=True, uniquify=False, maxMatches=1000
    )
    if not matches:
        print("SKIP_STEREO", f.name, flush=True)
        continue
    ids = np.array(matches[0])[head]
    heavy = mol.GetNumAtoms()
    lr = atom_radii([x.GetSymbol() for x in mol.GetAtoms()])
    mh = Chem.AddHs(mol, addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(mh)
    if props is None:
        print("SKIP_MMFF", f.name, flush=True)
        continue
    ff = AllChem.MMFFGetMoleculeForceField(mh, props, ignoreInterfragInteractions=False)
    x0 = mh.GetConformer().GetPositions()
    subsets = tree.query_ball_point(x0[:heavy], 20)
    used = np.unique(
        np.concatenate([np.asarray(v, dtype=int) for v in subsets if len(v)])
    )
    pc = protein[used]
    r = pr[used]
    local = cKDTree(pc)

    def objective(flat):
        x = flat.reshape(-1, 3)
        energy = float(ff.CalcEnergy(flat.tolist()))
        grad = np.asarray(ff.CalcGrad(flat.tolist())).reshape(-1, 3)
        delta = x[ids] - ref
        energy += 20 * (delta ** 2).sum()
        np.add.at(grad, ids, 40 * delta)
        ii = []
        jj = []
        for i, js in enumerate(local.query_ball_point(x[:heavy], 4.5)):
            ii.extend([i] * len(js))
            jj.extend(js)
        if ii:
            ii = np.asarray(ii)
            jj = np.asarray(jj)
            d = x[ii] - pc[jj]
            ds = np.maximum(np.linalg.norm(d, axis=1), 1e-8)
            over = np.maximum(lr[ii] + r[jj] - 0.6 - ds, 0)
            energy += 20 * np.sum(over ** 2)
            np.add.at(grad, ii, -40 * (over / ds)[:, None] * d)
        return energy, grad.ravel()

    result = minimize(
        objective,
        x0.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": a.iterations, "ftol": 1e-8},
    )
    x = result.x.reshape(-1, 3)[:heavy]
    for i, point in enumerate(x):
        mol.GetConformer().SetAtomPosition(i, point.tolist())
    # Reinterpret stereochemistry from resulting coordinates, not only input SDF labels.
    Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    stereo = bool(mol.GetSubstructMatch(q, useChirality=True))
    output = dest / f.name
    w = Chem.SDWriter(str(output))
    w.write(mol)
    w.close()
    rows.append(
        {
            "input": str(f),
            "output": str(output),
            "iterations": result.nit,
            "optimizer_converged": bool(result.success),
            "message": str(result.message),
            "stereo_consistent": stereo,
            "head_rmsd_A": float(np.sqrt(np.mean(np.sum((x[ids] - ref) ** 2, axis=1)))),
        }
    )
    print(
        "REPAIR",
        f.name,
        "head",
        round(rows[-1]["head_rmsd_A"], 3),
        "converged",
        result.success,
        "stereo",
        stereo,
        flush=True,
    )
if not rows:
    raise RuntimeError("No stereochemically matching input poses")
with (dest / "repair_audit.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
