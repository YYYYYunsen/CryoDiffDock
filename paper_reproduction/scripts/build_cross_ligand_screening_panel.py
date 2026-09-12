#!/usr/bin/env python3
"""Build a validation-only ligand-retrieval panel without coordinate leakage.

Each target is expanded into one true ligand and N property-matched decoys.
Every ligand input is regenerated from canonical SMILES, so deposited ligand
coordinates are never copied into the inference manifest.  The true/decoy
labels are written to a separate answer key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


DEFAULT_DESCRIPTORS = (
    "heavy_atoms", "molecular_weight", "rotatable_bonds", "formal_charge",
    "rings", "aromatic_rings", "logp", "tpsa",
)
GROUND_TRUTH_COLUMNS = {
    "reference_ligand", "reference_sdf", "native_ligand", "is_true",
    "ligand_rmsd_A", "rmsd", "rmsd_A",
}


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(molecule: Chem.Mol):
    # The legacy entry point is retained for the RDKit 2023.3 environment used
    # by the frozen docking workflow as well as newer RDKit releases.
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)


def molecule_from_smiles(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    return molecule


def write_input_sdf(smiles: str, destination: Path, seed: int) -> None:
    molecule = Chem.AddHs(molecule_from_smiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) & 0x7FFFFFFF
    status = AllChem.EmbedMolecule(molecule, params)
    if status != 0:
        raise ValueError(f"ETKDG embedding failed for {smiles}")
    try:
        AllChem.UFFOptimizeMolecule(molecule, maxIters=200)
    except Exception:
        pass
    molecule = Chem.RemoveHs(molecule)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(destination))
    writer.write(molecule)
    writer.close()


def target_id_column(frame: pd.DataFrame) -> str:
    for column in ("pdb_id", "target_id", "complex_name"):
        if column in frame.columns:
            return column
    raise ValueError("Target manifest needs pdb_id, target_id, or complex_name")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--chemistry-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-ids", type=Path)
    parser.add_argument("--target-split", default="val")
    parser.add_argument("--pool-splits", default="train")
    parser.add_argument("--ligand-class", default="drug_like")
    parser.add_argument("--num-targets", type=int, default=5)
    parser.add_argument("--decoys-per-target", type=int, default=19)
    parser.add_argument("--max-tanimoto", type=float, default=0.85)
    parser.add_argument("--descriptors", default=",".join(DEFAULT_DESCRIPTORS))
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_targets <= 0 or args.decoys_per_target <= 0:
        raise ValueError("num-targets and decoys-per-target must be positive")
    if not 0 <= args.max_tanimoto <= 1:
        raise ValueError("max-tanimoto must be in [0,1]")

    target_path = args.target_manifest.expanduser().resolve()
    chemistry_path = args.chemistry_manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(target_path, keep_default_na=False)
    chemistry = pd.read_csv(chemistry_path, keep_default_na=False)
    descriptors = tuple(x.strip() for x in args.descriptors.split(",") if x.strip())
    required = {"pdb_id", "canonical_smiles", "inchikey", "split", *descriptors}
    missing = sorted(required - set(chemistry.columns))
    if missing:
        raise ValueError(f"Chemistry manifest is missing: {missing}")
    if "model_included" in chemistry:
        chemistry = chemistry[truthy(chemistry["model_included"])].copy()
    class_column = "sampling_ligand_class" if "sampling_ligand_class" in chemistry else "ligand_class"
    if class_column not in chemistry:
        raise ValueError("Chemistry manifest has no ligand-class column")
    chemistry = chemistry[
        chemistry[class_column].astype(str).eq(args.ligand_class)
        & chemistry["canonical_smiles"].astype(str).str.len().gt(0)
        & chemistry["inchikey"].astype(str).str.len().gt(0)
    ].copy()
    chemistry["pdb_id"] = chemistry["pdb_id"].astype(str).str.upper()
    chemistry = chemistry.drop_duplicates("pdb_id", keep="first")

    id_column = target_id_column(targets)
    targets = targets.copy()
    targets["_target_id"] = targets[id_column].astype(str).str.upper()
    target_split = chemistry.loc[
        chemistry["split"].astype(str).eq(args.target_split), "pdb_id"
    ]
    targets = targets[targets["_target_id"].isin(set(target_split))]
    if args.target_ids:
        requested = [
            line.strip().upper() for line in args.target_ids.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        order = {value: index for index, value in enumerate(requested)}
        targets = targets[targets["_target_id"].isin(order)].copy()
        targets["_order"] = targets["_target_id"].map(order)
        targets = targets.sort_values("_order", kind="stable").drop(columns="_order")
        absent = sorted(set(requested) - set(targets["_target_id"]))
        if absent:
            raise ValueError(f"Requested validation targets are absent: {absent}")
    else:
        rng = np.random.default_rng(args.seed)
        targets = targets.sort_values("_target_id", kind="stable")
        if len(targets) > args.num_targets:
            chosen = rng.choice(len(targets), size=args.num_targets, replace=False)
            targets = targets.iloc[np.sort(chosen)]
    targets = targets.head(args.num_targets).reset_index(drop=True)
    if len(targets) < args.num_targets:
        raise ValueError(f"Only {len(targets)} eligible targets were available")

    pool_splits = {x.strip() for x in args.pool_splits.split(",") if x.strip()}
    pool = chemistry[chemistry["split"].astype(str).isin(pool_splits)].copy()
    pool = pool.drop_duplicates("inchikey", keep="first").reset_index(drop=True)
    numeric = chemistry.loc[:, descriptors].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median(axis=0)
    scales = numeric.std(axis=0, ddof=0).replace(0, 1).fillna(1)
    pool_numeric = pool.loc[:, descriptors].apply(pd.to_numeric, errors="coerce").fillna(medians)
    chemistry_by_id = chemistry.set_index("pdb_id", drop=False)

    rng = np.random.default_rng(args.seed)
    ligand_dir = output / "ligands_from_smiles"
    answer_rows: list[dict] = []
    inference_rows: list[dict] = []
    generated: dict[str, str] = {}

    for target_position, target_row in targets.iterrows():
        target_id = target_row["_target_id"]
        true = chemistry_by_id.loc[target_id]
        if isinstance(true, pd.DataFrame):
            true = true.iloc[0]
        true_mol = molecule_from_smiles(true["canonical_smiles"])
        true_fp = fingerprint(true_mol)
        true_values = pd.to_numeric(true.loc[list(descriptors)], errors="coerce").fillna(medians)
        distances = np.sqrt((((pool_numeric - true_values) / scales) ** 2).mean(axis=1))
        choices = []
        for pool_index in np.argsort(distances.to_numpy(), kind="stable"):
            candidate = pool.iloc[int(pool_index)]
            if candidate["inchikey"] == true["inchikey"]:
                continue
            candidate_mol = molecule_from_smiles(candidate["canonical_smiles"])
            similarity = float(DataStructs.TanimotoSimilarity(true_fp, fingerprint(candidate_mol)))
            if similarity > args.max_tanimoto:
                continue
            choices.append((candidate, float(distances.iloc[int(pool_index)]), similarity))
            if len(choices) >= args.decoys_per_target:
                break
        if len(choices) < args.decoys_per_target:
            raise ValueError(
                f"{target_id}: only {len(choices)} matched decoys passed filters; "
                "relax --max-tanimoto or pool restrictions"
            )

        candidates = [(true, 0.0, 1.0, True)] + [(*choice, False) for choice in choices]
        order = rng.permutation(len(candidates))
        for display_position, candidate_index in enumerate(order, start=1):
            candidate, distance, similarity, is_true = candidates[int(candidate_index)]
            ligand_key = safe_name(candidate["inchikey"])
            if ligand_key not in generated:
                sdf = ligand_dir / f"{ligand_key}.sdf"
                conformer_seed = args.seed + int(hashlib.sha256(ligand_key.encode()).hexdigest()[:7], 16)
                write_input_sdf(candidate["canonical_smiles"], sdf, conformer_seed)
                generated[ligand_key] = str(sdf.resolve())
            screen_name = safe_name(
                f"screen_{target_id}_{display_position:04d}_{candidate['pdb_id']}"
            )
            inference = {
                key: value for key, value in target_row.to_dict().items()
                if key not in GROUND_TRUTH_COLUMNS and not key.startswith("_")
            }
            inference["complex_name"] = screen_name
            inference["ligand"] = generated[ligand_key]
            inference["screen_target_id"] = target_id
            inference["screen_candidate_id"] = str(candidate["pdb_id"])
            inference_rows.append(inference)
            answer = {
                "complex_name": screen_name,
                "target_id": target_id,
                "candidate_position": display_position,
                "candidate_pdb_id": str(candidate["pdb_id"]),
                "candidate_inchikey": str(candidate["inchikey"]),
                "candidate_smiles": str(candidate["canonical_smiles"]),
                "is_true": bool(is_true),
                "property_distance": distance,
                "tanimoto_to_true": similarity,
            }
            answer.update({key: candidate[key] for key in descriptors})
            answer_rows.append(answer)

    inference_frame = pd.DataFrame(inference_rows)
    answer_frame = pd.DataFrame(answer_rows)
    if any(column in inference_frame for column in GROUND_TRUTH_COLUMNS):
        raise RuntimeError("Ground-truth column leaked into inference manifest")
    expected = args.num_targets * (args.decoys_per_target + 1)
    if len(inference_frame) != expected or answer_frame.groupby("target_id")["is_true"].sum().ne(1).any():
        raise RuntimeError("Panel cardinality or one-true-ligand invariant failed")

    inference_file = output / "inference_blind.csv"
    answer_file = output / "answer_key.csv"
    inference_frame.to_csv(inference_file, index=False)
    answer_frame.to_csv(answer_file, index=False)
    summary = {
        "task": "oracle-pocket cross-ligand retrieval audit",
        "num_targets": args.num_targets,
        "candidates_per_target": args.decoys_per_target + 1,
        "target_split": args.target_split,
        "pool_splits": sorted(pool_splits),
        "ligand_class": args.ligand_class,
        "max_tanimoto": args.max_tanimoto,
        "descriptors": list(descriptors),
        "seed": args.seed,
        "input_coordinates_regenerated_from_smiles": True,
        "ground_truth_present_in_inference_manifest": False,
        "target_manifest_sha256": sha256(target_path),
        "chemistry_manifest_sha256": sha256(chemistry_path),
        "inference_manifest_sha256": sha256(inference_file),
        "answer_key_sha256": sha256(answer_file),
    }
    (output / "panel_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
