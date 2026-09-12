#!/usr/bin/env python3
"""Score paired ChemEM and CryoDiffDock-Local Top-1 poses against one map crop.

The two poses for each complex are evaluated with the same experimental-density
crop, fixed spatial support, Gaussian width, and atom-weighting implementation.
Native coordinates are never used by the density scorer.  RMSDs are carried
through only for downstream diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cryodiffdock_pocket.density_ranking import (  # noqa: E402
    fixed_crop_mask,
    load_cartesian_density_crop,
    score_gaussian_density,
)


ID_COLUMNS = ("complex_name", "complex_id", "pdb_id", "id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--chemem-benchmark", required=True, type=Path)
    parser.add_argument("--local-metrics", required=True, type=Path)
    parser.add_argument("--local-manifest", required=True, type=Path)
    parser.add_argument("--chemem-root", action="append", default=[], type=Path)
    parser.add_argument("--local-selected-root", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-paired-n", type=int, default=165)
    parser.add_argument("--sigma-A", type=float, default=1.5)
    parser.add_argument("--mask-radius-A", type=float, default=10.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    values = [line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines()]
    values = [value for value in values if value]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Empty or duplicate cohort IDs: {path}")
    return values


def read_table(path: Path, label: str) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    frame = pd.read_csv(path, sep=separator)
    id_column = next((column for column in ID_COLUMNS if column in frame), None)
    if id_column is None:
        raise ValueError(f"{label} has no complex ID column: {list(frame.columns)}")
    frame = frame.copy()
    frame["complex_name"] = frame[id_column].astype(str).str.strip().str.upper()
    if frame["complex_name"].duplicated().any():
        duplicates = frame.loc[frame["complex_name"].duplicated(False), "complex_name"].unique()
        raise ValueError(f"Duplicate IDs in {label}: {duplicates[:10].tolist()}")
    return frame


def finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def resolve_record_path(value: object, table_path: Path) -> Path | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = table_path.parent / path
    return path.resolve()


def exactly_one(paths: list[Path], label: str) -> Path:
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    if len(unique) != 1:
        raise ValueError(f"Expected one {label}; found {len(unique)}: {[str(p) for p in unique[:8]]}")
    return unique[0]


def chemem_pose(
    record: pd.Series,
    roots: list[Path],
    iteration: int,
    complex_name: str,
) -> Path:
    fitting_candidates: list[Path] = []
    for column in ("chemem_results_path", "results_path"):
        path = resolve_record_path(record.get(column, ""), Path(str(record.get("_table_path"))))
        if path is not None:
            fitting_candidates.append(path.parent)
    for column in ("chemem_stage1_output", "stage1_output"):
        path = resolve_record_path(record.get(column, ""), Path(str(record.get("_table_path"))))
        if path is not None:
            fitting_candidates.append(path / "fitting")
    explicit_fitting_count = len(fitting_candidates)
    for root in roots:
        fitting_candidates.extend([
            root / "runs" / complex_name / "stage1" / "fitting",
            root / "01_stage1" / "runs" / complex_name / "stage1" / "fitting",
            root / "01_stage1" / "outputs" / complex_name / "stage1" / "fitting",
        ])
    direct = []
    for fitting in fitting_candidates:
        direct.extend([
            fitting / f"Ligand_{iteration}_refined.sdf",
            *sorted(fitting.glob(f"Ligand_{iteration}_*_refined.sdf")),
        ])
    existing = sorted({path.resolve() for path in direct if path.is_file()})
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        # Multiple roots can contain byte-identical reusable results.  Prefer
        # an explicit results/stage1 path over broad root discovery.
        explicit = [
            path for path in existing
            if any(path.parent == candidate.resolve() for candidate in fitting_candidates[:explicit_fitting_count])
        ]
        if explicit:
            return explicit[0]
        # Otherwise preserve the user-supplied root priority.  The selected
        # absolute pose path remains in the per-case audit table.
        for fitting in fitting_candidates[explicit_fitting_count:]:
            for candidate in (fitting / f"Ligand_{iteration}_refined.sdf", *sorted(fitting.glob(f"Ligand_{iteration}_*_refined.sdf"))):
                if candidate.is_file():
                    return candidate.resolve()
    recursive = []
    for root in roots:
        recursive.extend(root.glob(f"**/{complex_name}/stage1/fitting/Ligand_{iteration}_refined.sdf"))
    return exactly_one(recursive, f"ChemEM Top-1 pose for {complex_name}")


def local_pose(record: pd.Series, roots: list[Path], complex_name: str) -> Path:
    row_index = int(float(record["row_index"])) if math.isfinite(finite(record.get("row_index"))) else None
    # Explicit --local-selected-root inputs are authoritative and ordered by
    # caller priority.  Do not pool them with the historical source_file path:
    # the Top-1 runtime parity rerun intentionally materializes an identical
    # copy of the selected pose at a second absolute path.
    for root in roots:
        candidates: list[Path] = []
        if row_index is not None:
            candidates.extend((root / f"index{row_index}___{complex_name}").glob("rank1_confidence*.sdf"))
        candidates.extend(root.glob(f"index*___{complex_name}/rank1_confidence*.sdf"))
        unique = sorted({path.resolve() for path in candidates if path.is_file()})
        if unique:
            return exactly_one(unique, f"CryoDiffDock-Local V2 Top-1 pose for {complex_name} in {root}")

    candidates = []
    source = resolve_record_path(record.get("source_file", ""), Path(str(record.get("_table_path"))))
    if source is not None:
        run_root = source.parent.parent if source.parent.name == "metrics_v2_selected" else source.parent
        selected = run_root / "v2_selected"
        if row_index is not None:
            candidates.extend((selected / f"index{row_index}___{complex_name}").glob("rank1_confidence*.sdf"))
        candidates.extend(selected.glob(f"index*___{complex_name}/rank1_confidence*.sdf"))
    return exactly_one(candidates, f"CryoDiffDock-Local V2 Top-1 pose for {complex_name}")


def density_crop(record: pd.Series, manifest_path: Path, complex_name: str) -> Path:
    for column in ("ligand_density_crop", "density_crop", "crop_npz"):
        if column in record:
            path = resolve_record_path(record.get(column), manifest_path)
            if path is not None and path.is_file():
                return path
    raise FileNotFoundError(f"No density crop for {complex_name}")


def read_pose(path: Path):
    from rdkit import Chem

    suffix = path.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        molecule = next((value for value in supplier if value is not None), None)
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    elif suffix == ".pdb":
        molecule = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)
    else:
        raise ValueError(f"Unsupported pose format: {path}")
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read pose: {path}")
    return molecule


def score_pose(path: Path, crop, mask: np.ndarray, sigma_A: float) -> dict[str, float | int]:
    molecule = read_pose(path)
    conformer = molecule.GetConformer()
    indices = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    coordinates = np.asarray(
        [list(conformer.GetAtomPosition(index)) for index in indices], dtype=np.float32
    )
    atomic_numbers = np.asarray(
        [molecule.GetAtomWithIdx(index).GetAtomicNum() for index in indices], dtype=np.float32
    )
    score = score_gaussian_density(crop, coordinates, atomic_numbers, sigma_A=sigma_A, mask=mask)
    return {
        "ncc": float(score.ncc),
        "cosine": float(score.cosine_overlap),
        "mask_voxels": int(score.mask_voxels),
        "in_bounds_fraction": float(score.ligand_in_bounds_fraction),
        "heavy_atoms": len(indices),
    }


def bootstrap(values: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    medians = np.median(values[indices], axis=1)
    return tuple(float(value) for value in np.quantile(medians, [0.025, 0.975]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    cohort = read_ids(args.cohort.expanduser().resolve())
    benchmark_path = args.chemem_benchmark.expanduser().resolve()
    metrics_path = args.local_metrics.expanduser().resolve()
    manifest_path = args.local_manifest.expanduser().resolve()
    chemem = read_table(benchmark_path, "ChemEM benchmark")
    local = read_table(metrics_path, "Local V2 metrics")
    manifest = read_table(manifest_path, "Local manifest")
    for frame, path in ((chemem, benchmark_path), (local, metrics_path), (manifest, manifest_path)):
        frame["_table_path"] = str(path)
    chemem["chemem_top1_rmsd_A"] = pd.to_numeric(chemem["chemem_top1_rmsd_A"], errors="coerce")
    chemem["chemem_top1_iteration"] = pd.to_numeric(chemem["chemem_top1_iteration"], errors="coerce")
    local["top1_rmsd_A"] = pd.to_numeric(local["top1_rmsd_A"], errors="coerce")
    pair_ids = [
        case_id for case_id in cohort
        if case_id in set(chemem.loc[chemem["chemem_top1_rmsd_A"].notna(), "complex_name"])
        and case_id in set(local.loc[local["top1_rmsd_A"].notna(), "complex_name"])
    ]
    if len(pair_ids) != args.expected_paired_n:
        raise ValueError(f"Expected {args.expected_paired_n} ChemEM/Local pairs, found {len(pair_ids)}")

    chemem_index = chemem.set_index("complex_name")
    local_index = local.set_index("complex_name")
    manifest_index = manifest.set_index("complex_name")
    chemem_roots = [path.expanduser().resolve() for path in args.chemem_root]
    local_roots = [path.expanduser().resolve() for path in args.local_selected_root]
    absent_manifest = [case_id for case_id in pair_ids if case_id not in manifest_index.index]
    if absent_manifest:
        raise ValueError(
            "Paired cases absent from the density manifest: "
            f"{absent_manifest}"
        )

    # Resolve every required input before loading/scoring the first map.  This
    # makes missing completion-case crops or pose paths a preflight failure
    # instead of discarding work after dozens of completed pairs.
    resolved_inputs: dict[str, dict[str, object]] = {}
    resolution_errors: list[str] = []
    for case_id in pair_ids:
        c_record = chemem_index.loc[case_id]
        l_record = local_index.loc[case_id]
        m_record = manifest_index.loc[case_id]
        try:
            iteration = int(c_record["chemem_top1_iteration"])
            resolved_inputs[case_id] = {
                "iteration": iteration,
                "chemem_pose": chemem_pose(
                    c_record, chemem_roots, iteration, case_id
                ),
                "local_pose": local_pose(l_record, local_roots, case_id),
                "density_crop": density_crop(m_record, manifest_path, case_id),
            }
        except Exception as exc:  # report the complete path inventory at once
            resolution_errors.append(f"{case_id}: {type(exc).__name__}: {exc}")
    if resolution_errors:
        raise ValueError(
            "Ligand-map input preflight failed:\n" + "\n".join(resolution_errors)
        )
    print(
        f"Ligand-map input preflight OK: {len(resolved_inputs)}/{len(pair_ids)} "
        "paired cases have ChemEM pose, Local pose, and density crop",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for number, case_id in enumerate(pair_ids, start=1):
        c_record = chemem_index.loc[case_id]
        l_record = local_index.loc[case_id]
        resolved = resolved_inputs[case_id]
        iteration = int(resolved["iteration"])
        c_pose = Path(resolved["chemem_pose"])
        l_pose = Path(resolved["local_pose"])
        crop_path = Path(resolved["density_crop"])
        crop = load_cartesian_density_crop(crop_path)
        mask = fixed_crop_mask(crop, radius_A=args.mask_radius_A)
        c_score = score_pose(c_pose, crop, mask, args.sigma_A)
        l_score = score_pose(l_pose, crop, mask, args.sigma_A)
        if c_score["mask_voxels"] != l_score["mask_voxels"]:
            raise ValueError(f"Fixed-support mask mismatch for {case_id}")
        rows.append({
            "cohort_order": cohort.index(case_id),
            "complex_name": case_id,
            "density_crop": str(crop_path),
            "sigma_A": args.sigma_A,
            "mask_radius_A": args.mask_radius_A,
            "mask_voxels": c_score["mask_voxels"],
            "chemem_top1_iteration": iteration,
            "chemem_pose": str(c_pose),
            "chemem_top1_rmsd_A": float(c_record["chemem_top1_rmsd_A"]),
            "chemem_ncc": c_score["ncc"],
            "chemem_cosine": c_score["cosine"],
            "chemem_in_bounds_fraction": c_score["in_bounds_fraction"],
            "chemem_heavy_atoms": c_score["heavy_atoms"],
            "cryodiffdock_local_v2_pose": str(l_pose),
            "cryodiffdock_local_v2_top1_rmsd_A": float(l_record["top1_rmsd_A"]),
            "cryodiffdock_local_v2_ncc": l_score["ncc"],
            "cryodiffdock_local_v2_cosine": l_score["cosine"],
            "cryodiffdock_local_v2_in_bounds_fraction": l_score["in_bounds_fraction"],
            "cryodiffdock_local_v2_heavy_atoms": l_score["heavy_atoms"],
            "ncc_delta_local_minus_chemem": l_score["ncc"] - c_score["ncc"],
        })
        if number % 20 == 0 or number == len(pair_ids):
            print(f"Scored {number}/{len(pair_ids)} pairs", flush=True)

    per_case = pd.DataFrame(rows).sort_values("cohort_order")
    rng = np.random.default_rng(args.seed)
    summary_rows = []
    for method, column in (
        ("ChemEM Stage 1", "chemem_ncc"),
        ("CryoDiffDock-Local V2 selected", "cryodiffdock_local_v2_ncc"),
    ):
        values = per_case[column].to_numpy(float)
        low, high = bootstrap(values, args.bootstrap_repeats, rng)
        summary_rows.append({
            "method": method,
            "paired_n": len(values),
            "median_ncc": float(np.median(values)),
            "mean_ncc": float(np.mean(values)),
            "q1_ncc": float(np.quantile(values, 0.25)),
            "q3_ncc": float(np.quantile(values, 0.75)),
            "median_bootstrap_95ci_low": low,
            "median_bootstrap_95ci_high": high,
        })
    delta = per_case["ncc_delta_local_minus_chemem"].to_numpy(float)
    delta_low, delta_high = bootstrap(delta, args.bootstrap_repeats, rng)
    wilcoxon = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
    spearman = stats.spearmanr(per_case["chemem_ncc"], per_case["cryodiffdock_local_v2_ncc"])
    spearman_rho = getattr(spearman, "statistic", None)
    if spearman_rho is None:  # SciPy < 1.10 names this field correlation.
        spearman_rho = spearman.correlation
    statistics = pd.DataFrame([{
        "comparison": "CryoDiffDock-Local V2 selected minus ChemEM Stage 1",
        "paired_n": len(delta),
        "median_delta_ncc": float(np.median(delta)),
        "median_delta_bootstrap_95ci_low": delta_low,
        "median_delta_bootstrap_95ci_high": delta_high,
        "local_higher_ncc_count": int((delta > 0).sum()),
        "chemem_higher_ncc_count": int((delta < 0).sum()),
        "exact_tie_count": int((delta == 0).sum()),
        "paired_wilcoxon_statistic": float(wilcoxon.statistic),
        "paired_wilcoxon_pvalue": float(wilcoxon.pvalue),
        "paired_spearman_rho": float(spearman_rho),
        "paired_spearman_pvalue": float(spearman.pvalue),
    }])

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_case_path = output / "ligand_map_agreement_per_complex.csv"
    summary_path = output / "ligand_map_agreement_summary.csv"
    statistics_path = output / "ligand_map_agreement_statistics.csv"
    per_case.to_csv(per_case_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    statistics.to_csv(statistics_path, index=False)
    audit = {
        "paired_n": len(per_case),
        "cohort_n": len(cohort),
        "density_metric": "fixed-support Gaussian ligand-map NCC",
        "same_density_crop_for_both_methods": True,
        "native_pose_used_for_density_scoring": False,
        "sigma_A": args.sigma_A,
        "mask_radius_A": args.mask_radius_A,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
        "inputs": {
            "cohort": str(args.cohort.expanduser().resolve()),
            "chemem_benchmark": str(benchmark_path),
            "local_metrics": str(metrics_path),
            "local_manifest": str(manifest_path),
        },
        "outputs_sha256": {
            per_case_path.name: sha256(per_case_path),
            summary_path.name: sha256(summary_path),
            statistics_path.name: sha256(statistics_path),
        },
    }
    (output / "ligand_map_agreement_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(statistics.to_string(index=False))
    print(f"FIGURE2_LIGAND_MAP_AGREEMENT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
