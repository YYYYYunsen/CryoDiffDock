#!/usr/bin/env python3
"""Merge available cascade branches for a fixed parent cohort.

Unlike the strict full-run merger, this utility records missing branch outputs
instead of aborting.  A branch is computationally available only when every
requested condition has exactly one result directory.  The requested parent
cohort and all unavailable branches remain in the audit so downstream metrics
can retain the fixed parent denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path

import pandas as pd


RANKED_SDF = re.compile(
    r"^rank(?P<rank>\d+)_confidence(?P<confidence>[-+0-9.eE]+)\.sdf$"
)


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "complex"


def read_ids(path: Path) -> list[str]:
    values = [
        normalize_id(line.split()[0])
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return values


def validate_sdf(path: Path) -> bool:
    """Return whether RDKit can read one finite-coordinate molecule."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    molecule = next((item for item in supplier if item is not None), None)
    if molecule is None or molecule.GetNumConformers() == 0:
        return False
    positions = molecule.GetConformer().GetPositions()
    return all(math.isfinite(float(value)) for row in positions for value in row)


def ranked_pose_paths(directory: Path, limit: int) -> list[Path]:
    """Use exactly the same ranked-pose filename policy as NCC reranking."""
    paths: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = RANKED_SDF.match(path.name)
        if match and (limit <= 0 or int(match.group("rank")) <= limit):
            paths.append((int(match.group("rank")), path))
    return [path for _, path in sorted(paths)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--parent-cohort", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--conditions", default="baseline,guided")
    parser.add_argument("--expected-parents", type=int, default=174)
    parser.add_argument("--expected-branches-per-parent", type=int, default=5)
    parser.add_argument(
        "--expected-pose-files",
        type=int,
        default=40,
        help="Require this many rank*.sdf files per result directory; use 0 to skip.",
    )
    parser.add_argument(
        "--validate-sdf",
        action="store_true",
        help="Parse every pose with RDKit and reject branches containing unreadable/nonfinite SDFs.",
    )
    args = parser.parse_args()

    manifest_path = args.full_manifest.expanduser().resolve()
    shard_root = args.shard_root.expanduser().resolve()
    cohort_path = args.parent_cohort.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    if not conditions:
        raise ValueError("At least one condition is required")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")

    full = pd.read_csv(manifest_path, dtype=str)
    for column in ("complex_name", "parent_complex_name"):
        if column not in full.columns:
            raise ValueError(f"Full manifest lacks {column}")
    if full["complex_name"].isna().any() or full["complex_name"].duplicated().any():
        raise ValueError("Full-manifest complex_name values must be nonempty and unique")
    full["parent_complex_name"] = full["parent_complex_name"].map(normalize_id)

    requested_parents = read_ids(cohort_path)
    if len(requested_parents) != args.expected_parents:
        raise ValueError(
            f"Expected {args.expected_parents} parent IDs, found {len(requested_parents)}"
        )
    requested_set = set(requested_parents)
    available_parent_rows = set(full["parent_complex_name"])
    missing_parents = sorted(requested_set - available_parent_rows)
    if missing_parents:
        raise ValueError(f"Requested parents absent from full manifest: {missing_parents}")

    requested = full.loc[full["parent_complex_name"].isin(requested_set)].copy()
    requested["source_full_row_index"] = requested.index.astype(int)
    branch_counts = requested.groupby("parent_complex_name").size()
    bad_branch_counts = branch_counts[branch_counts.ne(args.expected_branches_per_parent)]
    if len(bad_branch_counts):
        raise ValueError(
            "Unexpected requested branch counts: "
            f"{bad_branch_counts.sort_index().to_dict()}"
        )
    expected_branch_n = args.expected_parents * args.expected_branches_per_parent
    if len(requested) != expected_branch_n:
        raise RuntimeError(f"Expected {expected_branch_n} branches, found {len(requested)}")

    locations: dict[str, tuple[int, int]] = {}
    for shard_index in range(args.num_shards):
        shard_manifest_path = shard_root / f"shard{shard_index}" / "manifest.csv"
        shard = pd.read_csv(shard_manifest_path, dtype=str)
        if "complex_name" not in shard.columns or shard["complex_name"].duplicated().any():
            raise ValueError(f"Invalid shard manifest: {shard_manifest_path}")
        for local_index, name in enumerate(shard["complex_name"].astype(str)):
            if name in locations:
                raise ValueError(f"Branch occurs in multiple shards: {name}")
            locations[name] = (shard_index, local_index)

    audit_rows: list[dict] = []
    available_records: list[dict] = []
    available_sources: list[dict[str, Path]] = []
    for _, row in requested.iterrows():
        name = str(row["complex_name"])
        location = locations.get(name)
        record = row.to_dict()
        record.update({"shard_index": None, "shard_local_index": None})
        source_by_condition: dict[str, Path] = {}
        problems: list[str] = []
        if location is None:
            problems.append("missing_shard_manifest_row")
        else:
            shard_index, local_index = location
            record.update({"shard_index": shard_index, "shard_local_index": local_index})
            for condition in conditions:
                parent = shard_root / f"shard{shard_index}" / "results" / condition
                matches = sorted(parent.glob(f"index{local_index}___*"))
                record[f"{condition}_result_directory_count"] = len(matches)
                if len(matches) != 1:
                    problems.append(f"{condition}_result_directory_count_{len(matches)}")
                    continue
                pose_paths = ranked_pose_paths(matches[0], args.expected_pose_files)
                pose_count = len(pose_paths)
                record[f"{condition}_pose_file_count"] = pose_count
                if args.expected_pose_files and pose_count != args.expected_pose_files:
                    problems.append(f"{condition}_pose_file_count_{pose_count}")
                    continue
                if args.validate_sdf:
                    invalid_pose = next((path for path in pose_paths if not validate_sdf(path)), None)
                    if invalid_pose is not None:
                        problems.append(f"{condition}_invalid_sdf_{invalid_pose.name}")
                        continue
                source_by_condition[condition] = matches[0].resolve()

        record["computationally_available"] = not problems
        record["unavailable_reason"] = ";".join(problems)
        audit_rows.append(record)
        if not problems:
            available_records.append(row.to_dict())
            available_sources.append(source_by_condition)

    reason_counts = Counter(
        reason
        for row in audit_rows
        for reason in str(row["unavailable_reason"]).split(";")
        if reason
    )
    print(
        "CASCADE MERGE PREFLIGHT: "
        f"requested={len(requested)}, available={len(available_records)}, "
        f"unavailable={len(requested) - len(available_records)}"
    )
    if reason_counts:
        print("UNAVAILABLE REASON COUNTS")
        for reason, count in reason_counts.most_common():
            print(f"{count}\t{reason}")
    if not available_records:
        raise RuntimeError(
            "No computationally available branches passed preflight; "
            f"reason_counts={dict(reason_counts)}"
        )

    available = pd.DataFrame(available_records, columns=requested.columns).reset_index(drop=True)
    filtered_evaluation: pd.DataFrame | None = None
    if args.evaluation_manifest is not None:
        evaluation_path = args.evaluation_manifest.expanduser().resolve()
        if not evaluation_path.is_file():
            raise FileNotFoundError(
                f"Evaluation manifest does not exist: {evaluation_path}. "
                "Create the row-aligned reveal manifest before merging."
            )
        evaluation = pd.read_csv(evaluation_path, dtype=str)
        if "complex_name" not in evaluation.columns or evaluation["complex_name"].duplicated().any():
            raise ValueError("Evaluation manifest complex_name values must be unique")
        if "reference_ligand" not in evaluation.columns:
            raise ValueError("Evaluation manifest lacks reference_ligand")
        filtered_evaluation = available[["complex_name"]].merge(
            evaluation,
            on="complex_name",
            how="left",
            validate="one_to_one",
        )
        if len(filtered_evaluation) != len(available):
            raise RuntimeError("Evaluation-manifest filtering changed the branch count")
        if filtered_evaluation["reference_ligand"].isna().any():
            missing_names = filtered_evaluation.loc[
                filtered_evaluation["reference_ligand"].isna(), "complex_name"
            ].tolist()
            raise RuntimeError(
                f"Available branches lack reveal references: {missing_names[:20]}"
            )

    # All input validation above is complete.  Only now create output files and
    # links, so a missing reveal manifest cannot leave a partial merge behind.
    output_root.mkdir(parents=True, exist_ok=False)
    available.to_csv(output_root / "inference_blind_available.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_root / "requested_branch_audit.csv", index=False)
    unavailable = audit.loc[~audit["computationally_available"]].copy()
    unavailable.to_csv(output_root / "unavailable_branches.csv", index=False)
    Path(output_root / "requested_parent_ids.txt").write_text(
        "".join(f"{pdb_id}\n" for pdb_id in requested_parents)
    )

    for available_index, (row, sources) in enumerate(zip(available_records, available_sources)):
        name = str(row["complex_name"])
        for condition, source in sources.items():
            target_parent = output_root / condition
            target_parent.mkdir(parents=True, exist_ok=True)
            target = target_parent / f"index{available_index}___{safe_name(name)}"
            target.symlink_to(source, target_is_directory=True)

    if filtered_evaluation is not None:
        filtered_evaluation.to_csv(
            output_root / "evaluation_manifest_available.csv", index=False
        )

    incomplete_parent_ids = sorted(set(unavailable["parent_complex_name"].astype(str)))
    summary = {
        "full_manifest": str(manifest_path),
        "parent_cohort": str(cohort_path),
        "requested_parent_n": len(requested_parents),
        "branches_per_parent": args.expected_branches_per_parent,
        "requested_branch_n": len(requested),
        "available_branch_n": len(available),
        "unavailable_branch_n": len(unavailable),
        "parents_with_any_unavailable_branch_n": len(incomplete_parent_ids),
        "parents_with_any_unavailable_branch": incomplete_parent_ids,
        "conditions_required": conditions,
        "expected_pose_files_per_condition": args.expected_pose_files,
        "sdf_semantic_validation": bool(args.validate_sdf),
        "fixed_parent_denominator_policy": (
            "All requested parents remain in downstream metrics; unavailable branches "
            "are computationally skipped and counted as failures where required."
        ),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"CASCADE_AVAILABLE_MERGE_OK: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
