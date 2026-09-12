#!/usr/bin/env python3
"""Unify V2 source manifests/scores for a frozen Figure 3 ablation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PATH_COLUMNS = {
    "experimental_protein",
    "ligand",
    "reference_ligand",
    "map_path",
    "mask_structure",
    "ligand_density_crop",
    "prediction_path",
    "crop_path",
}


def normalize(value: object) -> str:
    return str(value).strip().upper()


def read_ids(path: Path) -> list[str]:
    values = [normalize(line) for line in path.read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Cohort must contain unique non-empty IDs: {path}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute_paths(frame: pd.DataFrame, base: Path) -> pd.DataFrame:
    frame = frame.copy()
    for column in PATH_COLUMNS & set(frame.columns):
        def resolve(value: object) -> object:
            if pd.isna(value) or not str(value).strip():
                return value
            path = Path(str(value)).expanduser()
            return str((path if path.is_absolute() else base / path).resolve())

        frame[column] = frame[column].map(resolve)
    return frame


def reveal_candidates(summary_path: Path, summary: dict, manifest_path: Path) -> list[Path]:
    search_root = summary_path.parent
    candidates = [
        search_root / "evaluation_manifest_reveal.csv",
        manifest_path.parent / "inference_reveal.csv",
        manifest_path.parent / "inference.csv",
    ]
    evaluation = summary.get("evaluation_manifest")
    if evaluation:
        evaluation_path = Path(str(evaluation)).expanduser()
        candidates.append(evaluation_path)
    return [path.resolve() for path in candidates]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument(
        "--source-summary",
        required=True,
        action="append",
        type=Path,
        help="V2 summary.json; later entries take priority for overlapping IDs",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    cohort_path = args.cohort.expanduser().resolve()
    cohort = read_ids(cohort_path)
    cohort_set = set(cohort)
    sources: list[dict] = []
    reveal_lookup: dict[str, str] = {}

    for priority, raw_summary in enumerate(args.source_summary):
        summary_path = raw_summary.expanduser().resolve()
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        manifest_path = Path(str(summary["manifest"])).expanduser().resolve()
        scores_path = Path(str(summary["scores"])).expanduser().resolve()
        if not manifest_path.is_file() or not scores_path.is_file():
            raise FileNotFoundError(
                f"Source inputs missing for {summary_path}: {manifest_path}, {scores_path}"
            )

        manifest = pd.read_csv(manifest_path, dtype={"complex_name": str})
        scores = pd.read_csv(scores_path, dtype={"complex_name": str})
        if "complex_name" not in manifest or "complex_name" not in scores:
            raise ValueError(f"Missing complex_name in source: {summary_path}")
        manifest["_normalized_id"] = manifest["complex_name"].map(normalize)
        scores["_normalized_id"] = scores["complex_name"].map(normalize)
        if manifest["_normalized_id"].duplicated().any():
            raise ValueError(f"Duplicate manifest IDs: {manifest_path}")
        scores["row_index"] = pd.to_numeric(scores["row_index"], errors="raise").astype(int)

        reveal_path = None
        for candidate in reveal_candidates(summary_path, summary, manifest_path):
            if not candidate.is_file():
                continue
            table = pd.read_csv(candidate, dtype={"complex_name": str})
            if {"complex_name", "reference_ligand"} <= set(table.columns):
                table["_normalized_id"] = table["complex_name"].map(normalize)
                for _, row in table.iterrows():
                    name = row["_normalized_id"]
                    if name in cohort_set and pd.notna(row["reference_ligand"]):
                        reference = Path(str(row["reference_ligand"])).expanduser()
                        if not reference.is_absolute():
                            reference = candidate.parent / reference
                        reveal_lookup[name] = str(reference.resolve())
                reveal_path = candidate
                break

        sources.append({
            "priority": priority,
            "summary_path": summary_path,
            "manifest_path": manifest_path,
            "scores_path": scores_path,
            "reveal_path": reveal_path,
            "manifest": manifest,
            "scores": scores,
            "manifest_ids": set(manifest["_normalized_id"]),
        })

    selected_sources: dict[str, dict] = {}
    for source in sources:
        for name in cohort_set & source["manifest_ids"]:
            source_rows = source["manifest"].index[
                source["manifest"]["_normalized_id"].eq(name)
            ].tolist()
            if len(source_rows) != 1:
                raise AssertionError(f"Ambiguous manifest row for {name}")
            source_row = int(source_rows[0])
            score_count = int((source["scores"]["row_index"] == source_row).sum())
            if score_count:
                selected_sources[name] = {**source, "source_row": source_row}

    missing = [name for name in cohort if name not in selected_sources]
    if missing:
        raise ValueError(f"Cohort IDs absent from every source manifest/scores pair: {missing}")
    missing_references = [name for name in cohort if name not in reveal_lookup]
    if missing_references:
        raise ValueError(f"Reference ligands missing for cohort IDs: {missing_references}")

    manifest_rows = []
    score_frames = []
    audit_rows = []
    for new_index, name in enumerate(cohort):
        source = selected_sources[name]
        source_row = source["source_row"]
        manifest_row = source["manifest"].loc[[source_row]].drop(
            columns=["_normalized_id", "reference_ligand"], errors="ignore"
        )
        manifest_row = absolute_paths(manifest_row, source["manifest_path"].parent)
        manifest_row.loc[:, "complex_name"] = name
        manifest_rows.append(manifest_row)

        score_rows = source["scores"].loc[
            source["scores"]["row_index"].eq(source_row)
        ].drop(columns=["_normalized_id"], errors="ignore")
        score_rows = absolute_paths(score_rows, source["scores_path"].parent)
        score_rows.loc[:, "row_index"] = new_index
        score_rows.loc[:, "complex_name"] = name
        score_frames.append(score_rows)
        audit_rows.append({
            "row_index": new_index,
            "complex_name": name,
            "source_summary": str(source["summary_path"]),
            "source_manifest": str(source["manifest_path"]),
            "source_scores": str(source["scores_path"]),
            "source_row_index": source_row,
            "num_pose_scores": len(score_rows),
        })

    blind = pd.concat(manifest_rows, ignore_index=True, sort=False)
    scores = pd.concat(score_frames, ignore_index=True, sort=False)
    reveal = blind.copy()
    reveal["reference_ligand"] = reveal["complex_name"].map(reveal_lookup)
    if blind["complex_name"].map(normalize).tolist() != cohort:
        raise AssertionError("Unified manifest order differs from cohort")
    if set(scores["row_index"].unique()) != set(range(len(cohort))):
        raise AssertionError("Unified scores do not cover every manifest row")

    output = args.output_root.expanduser().resolve()
    prepared = output / "02_prepared"
    ncc = output / "04_ncc"
    prepared.mkdir(parents=True, exist_ok=True)
    ncc.mkdir(parents=True, exist_ok=True)
    blind_path = prepared / "inference_blind.csv"
    reveal_path = prepared / "inference.csv"
    scores_path = ncc / "pose_scores.csv"
    blind.to_csv(blind_path, index=False)
    reveal.to_csv(reveal_path, index=False)
    scores.to_csv(scores_path, index=False)
    pd.DataFrame(audit_rows).to_csv(output / "source_audit.csv", index=False)

    source_counts = pd.Series(
        [row["source_summary"] for row in audit_rows]
    ).value_counts().to_dict()
    record = {
        "cohort": str(cohort_path),
        "cohort_n": len(cohort),
        "cohort_sha256": sha256(cohort_path),
        "source_priority": [str(item["summary_path"]) for item in sources],
        "selected_complex_counts_by_source": {
            str(key): int(value) for key, value in source_counts.items()
        },
        "output_manifest": str(blind_path),
        "output_scores": str(scores_path),
        "output_manifest_sha256": sha256(blind_path),
        "output_scores_sha256": sha256(scores_path),
    }
    (output / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"FIGURE3_ABLATION_UNIFIED_INPUTS_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
