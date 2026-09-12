#!/usr/bin/env python3
"""Assemble row-aligned authoritative inputs for the V2 Top-1 timing rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def read_ids(path: Path) -> list[str]:
    values = [line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines()]
    values = [value for value in values if value]
    if len(values) != 174 or len(set(values)) != 174:
        raise ValueError(f"Expected 174 unique authoritative IDs, found {len(values)}")
    return values


def table(path: Path, label: str) -> pd.DataFrame:
    result = pd.read_csv(path)
    if "complex_name" not in result:
        raise ValueError(f"{label} lacks complex_name: {path}")
    result = result.copy()
    result["complex_name"] = result["complex_name"].astype(str).str.strip().str.upper()
    return result


def manifest_index(frame: pd.DataFrame, label: str) -> dict[str, pd.Series]:
    if frame["complex_name"].duplicated().any():
        duplicate = frame.loc[frame["complex_name"].duplicated(False), "complex_name"].unique()
        raise ValueError(f"Duplicate {label} manifest IDs: {duplicate[:20].tolist()}")
    return {str(row["complex_name"]): row for _, row in frame.iterrows()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-ids", required=True, type=Path)
    parser.add_argument("--primary-manifest", required=True, type=Path)
    parser.add_argument("--primary-scores", required=True, type=Path)
    parser.add_argument("--completion-manifest", required=True, type=Path)
    parser.add_argument("--completion-scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-runnable-n", type=int, default=173)
    args = parser.parse_args()

    source_paths = {
        "authoritative_ids": args.authoritative_ids.expanduser().resolve(),
        "primary_manifest": args.primary_manifest.expanduser().resolve(),
        "primary_scores": args.primary_scores.expanduser().resolve(),
        "completion_manifest": args.completion_manifest.expanduser().resolve(),
        "completion_scores": args.completion_scores.expanduser().resolve(),
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    ids = read_ids(source_paths["authoritative_ids"])
    manifests = {
        "original_289": table(source_paths["primary_manifest"], "primary"),
        "completion_numericfix_v3000_12": table(
            source_paths["completion_manifest"], "completion"
        ),
    }
    scores = {
        "original_289": table(source_paths["primary_scores"], "primary scores"),
        "completion_numericfix_v3000_12": table(
            source_paths["completion_scores"], "completion scores"
        ),
    }
    manifest_rows = {
        label: manifest_index(frame, label) for label, frame in manifests.items()
    }
    score_groups = {
        label: {case_id: group.copy() for case_id, group in frame.groupby("complex_name")}
        for label, frame in scores.items()
    }

    output_manifest_rows: list[dict[str, object]] = []
    output_score_frames: list[pd.DataFrame] = []
    selected_sources: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    source_priority = ("original_289", "completion_numericfix_v3000_12")

    for authoritative_index, case_id in enumerate(ids):
        counts = {
            label: len(score_groups[label].get(case_id, ())) for label in source_priority
        }
        chosen = next(
            (
                label for label in source_priority
                if counts[label] == 80 and case_id in manifest_rows[label]
            ),
            None,
        )
        if chosen is None:
            excluded.append({
                "authoritative_index": authoritative_index,
                "complex_name": case_id,
                "reason": "no source with exactly 80 scored inference poses",
                "primary_pose_rows": counts["original_289"],
                "completion_pose_rows": counts["completion_numericfix_v3000_12"],
            })
            continue

        new_row_index = len(output_manifest_rows)
        manifest_row = manifest_rows[chosen][case_id].to_dict()
        manifest_row["complex_name"] = case_id
        output_manifest_rows.append(manifest_row)

        score_frame = score_groups[chosen][case_id].copy()
        score_frame["row_index"] = new_row_index
        score_frame["complex_name"] = case_id
        output_score_frames.append(score_frame)
        selected_sources.append({
            "authoritative_index": authoritative_index,
            "runtime_row_index": new_row_index,
            "complex_name": case_id,
            "input_source": chosen,
            "pose_score_rows": len(score_frame),
        })

    if len(output_manifest_rows) != args.expected_runnable_n:
        raise ValueError(
            f"Expected {args.expected_runnable_n} runnable cases, found "
            f"{len(output_manifest_rows)}; excluded={excluded}"
        )
    if len(excluded) != 174 - args.expected_runnable_n:
        raise ValueError(f"Unexpected excluded-case count: {excluded}")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_output = output / "v2_top1_runtime_manifest.csv"
    scores_output = output / "v2_top1_runtime_pose_scores.csv"
    ids_output = output / "v2_top1_runtime_ids.txt"
    source_output = output / "v2_top1_runtime_input_sources.csv"
    excluded_output = output / "v2_top1_runtime_excluded_cases.csv"

    manifest_frame = pd.DataFrame(output_manifest_rows)
    scores_frame = pd.concat(output_score_frames, ignore_index=True, sort=False)
    manifest_frame.to_csv(manifest_output, index=False)
    scores_frame.to_csv(scores_output, index=False)
    ids_output.write_text("".join(f"{row['complex_name']}\n" for row in selected_sources))
    pd.DataFrame(selected_sources).to_csv(source_output, index=False)
    pd.DataFrame(excluded).to_csv(excluded_output, index=False)

    counts = scores_frame.groupby("complex_name").size()
    if len(manifest_frame) != args.expected_runnable_n or counts.ne(80).any():
        raise RuntimeError("Written manifest/pose-score alignment failed")
    expected_indices = set(range(args.expected_runnable_n))
    observed_indices = set(pd.to_numeric(scores_frame["row_index"], errors="raise").astype(int))
    if observed_indices != expected_indices:
        raise RuntimeError("Written pose-score row indices do not cover the new manifest")

    audit = {
        "authoritative_n": len(ids),
        "runnable_n": len(manifest_frame),
        "excluded_n": len(excluded),
        "excluded_cases": excluded,
        "pose_rows_per_runnable_case": 80,
        "input_source_counts": pd.DataFrame(selected_sources)["input_source"].value_counts().to_dict(),
        "selection_rule": "prefer original_289 when it has exactly 80 poses; otherwise use completion_numericfix_v3000_12",
        "outputs": {
            "manifest": str(manifest_output),
            "pose_scores": str(scores_output),
            "ids": str(ids_output),
            "sources": str(source_output),
            "excluded": str(excluded_output),
        },
        "input_sha256": {label: sha256(path) for label, path in source_paths.items()},
    }
    (output / "v2_top1_runtime_input_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))
    print(f"FIGURE2_V2_TOP1_RUNTIME_INPUTS_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
