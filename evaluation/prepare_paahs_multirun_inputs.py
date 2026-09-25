#!/usr/bin/env python3
"""Prepare blind 173-case PAAHS inputs from completed multirun inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROTEIN_COLUMNS = ("protein_path", "experimental_protein")
DENSITY_COLUMNS = ("density_crop", "ligand_density_crop")


def choose(columns, candidates, label):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not resolve {label}; columns={list(columns)}")


def normalize(frame):
    frame = frame.copy()
    frame["complex_name"] = frame["complex_name"].astype(str).str.strip().str.upper()
    return frame


def existing_file(candidates, label: str) -> Path:
    """Return the first existing file while retaining useful failure context."""
    attempted = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(str(candidate)).expanduser().resolve()
        attempted.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not resolve {label}; tried: " + ", ".join(attempted)
    )


def resolve_manifest(summary: dict, replicate_root: Path) -> Path:
    return existing_file(
        (
            summary.get("input_manifest"),
            replicate_root / "input_manifest.csv",
            replicate_root / "evaluation_manifest.csv",
        ),
        f"input manifest for {replicate_root.name}",
    )


def resolve_pose_scores(summary: dict, replicate_root: Path) -> Path:
    """Support both independently run and provenance-reused replicates."""
    linked_ncc = summary.get("linked_inputs", {}).get("ncc")
    return existing_file(
        (
            summary.get("ncc_pose_scores"),
            summary.get("pose_scores"),
            Path(str(linked_ncc)) / "pose_scores.csv" if linked_ncc else None,
            replicate_root / "02_ncc" / "pose_scores.csv",
        ),
        f"NCC pose scores for {replicate_root.name}",
    )


def locate_rankings(root: Path) -> list[Path]:
    preferred = root / "03_v2_top1" / "v2_selected"
    paths = sorted(preferred.glob("index*___*/ranking.csv"))
    if paths:
        return paths
    paths = sorted(root.glob("**/v2_selected/index*___*/ranking.csv"))
    if not paths:
        raise FileNotFoundError(f"No V2 ranking files below {root}")
    return paths


def ranking_top1(path: Path) -> pd.Series:
    frame = pd.read_csv(path, dtype={"complex_name": str})
    if "materialized_rank" in frame:
        frame = frame[pd.to_numeric(frame["materialized_rank"], errors="coerce").eq(1)]
    if len(frame) != 1:
        raise ValueError(f"Expected one Top-1 row: {path}")
    return frame.iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--exclude-case", action="append", default=["9O93"])
    parser.add_argument("--expected-cases", type=int, default=173)
    args = parser.parse_args()

    budget_root = args.budget_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    excluded = {str(value).strip().upper() for value in args.exclude_case}
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = sorted(budget_root.glob("replicate*_seed*/replicate_summary.json"))
    if len(summaries) != 8:
        raise ValueError(f"Expected 8 replicate summaries, found {len(summaries)}")

    run_rows = []
    for order, summary_path in enumerate(summaries, start=1):
        replicate_root = summary_path.parent.resolve()
        summary = json.loads(summary_path.read_text())
        manifest_path = resolve_manifest(summary, replicate_root)
        scores_path = resolve_pose_scores(summary, replicate_root)

        manifest = normalize(pd.read_csv(manifest_path, dtype={"complex_name": str}))
        scores = normalize(pd.read_csv(scores_path, dtype={"complex_name": str}))
        manifest = manifest[~manifest["complex_name"].isin(excluded)].copy()
        scores = scores[scores["complex_name"].isin(set(manifest["complex_name"]))].copy()
        if len(manifest) != args.expected_cases or manifest["complex_name"].duplicated().any():
            raise ValueError(f"{replicate_root.name}: invalid filtered manifest")
        score_counts = scores.groupby("complex_name").size()
        if len(score_counts) != args.expected_cases or not score_counts.eq(80).all():
            raise ValueError(f"{replicate_root.name}: expected 80 scored poses per case")

        protein_column = choose(manifest.columns, PROTEIN_COLUMNS, "protein path")
        density_column = choose(manifest.columns, DENSITY_COLUMNS, "density crop")
        manifest = manifest.reset_index(drop=True)
        manifest["case_order"] = range(len(manifest))

        rankings = locate_rankings(replicate_root)
        pose_map = {}
        ranking_map = {}
        for ranking_path in rankings:
            row = ranking_top1(ranking_path)
            name = str(row["complex_name"]).strip().upper()
            if name in excluded:
                continue
            pose = Path(str(row["materialized_path"])).expanduser().resolve()
            if not pose.is_file():
                raise FileNotFoundError(pose)
            if name in pose_map:
                raise ValueError(f"Duplicate ranking for {name} in {replicate_root}")
            pose_map[name] = pose
            ranking_map[name] = ranking_path.resolve()
        names = manifest["complex_name"].tolist()
        if set(pose_map) != set(names):
            raise ValueError(
                f"{replicate_root.name}: ranking mismatch; "
                f"missing={sorted(set(names)-set(pose_map))[:10]}"
            )

        baseline_rows = []
        for row in manifest.itertuples(index=False):
            name = str(row.complex_name)
            protein = Path(str(getattr(row, protein_column))).expanduser().resolve()
            density = Path(str(getattr(row, density_column))).expanduser().resolve()
            for path in (protein, density, pose_map[name]):
                if not path.is_file():
                    raise FileNotFoundError(path)
            baseline_rows.append({
                "case_order": int(row.case_order),
                "complex_name": name,
                "baseline_pose_path": str(pose_map[name]),
                "protein_path": str(protein),
                "density_crop": str(density),
            })

        run_dir = output_root / f"replicate{order:02d}"
        run_dir.mkdir()
        pd.DataFrame(baseline_rows).to_csv(run_dir / "baseline_manifest.csv", index=False)
        manifest.to_csv(run_dir / "search_manifest.csv", index=False)
        scores.to_csv(run_dir / "pose_scores.csv", index=False)
        pd.DataFrame([
            {"complex_name": name, "ranking_path": str(ranking_map[name])}
            for name in names
        ]).to_csv(run_dir / "original_rankings.csv", index=False)
        metadata = {
            "replicate_order": order,
            "source_replicate_root": str(replicate_root),
            "source_summary": str(summary_path.resolve()),
            "source_seed": summary.get("seed"),
            "source_manifest": str(manifest_path),
            "source_pose_scores": str(scores_path),
            "excluded_cases": sorted(excluded),
            "cases_n": len(names),
            "poses_per_case": 80,
            "uses_reference_or_native_rmsd": False,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        run_rows.append(metadata)
        print(f"Prepared replicate {order}: {run_dir}", flush=True)

    pd.DataFrame(run_rows).to_csv(output_root / "replicate_manifest.csv", index=False)
    print(f"PAAHS_MULTIRUN_INPUTS_OK: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
