#!/usr/bin/env python3
"""Summarize rank-1 and Top-K multi-start cascade accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize(value: object) -> str:
    return str(value).strip().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-manifest", required=True, type=Path)
    parser.add_argument("--branch-metrics", required=True, type=Path)
    parser.add_argument("--requested-cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--local-method-name", default="Local V2")
    args = parser.parse_args()

    manifest = pd.read_csv(args.branch_manifest, dtype=str)
    metrics = pd.read_csv(args.branch_metrics, dtype=str)
    manifest["row_index"] = np.arange(len(manifest))
    for column in ("row_index", "top1_rmsd_A"):
        if column not in metrics.columns:
            raise ValueError(f"Branch metrics lacks {column!r}")
    metrics["row_index"] = pd.to_numeric(metrics["row_index"], errors="raise").astype(int)
    metrics["top1_rmsd_A"] = pd.to_numeric(metrics["top1_rmsd_A"], errors="coerce")
    merged = manifest.merge(
        metrics[["row_index", "top1_rmsd_A"]], on="row_index", how="left", validate="one_to_one"
    )
    merged["parent_complex_name"] = merged["parent_complex_name"].map(normalize)
    merged["global_rank"] = pd.to_numeric(merged["global_rank"], errors="raise").astype(int)
    requested = [
        normalize(line.split()[0])
        for line in args.requested_cohort.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(requested) != len(set(requested)):
        raise ValueError("Requested cohort contains duplicate IDs")

    rows = []
    for pdb_id in requested:
        group = merged[merged["parent_complex_name"].eq(pdb_id)].copy()
        group = group[np.isfinite(group["top1_rmsd_A"])]
        rank1 = group[group["global_rank"].eq(1)]
        complete_topk = set(group["global_rank"]) >= set(range(1, args.top_k + 1))
        top1 = float(rank1.iloc[0]["top1_rmsd_A"]) if len(rank1) == 1 else np.nan
        best = (
            float(group[group["global_rank"].le(args.top_k)]["top1_rmsd_A"].min())
            if complete_topk else np.nan
        )
        best_rank = (
            int(group.loc[group["top1_rmsd_A"].idxmin(), "global_rank"])
            if complete_topk else np.nan
        )
        rows.append(
            {
                "pdb_id": pdb_id,
                "top1_cascade_rmsd_A": top1,
                f"best_of_{args.top_k}_cascade_rmsd_A": best,
                f"best_of_{args.top_k}_global_rank": best_rank,
                "available_branch_count": int(len(group)),
                "complete_top_k": bool(complete_topk),
            }
        )
    per_complex = pd.DataFrame(rows)
    denominator = len(per_complex)

    summary: dict[str, object] = {
        "requested_cohort_n": denominator,
        "top_k": args.top_k,
        "top1_scored_n": int(per_complex["top1_cascade_rmsd_A"].notna().sum()),
        "complete_top_k_n": int(per_complex["complete_top_k"].sum()),
        "missing_counts_as_failure": True,
        "top1_definition": (
            f"{args.local_method_name} selected pose launched from Global rank-1 centre"
        ),
        "best_of_k_definition": (
            f"minimum RMSD among one {args.local_method_name} selected output from each "
            "of the first K Global centres; retrospective candidate coverage, not "
            "operational Top-1"
        ),
    }
    endpoints = {
        "top1": "top1_cascade_rmsd_A",
        f"best_of_{args.top_k}": f"best_of_{args.top_k}_cascade_rmsd_A",
    }
    metric_rows = []
    for endpoint, column in endpoints.items():
        values = per_complex[column]
        finite = values[np.isfinite(values)]
        summary[f"{endpoint}_median_rmsd_A_among_scored"] = (
            float(finite.median()) if len(finite) else None
        )
        for cutoff in (2.0, 5.0):
            hits = int((values < cutoff).fillna(False).sum())
            pct = 100.0 * hits / denominator
            summary[f"{endpoint}_lt{cutoff:g}A_count"] = hits
            summary[f"{endpoint}_lt{cutoff:g}A_pct"] = pct
            metric_rows.append(
                {
                    "endpoint": endpoint,
                    "cutoff_A": cutoff,
                    "success_count": hits,
                    "denominator": denominator,
                    "success_pct": pct,
                }
            )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_complex.to_csv(output / "cascade_per_complex.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "cascade_success_rates.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"CASCADE_SUMMARY_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
