from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


AUDIT_NAME = "density_particle_resampling_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate density particle-resampling ESS and genealogy audits."
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def _complex_name(result_dir: Path) -> str:
    return result_dir.name.split("___", 1)[-1]


def summarize_one(path: Path) -> Mapping[str, Any]:
    audit = json.loads(path.read_text())
    events = audit.get("events", [])
    resampled = [row for row in events if row.get("resampled")]
    ess_fractions = [float(row["ess_fraction"]) for row in events]
    selected_parent_counts = [
        int(row["num_active_parents_selected"]) for row in resampled
    ]
    max_offspring = [int(row["max_parent_offspring"]) for row in resampled]
    result_dir = path.parent
    return {
        "result_directory": str(result_dir.resolve()),
        "complex_name": _complex_name(result_dir),
        "num_particles": int(audit["num_particles"]),
        "num_protected": int(audit["num_protected"]),
        "num_active": int(audit["num_active"]),
        "num_checkpoint_events": int(audit["num_checkpoint_events"]),
        "num_resampling_events": int(audit["num_resampling_events"]),
        "min_ess_fraction": min(ess_fractions) if ess_fractions else np.nan,
        "mean_ess_fraction": float(np.mean(ess_fractions)) if ess_fractions else np.nan,
        "min_selected_active_parents": min(selected_parent_counts)
        if selected_parent_counts else int(audit["num_active"]),
        "max_parent_offspring": max(max_offspring) if max_offspring else 1,
        "final_unique_ancestors": int(audit["final_unique_ancestors"]),
        "final_unique_active_ancestors": int(audit["final_unique_active_ancestors"]),
        "final_max_family_fraction": float(audit["final_max_family_fraction"]),
        "final_max_active_family_fraction": float(
            audit["final_max_active_family_fraction"]
        ),
        "checkpoint_reasons": ";".join(str(row.get("reason", "")) for row in events),
    }


def main() -> int:
    args = parse_args()
    results = args.results.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(results.glob(f"index*___*/{AUDIT_NAME}"))
    rows = []
    failures = []
    for path in paths:
        try:
            rows.append(summarize_one(path))
        except Exception as exc:  # per-complex batch reporting
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_on_error:
                raise
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_complex.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    if frame.empty:
        summary = {
            "results": str(results),
            "num_complexes": 0,
            "num_failures": len(failures),
        }
    else:
        summary = {
            "results": str(results),
            "num_complexes": int(len(frame)),
            "num_failures": len(failures),
            "complexes_with_resampling_pct": float(
                100.0 * (frame["num_resampling_events"] > 0).mean()
            ),
            "mean_num_resampling_events": float(frame["num_resampling_events"].mean()),
            "median_min_ess_fraction": float(frame["min_ess_fraction"].median()),
            "median_final_unique_active_ancestors": float(
                frame["final_unique_active_ancestors"].median()
            ),
            "median_final_max_active_family_fraction": float(
                frame["final_max_active_family_fraction"].median()
            ),
            "max_final_active_family_fraction": float(
                frame["final_max_active_family_fraction"].max()
            ),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not frame.empty:
        print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))
    if args.fail_on_error and (failures or not rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
