#!/usr/bin/env python3
"""Write the frozen native-V2 ablation manifest for Figure 3."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-main-root", required=True, type=Path)
    parser.add_argument("--ablations-root", required=True, type=Path)
    parser.add_argument("--hardware-id", default="")
    parser.add_argument("--process-count", type=int, default=1)
    parser.add_argument("--timing-boundary", default="native search/scoring/selection")
    parser.add_argument("--runtime-column", default="selection_ready_seconds")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.process_count <= 0:
        raise ValueError("process-count must be positive")

    main_root = args.v2_main_root.expanduser().resolve()
    ablations = args.ablations_root.expanduser().resolve()

    columns = [
        "experiment_id", "display_label", "panel", "method", "selected_metrics",
        "oracle_metrics", "audit", "summary", "selected_column", "oracle_column",
        "display_order", "match_group", "budget_type", "budget_value", "ablation_key",
        "hardware_id", "process_count", "timing_boundary", "runtime_column",
    ]
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {column: "" for column in columns}
        row.update(values)
        row["selected_column"] = row["selected_column"] or "top1_rmsd_A"
        row["oracle_column"] = row["oracle_column"] or "best_of_n_rmsd_A"
        row["hardware_id"] = row["hardware_id"] or args.hardware_id
        row["process_count"] = row["process_count"] or args.process_count
        row["timing_boundary"] = row["timing_boundary"] or args.timing_boundary
        row["runtime_column"] = row["runtime_column"] or args.runtime_column
        rows.append(row)

    ablation_specs = (
        ("no_density_gradients", "No density gradients"),
        ("no_hierarchical_composition", "No hierarchical composition (depth 1)"),
        ("no_combined_rigid", "No combined rigid moves"),
        ("no_torsion", "No torsional moves"),
        ("no_coarse_orientations", "No coarse orientations"),
        ("no_basin_robustness", "No basin robustness"),
        ("no_clash_penalty", "No clash penalty"),
    )
    for order, (key, label) in enumerate(ablation_specs):
        root = ablations / key
        add(
            experiment_id=key, display_label=label, panel="ablation", method="V2",
            selected_metrics=root / "metrics_v2_selected/per_complex.csv",
            oracle_metrics=root / "metrics_expanded_pool/per_complex.csv",
            audit=root / "per_complex_audit.csv", summary=root / "summary.json",
            display_order=order, ablation_key=key,
        )
    add(
        experiment_id="main_soft_clash", display_label="Main V2: soft clash penalty",
        panel="ablation", method="V2",
        selected_metrics=main_root / "metrics_v2_selected/per_complex.csv",
        oracle_metrics=main_root / "metrics_expanded_pool/per_complex.csv",
        audit=main_root / "per_complex_audit.csv", summary=main_root / "summary.json",
        display_order=7, ablation_key="main_soft_clash",
    )
    add(
        experiment_id="hard_relative_physics", display_label="Hard relative-physics gate",
        panel="ablation", method="V2",
        selected_metrics=main_root / "metrics_expanded_density_physics/per_complex.csv",
        oracle_metrics=main_root / "metrics_expanded_pool/per_complex.csv",
        audit=main_root / "per_complex_audit.csv", summary=main_root / "summary.json",
        display_order=8, ablation_key="hard_relative_physics",
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=columns)
    missing_files = sorted({
        str(Path(value))
        for column in ("selected_metrics", "oracle_metrics", "audit", "summary")
        for value in table[column]
        if str(value) and not Path(value).is_file()
    })
    if missing_files:
        raise FileNotFoundError(
            "Figure 3 experiment outputs are incomplete:\n" + "\n".join(missing_files)
        )
    table.to_csv(output, index=False)
    print(f"FIGURE3_EXPERIMENT_MANIFEST_OK: {output} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
