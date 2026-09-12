#!/usr/bin/env python3
"""Evaluate DiffDock-Pocket protein conformers without reading ligand GT.

Single-pair mode:
  python evaluation/evaluate_sidechain_accuracy.py pair \
    --reference holo.pdb --prediction rank1_protein.pdb \
    --residues A:160-A:193-A:197

Batch mode consumes a CSV with columns ``complex_name``, ``reference_protein``
and ``prediction_protein``. Optional columns are ``input_protein``, ``rank`` and
``flexible_residues``. Relative paths are resolved against the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.evaluation import compare_sidechains  # noqa: E402


SUMMARY_FIELDS = (
    "complex_name", "condition", "rank", "reference_path", "prediction_path",
    "num_evaluated_residues", "num_matched_sidechain_heavy_atoms",
    "sidechain_rmsd_A", "sidechain_rmsd_lt_1A", "mean_chi_error_deg",
    "chi_accuracy_20deg", "chi_accuracy_40deg", "num_chi_angles",
)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _write_csv(path: Path, rows: list[dict[str, object]], fields=SUMMARY_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pair_command(args: argparse.Namespace) -> int:
    metrics = compare_sidechains(
        args.reference, args.prediction, residue_ids=args.residues,
        align_backbone=not args.no_align,
    )
    payload = _json_safe(metrics)
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")
    return 0


def batch_command(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    per_residue_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    evaluated_inputs: set[tuple[str, str]] = set()
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"complex_name", "reference_protein", "prediction_protein"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    for row in rows:
        name = row["complex_name"].strip()
        try:
            metrics = compare_sidechains(
                _resolve(manifest.parent, row["reference_protein"]),
                _resolve(manifest.parent, row["prediction_protein"]),
                residue_ids=row.get("flexible_residues") or None,
                align_backbone=not args.no_align,
            )
            summary = {key: value for key, value in metrics.items() if key != "per_residue"}
            summary.update({
                "complex_name": name,
                "condition": row.get("condition", "prediction"),
                "rank": row.get("rank", ""),
            })
            summary_rows.append(summary)
            for residue in metrics["per_residue"]:
                per_residue_rows.append({
                    "complex_name": name,
                    "condition": summary["condition"],
                    "rank": summary["rank"],
                    **residue,
                })
            input_key = (name, row.get("input_protein", ""))
            if row.get("input_protein") and input_key not in evaluated_inputs:
                baseline = compare_sidechains(
                    _resolve(manifest.parent, row["reference_protein"]),
                    _resolve(manifest.parent, row["input_protein"]),
                    residue_ids=row.get("flexible_residues") or None,
                    align_backbone=not args.no_align,
                )
                baseline_summary = {key: value for key, value in baseline.items() if key != "per_residue"}
                baseline_summary.update({"complex_name": name, "condition": "input", "rank": 0})
                summary_rows.append(baseline_summary)
                evaluated_inputs.add(input_key)
        except Exception as exc:  # continue the batch and preserve an audit trail
            failures.append({"complex_name": name, "error": f"{type(exc).__name__}: {exc}"})
    _write_csv(output_dir / "sidechain_summary.csv", summary_rows)
    residue_fields = sorted({key for row in per_residue_rows for key in row})
    if residue_fields:
        _write_csv(output_dir / "sidechain_per_residue.csv", per_residue_rows, residue_fields)
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(f"evaluated={len(summary_rows)} failures={len(failures)} output={output_dir}")
    return 1 if failures and args.fail_on_error else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pair = subparsers.add_parser("pair")
    pair.add_argument("--reference", required=True)
    pair.add_argument("--prediction", required=True)
    pair.add_argument("--residues", default=None)
    pair.add_argument("--no-align", action="store_true")
    pair.add_argument("--output", default=None)
    pair.set_defaults(func=pair_command)
    batch = subparsers.add_parser("batch")
    batch.add_argument("--manifest", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--no-align", action="store_true")
    batch.add_argument("--fail-on-error", action="store_true")
    batch.set_defaults(func=batch_command)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
