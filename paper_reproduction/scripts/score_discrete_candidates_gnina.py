#!/usr/bin/env python3
"""Score an existing discrete-search shortlist with GNINA without moving poses.

The scorer deliberately uses ``gnina --score_only`` rather than ``--minimize``
so that a ranking experiment does not silently modify candidate coordinates.
All candidates belonging to a complex are evaluated against the same receptor
from the input manifest.  This makes score differences comparable and avoids
confounding ligand quality with DiffDock-Pocket's paired flexible-protein pose.

The output CSV is appended after every completed candidate and is safe to
resume: successful ``(row_index, candidate_id)`` pairs are skipped on rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SCORE_PATTERNS = {
    "gnina_affinity_kcal_mol": re.compile(rf"^Affinity:\s*({FLOAT})", re.MULTILINE),
    "gnina_cnnscore": re.compile(rf"^CNNscore:\s*({FLOAT})", re.MULTILINE),
    "gnina_cnnaffinity": re.compile(rf"^CNNaffinity:\s*({FLOAT})", re.MULTILINE),
    "gnina_cnnvariance": re.compile(rf"^CNNvariance:\s*({FLOAT})", re.MULTILINE),
    "gnina_intramolecular_energy": re.compile(
        rf"^Intramolecular energy:\s*({FLOAT})", re.MULTILINE
    ),
}


def parse_gnina_score_only(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, pattern in SCORE_PATTERNS.items():
        match = pattern.search(text)
        if match is not None:
            values[name] = float(match.group(1))
    required = {"gnina_affinity_kcal_mol", "gnina_cnnscore", "gnina_cnnaffinity"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"GNINA output is missing score fields: {missing}")
    return values


def resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_rankings(search_root: Path) -> pd.DataFrame:
    paths = sorted((search_root / "expanded_pool").glob("index*___*/ranking.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No expanded_pool ranking.csv files under {search_root}"
        )
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["ranking_csv"] = str(path.resolve())
        frames.append(frame)
    rankings = pd.concat(frames, ignore_index=True)
    required = {
        "row_index", "complex_name", "candidate_id", "materialized_path",
        "move_type",
    }
    missing = sorted(required - set(rankings.columns))
    if missing:
        raise ValueError(f"Expanded rankings are missing columns: {missing}")
    if rankings.duplicated(["row_index", "candidate_id"]).any():
        raise ValueError("Expanded rankings contain duplicate candidate identifiers")
    return rankings


def candidate_keys(path: Path) -> set[tuple[int, str]]:
    frame = pd.read_csv(path)
    required = {"row_index", "candidate_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return set(zip(frame["row_index"].astype(int), frame["candidate_id"].astype(str)))


def select_candidates(
    rankings: pd.DataFrame,
    candidate_table: Path | None,
    s0_selected: Path | None,
    move_types: set[str] | None,
) -> pd.DataFrame:
    roles: dict[tuple[int, str], set[str]] = {}

    if candidate_table is not None:
        for key in candidate_keys(candidate_table):
            roles.setdefault(key, set()).add("shortlist")
    else:
        for row in rankings.itertuples():
            if move_types is None or str(row.move_type) in move_types:
                roles.setdefault((int(row.row_index), str(row.candidate_id)), set()).add(
                    "shortlist"
                )

    if s0_selected is not None:
        for key in candidate_keys(s0_selected):
            roles.setdefault(key, set()).add("s0")

    if not roles:
        raise ValueError("Candidate selection is empty")

    keys = pd.DataFrame(
        [
            {"row_index": row_index, "candidate_id": candidate_id,
             "gnina_role": "+".join(sorted(role))}
            for (row_index, candidate_id), role in roles.items()
        ]
    )
    selected = rankings.merge(
        keys, on=["row_index", "candidate_id"], how="inner", validate="one_to_one"
    )
    missing_count = len(keys) - len(selected)
    if missing_count:
        available = set(zip(
            rankings["row_index"].astype(int), rankings["candidate_id"].astype(str)
        ))
        absent = [
            (int(row.row_index), str(row.candidate_id))
            for row in keys.itertuples()
            if (int(row.row_index), str(row.candidate_id)) not in available
        ]
        raise ValueError(
            f"{missing_count} requested candidates are absent from expanded_pool; "
            f"first examples: {absent[:10]}"
        )
    return selected.sort_values(
        ["row_index", "gnina_role", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def receptor_for_candidate(
    candidate: pd.Series,
    manifest: pd.DataFrame,
    manifest_base: Path,
) -> Path:
    source_index_value = candidate.get("source_row_index", candidate["row_index"])
    source_index = int(source_index_value)
    if 0 <= source_index < len(manifest):
        manifest_row = manifest.iloc[source_index]
        if str(manifest_row.get("complex_name", "")) != str(candidate["complex_name"]):
            matches = manifest[
                manifest["complex_name"].astype(str) == str(candidate["complex_name"])
            ]
            if len(matches) == 1:
                manifest_row = matches.iloc[0]
    else:
        matches = manifest[
            manifest["complex_name"].astype(str) == str(candidate["complex_name"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot map {candidate['complex_name']} to exactly one manifest row"
            )
        manifest_row = matches.iloc[0]

    for column in ("experimental_protein", "protein_path", "protein"):
        value = manifest_row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return resolve_path(value, manifest_base)
    raise ValueError(f"No receptor column found for {candidate['complex_name']}")


def build_command(args: argparse.Namespace, receptor: Path, ligand: Path) -> list[str]:
    command = [str(args.apptainer), "exec"]
    if args.use_gpu:
        command.append("--nv")
    command.extend([
        "--bind", f"{args.bind_root}:{args.bind_root}", str(args.gnina_sif),
        args.gnina_bin, "--receptor", str(receptor), "--ligand", str(ligand),
        "--score_only", "--cpu", str(args.cpu_per_task),
    ])
    if not args.use_gpu:
        command.append("--no_gpu")
    return command


def score_one(
    args: argparse.Namespace,
    record: dict[str, object],
    receptor: Path,
    ligand: Path,
) -> tuple[dict[str, object], str]:
    started = time.monotonic()
    command = build_command(args, receptor, ligand)
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        output = completed.stdout or ""
        result = {
            **record,
            "receptor_path": str(receptor),
            "ligand_path": str(ligand),
            "gnina_returncode": int(completed.returncode),
            "gnina_runtime_seconds": time.monotonic() - started,
        }
        if completed.returncode != 0:
            raise RuntimeError(f"GNINA exited with code {completed.returncode}")
        result.update(parse_gnina_score_only(output))
        result["gnina_status"] = "ok"
        result["gnina_error"] = ""
        return result, output
    except Exception as exc:
        output = locals().get("output", "")
        result = {
            **record,
            "receptor_path": str(receptor),
            "ligand_path": str(ligand),
            "gnina_returncode": int(locals().get("completed").returncode)
            if locals().get("completed") is not None else -1,
            "gnina_runtime_seconds": time.monotonic() - started,
            "gnina_status": "failed",
            "gnina_error": f"{type(exc).__name__}: {exc}",
        }
        for column in SCORE_PATTERNS:
            result[column] = math.nan
        return result, output


def safe_label(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:180]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-table", type=Path, default=None,
                        help="Optional CSV restricting (row_index, candidate_id) pairs")
    parser.add_argument("--s0-selected", type=Path, default=None,
                        help="Optional S0 best_selected_candidates.csv to add as controls")
    parser.add_argument("--move-types", default="coarse_orientation",
                        help="Comma-separated types used only without --candidate-table")
    parser.add_argument("--apptainer", required=True, type=Path)
    parser.add_argument("--gnina-sif", required=True, type=Path)
    parser.add_argument("--gnina-bin", default="gnina")
    parser.add_argument("--bind-root", default="/Scr")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-per-task", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit candidates after deterministic sorting (smoke tests)")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    args.manifest = args.manifest.expanduser().resolve()
    args.search_root = args.search_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.apptainer = args.apptainer.expanduser().resolve()
    args.gnina_sif = args.gnina_sif.expanduser().resolve()
    if args.candidate_table is not None:
        args.candidate_table = args.candidate_table.expanduser().resolve()
    if args.s0_selected is not None:
        args.s0_selected = args.s0_selected.expanduser().resolve()
    if not args.apptainer.is_file():
        raise FileNotFoundError(args.apptainer)
    if not args.gnina_sif.is_file():
        raise FileNotFoundError(args.gnina_sif)

    move_types = {value.strip() for value in args.move_types.split(",") if value.strip()}
    rankings = load_rankings(args.search_root)
    selected = select_candidates(
        rankings, args.candidate_table, args.s0_selected,
        move_types if move_types else None,
    )
    if args.limit is not None:
        selected = selected.iloc[:args.limit].copy()

    manifest = pd.read_csv(args.manifest)
    if "complex_name" not in manifest.columns:
        raise ValueError("Manifest is missing complex_name")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_dir = args.output_dir / "failure_logs"
    failure_dir.mkdir(exist_ok=True)
    output_csv = args.output_dir / "gnina_scores.csv"

    completed_keys: set[tuple[int, str]] = set()
    if output_csv.is_file():
        previous = pd.read_csv(output_csv)
        if {"row_index", "candidate_id", "gnina_status"}.issubset(previous.columns):
            successful = previous[previous["gnina_status"] == "ok"]
            completed_keys = set(zip(
                successful["row_index"].astype(int),
                successful["candidate_id"].astype(str),
            ))

    jobs = []
    for row in selected.to_dict(orient="records"):
        key = (int(row["row_index"]), str(row["candidate_id"]))
        if key in completed_keys:
            continue
        candidate = pd.Series(row)
        ligand = resolve_path(row["materialized_path"], args.search_root)
        receptor = receptor_for_candidate(candidate, manifest, args.manifest.parent)
        if not ligand.is_file():
            raise FileNotFoundError(ligand)
        if not receptor.is_file():
            raise FileNotFoundError(receptor)
        record = {
            "row_index": key[0],
            "source_row_index": row.get("source_row_index", key[0]),
            "complex_name": row["complex_name"],
            "candidate_id": key[1],
            "gnina_role": row["gnina_role"],
            "move_type": row["move_type"],
            "is_original": row.get("is_original", False),
            "parent_id": row.get("parent_id", ""),
            "ncc_gain": row.get("ncc_gain", math.nan),
            "expanded_density_z": row.get("expanded_density_z", math.nan),
            "confidence_z": row.get("confidence_z", math.nan),
            "severe_clash_count": row.get("severe_clash_count", math.nan),
            "parent_severe_clash_count": row.get(
                "parent_severe_clash_count", math.nan
            ),
        }
        jobs.append((record, receptor, ligand))

    print(
        f"GNINA candidates: total={len(selected)} completed={len(completed_keys)} "
        f"pending={len(jobs)} workers={args.workers}", flush=True
    )
    write_header = not output_csv.is_file() or output_csv.stat().st_size == 0
    failures = 0
    completed_now = 0
    with output_csv.open("a", newline="") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {
                executor.submit(score_one, args, record, receptor, ligand): record
                for record, receptor, ligand in jobs
            }
            for future in as_completed(future_to_job):
                result, raw_output = future.result()
                pd.DataFrame([result]).to_csv(
                    handle, index=False, header=write_header
                )
                handle.flush()
                write_header = False
                completed_now += 1
                if result["gnina_status"] != "ok":
                    failures += 1
                    log_name = (
                        f"row{result['row_index']}__"
                        f"{safe_label(result['candidate_id'])}.log"
                    )
                    (failure_dir / log_name).write_text(raw_output)
                print(
                    f"[{completed_now}/{len(jobs)}] {result['complex_name']} "
                    f"{result['candidate_id']}: {result['gnina_status']}", flush=True
                )

    scores = pd.read_csv(output_csv)
    scores = scores.sort_values(
        ["row_index", "candidate_id", "gnina_status"], kind="stable"
    ).drop_duplicates(["row_index", "candidate_id"], keep="last")
    scores.to_csv(output_csv, index=False)
    successful = scores[scores["gnina_status"] == "ok"]
    summary = {
        "manifest": str(args.manifest),
        "search_root": str(args.search_root),
        "candidate_table": None if args.candidate_table is None else str(args.candidate_table),
        "s0_selected": None if args.s0_selected is None else str(args.s0_selected),
        "num_requested_candidates": int(len(selected)),
        "num_scored_candidates": int(len(scores)),
        "num_successful_candidates": int(len(successful)),
        "num_failed_candidates": int((scores["gnina_status"] != "ok").sum()),
        "num_complexes": int(scores["row_index"].nunique()),
        "median_runtime_seconds": None if successful.empty else float(
            successful["gnina_runtime_seconds"].median()
        ),
        "uses_ground_truth": False,
        "coordinates_modified": False,
        "fixed_manifest_receptor": True,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if args.fail_on_error and (failures or summary["num_failed_candidates"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
