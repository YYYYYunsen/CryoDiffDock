#!/usr/bin/env python3
"""Freeze a chemistry-reviewed drug-like test cohort without reading outcomes.

The builder deliberately accepts only a chemistry manifest, a source cohort,
and a chemistry decision table.  Docking RMSD, NCC, confidence, runtime, and
completion files are neither accepted nor inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ID_COLUMNS = ("pdb_id", "pdb_id_upper", "complex_name")
CHEMISTRY_COLUMNS = (
    "pdb_id",
    "pdb_id_upper",
    "rcsb_matched_ccd_id",
    "rcsb_matched_name",
    "rcsb_matched_type",
    "rcsb_matched_formula",
    "rcsb_matched_inchikey",
    "rcsb_matched_drugbank_id",
    "rcsb_class",
    "rcsb_class_rule",
    "split_membership",
    "audit_class",
    "audit_rule",
    "n_heavy_atoms",
    "n_rotatable_bonds",
    "nucleotide_like_flag",
    "has_metal",
)
REQUIRED_DECISION_COLUMNS = (
    "pdb_id",
    "reviewed_ccd_id",
    "reviewed_category",
    "decision",
    "rationale",
    "source_url",
    "decision_basis",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def read_ids(path: Path) -> list[str]:
    values = [normalize_id(line) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate IDs in {path}")
    return values


def git_revision(workdir: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=workdir,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def parse_timestamp(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--freeze-timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemistry-manifest", required=True, type=Path)
    parser.add_argument("--source-cohort", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--historical-cohort", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-source-n", type=int, default=192)
    parser.add_argument("--expected-excluded-n", type=int, default=18)
    parser.add_argument("--expected-final-n", type=int, default=174)
    parser.add_argument("--freeze-timestamp")
    parser.add_argument("--local-timezone", default="America/Chicago")
    args = parser.parse_args()

    chemistry_path = args.chemistry_manifest.expanduser().resolve()
    source_path = args.source_cohort.expanduser().resolve()
    decisions_path = args.decisions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    available_columns = set(pd.read_csv(chemistry_path, nrows=0).columns)
    missing_columns = sorted(set(CHEMISTRY_COLUMNS) - available_columns)
    if missing_columns:
        raise ValueError(f"chemistry manifest lacks columns: {missing_columns}")
    chemistry = pd.read_csv(
        chemistry_path,
        dtype=str,
        usecols=list(CHEMISTRY_COLUMNS),
    ).fillna("")
    chemistry["pdb_id"] = chemistry["pdb_id"].map(normalize_id)
    if chemistry["pdb_id"].duplicated().any():
        duplicate_ids = sorted(chemistry.loc[chemistry["pdb_id"].duplicated(False), "pdb_id"].unique())
        raise ValueError(f"chemistry manifest contains duplicate PDB IDs: {duplicate_ids[:20]}")

    source_ids = read_ids(source_path)
    if len(source_ids) != args.expected_source_n:
        raise ValueError(f"expected {args.expected_source_n} source IDs, found {len(source_ids)}")
    source_set = set(source_ids)
    manifest_rows = chemistry.set_index("pdb_id").reindex(source_ids)
    if manifest_rows["rcsb_class"].isna().any():
        missing = manifest_rows.index[manifest_rows["rcsb_class"].isna()].tolist()
        raise ValueError(f"source IDs absent from chemistry manifest: {missing}")
    bad_split = manifest_rows.index[manifest_rows["split_membership"] != "em_test"].tolist()
    bad_class = manifest_rows.index[manifest_rows["rcsb_class"] != "drug_like"].tolist()
    if bad_split or bad_class:
        raise ValueError(f"source cohort mismatch: bad split={bad_split}, bad class={bad_class}")

    decisions = pd.read_csv(decisions_path, dtype=str).fillna("")
    missing_decision_columns = sorted(set(REQUIRED_DECISION_COLUMNS) - set(decisions.columns))
    if missing_decision_columns:
        raise ValueError(f"decision table lacks columns: {missing_decision_columns}")
    decisions["pdb_id"] = decisions["pdb_id"].map(normalize_id)
    if decisions["pdb_id"].duplicated().any():
        raise ValueError("decision table contains duplicate PDB IDs")
    if set(decisions["decision"]) != {"exclude"}:
        raise ValueError("every decision row must have decision=exclude")
    excluded_ids = decisions["pdb_id"].tolist()
    excluded_set = set(excluded_ids)
    if len(excluded_set) != args.expected_excluded_n:
        raise ValueError(f"expected {args.expected_excluded_n} exclusions, found {len(excluded_set)}")
    if not excluded_set <= source_set:
        raise ValueError(f"exclusions outside source cohort: {sorted(excluded_set - source_set)}")

    decision_index = decisions.set_index("pdb_id")
    for pdb_id in excluded_ids:
        observed_ccd = manifest_rows.loc[pdb_id, "rcsb_matched_ccd_id"]
        reviewed_ccd = decision_index.loc[pdb_id, "reviewed_ccd_id"]
        if observed_ccd != reviewed_ccd:
            raise ValueError(f"CCD mismatch for {pdb_id}: manifest={observed_ccd}, review={reviewed_ccd}")

    # A chemistry decision must apply to every occurrence of the same CCD in
    # the source cohort.  This prevents per-complex performance cherry-picking.
    excluded_ccd_ids = set(decisions["reviewed_ccd_id"])
    expected_ccd_exclusions = set(
        manifest_rows.index[manifest_rows["rcsb_matched_ccd_id"].isin(excluded_ccd_ids)]
    )
    if excluded_set != expected_ccd_exclusions:
        raise ValueError(
            "chemistry decisions do not cover every source-cohort occurrence of each excluded CCD; "
            f"missing={sorted(expected_ccd_exclusions - excluded_set)}, "
            f"unexpected={sorted(excluded_set - expected_ccd_exclusions)}"
        )

    final_ids = sorted(source_set - excluded_set)
    if len(final_ids) != args.expected_final_n:
        raise ValueError(f"expected {args.expected_final_n} retained IDs, found {len(final_ids)}")

    membership = manifest_rows.reset_index().copy()
    membership["source_cohort_member"] = True
    membership["final_authoritative_drug_like"] = ~membership["pdb_id"].isin(excluded_set)
    membership["final_class"] = membership["final_authoritative_drug_like"].map(
        {True: "drug_like", False: "excluded_non_drug_like"}
    )
    for column in ("reviewed_category", "rationale", "source_url", "decision_basis"):
        membership[column] = membership["pdb_id"].map(decision_index[column]).fillna("")
    membership["curation_uses_docking_outcomes"] = False
    membership = membership.sort_values("pdb_id")

    frozen_at = parse_timestamp(args.freeze_timestamp)
    local_tz = ZoneInfo(args.local_timezone)
    frozen_at_utc = frozen_at.isoformat().replace("+00:00", "Z")
    frozen_at_local = frozen_at.astimezone(local_tz).isoformat()
    revision, dirty = git_revision(Path.cwd())

    ids_path = output_dir / "authoritative_druglike174_ids.txt"
    cohort_csv_path = output_dir / "authoritative_druglike174_cohort.csv"
    exclusions_path = output_dir / "chemistry_exclusions18_ids.txt"
    membership_path = output_dir / "cohort_membership_audit.csv"
    counts_path = output_dir / "cohort_counts.csv"
    summary_path = output_dir / "cohort_summary.json"
    methods_path = output_dir / "METHODS_NOTE.md"
    environment_path = output_dir / "use_authoritative_cohort.sh"

    ids_path.write_text("\n".join(final_ids) + "\n")
    retained = membership.loc[membership["final_authoritative_drug_like"]].copy()
    if len(retained) != args.expected_final_n or retained["pdb_id"].nunique() != args.expected_final_n:
        raise RuntimeError("retained-only cohort CSV failed its 174-row uniqueness check")
    retained.to_csv(cohort_csv_path, index=False)
    exclusions_path.write_text("\n".join(sorted(excluded_set)) + "\n")
    membership.to_csv(membership_path, index=False)
    counts = pd.DataFrame(
        [
            {"definition": "source_rcsb_drug_like_em_test", "count": len(source_ids)},
            {"definition": "chemistry_only_exclusions", "count": len(excluded_set)},
            {"definition": "excluded_unique_chemical_components", "count": len(excluded_ccd_ids)},
            {"definition": "authoritative_strict_drug_like", "count": len(final_ids)},
        ]
    )
    counts.to_csv(counts_path, index=False)

    historical = None
    if args.historical_cohort is not None:
        historical_path = args.historical_cohort.expanduser().resolve()
        historical_ids = set(read_ids(historical_path))
        historical = {
            "source": str(historical_path),
            "sha256": sha256(historical_path),
            "count": len(historical_ids),
            "intersection_with_authoritative174_n": len(historical_ids & set(final_ids)),
            "authoritative174_missing_from_historical_n": len(set(final_ids) - historical_ids),
            "historical_entries_excluded_by_chemistry_review_n": len(historical_ids & excluded_set),
            "authoritative174_missing_from_historical": sorted(set(final_ids) - historical_ids),
            "historical_entries_excluded_by_chemistry_review": sorted(historical_ids & excluded_set),
            "interpretation": "Historical analysis cohort; not the current chemistry-authoritative complete cohort.",
        }

    summary = {
        "cohort_name": "authoritative_strict_druglike174_v1",
        "frozen_at_utc": frozen_at_utc,
        "frozen_at_local": frozen_at_local,
        "local_timezone": args.local_timezone,
        "definition": (
            "EM test complexes labelled drug_like in final_reviewed_v2, followed by a uniform "
            "structure/identity review excluding clear nucleotides, cofactors or cofactor-associated "
            "metabolites, endogenous bile-acid lipids, inorganic phosphate, biochemical reagent, "
            "oxidized glutathione, and the previously adjudicated peptidomimetic case."
        ),
        "counts": {
            "source": len(source_ids),
            "excluded_by_chemistry_review": len(excluded_set),
            "excluded_unique_chemical_components": len(excluded_ccd_ids),
            "retained": len(final_ids),
        },
        "curation_stage": "post hoc chemistry-label correction performed with an outcome-independent rule",
        "outcome_independence": {
            "uses_docking_outcomes": False,
            "outcome_files_read": [],
            "outcome_columns_read": [],
            "allowed_inputs": ["chemistry manifest", "source cohort IDs", "chemistry decision table"],
            "statement": (
                "Membership correction is based only on ligand identity and biological/structural role. "
                "The same exclusions apply to every evaluated method and are not conditioned on RMSD, "
                "NCC, confidence, runtime, completion, or success/failure."
            ),
        },
        "technical_failure_policy": (
            "Technical failures are not chemistry exclusions. A chemistry-eligible complex remains in "
            "the fixed denominator and is counted according to the prespecified missing/failure policy."
        ),
        "primary_analysis_inputs": {
            "cohort_csv": "authoritative_druglike174_cohort.csv",
            "id_list": "authoritative_druglike174_ids.txt",
            "fixed_denominator": len(final_ids),
        },
        "inputs": {
            "chemistry_manifest": portable_path(chemistry_path, Path.cwd()),
            "chemistry_manifest_sha256": sha256(chemistry_path),
            "chemistry_manifest_mtime_utc": datetime.fromtimestamp(
                chemistry_path.stat().st_mtime, timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "source_cohort": portable_path(source_path, Path.cwd()),
            "source_cohort_sha256": sha256(source_path),
            "source_cohort_mtime_utc": datetime.fromtimestamp(
                source_path.stat().st_mtime, timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "chemistry_decisions": portable_path(decisions_path, Path.cwd()),
            "chemistry_decisions_sha256": sha256(decisions_path),
        },
        "code": {
            "builder": portable_path(Path(__file__).resolve(), Path.cwd()),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "git_revision": revision,
            "git_worktree_dirty_at_freeze": dirty,
        },
        "historical171_comparison": historical,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    methods_path.write_text(
        "# Authoritative strict drug-like test cohort\n\n"
        f"Frozen: `{frozen_at_utc}` (`{frozen_at_local}`)\n\n"
        "The source set contains 192 EM-test complexes labelled `drug_like` in the "
        "`final_reviewed_v2` chemistry manifest. A structure- and identity-based review "
        "removed 18 clear non-drug-like cases, leaving 174 complexes. The exclusions cover "
        "nucleotides, cofactor/cofactor-associated metabolites, endogenous bile-acid lipids, "
        "inorganic pyrophosphate, a biochemical reducing reagent, oxidized glutathione, and "
        "one previously adjudicated peptidomimetic.\n\n"
        "This is a chemistry-label correction, not a performance-based case selection. The "
        "builder reads no RMSD, NCC, confidence, runtime, completion, or docking-result file. "
        "The same 174-complex membership must be applied to all methods. Chemistry-eligible "
        "technical failures remain in the denominator and must not be removed or replaced.\n\n"
        "The historical 171-complex Local list is preserved as a legacy analysis cohort. It is "
        "not the current complete chemistry-authoritative cohort because it contains cases now "
        "excluded by chemistry review and omits chemistry-eligible cases for historical technical "
        "or availability reasons.\n"
    )

    environment_path.write_text(
        "#!/usr/bin/env bash\n"
        "# Source this file before any formal drug-like analysis.\n"
        "AUTHORITATIVE_DRUGLIKE_ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "export AUTHORITATIVE_DRUGLIKE_COHORT_CSV=\"$AUTHORITATIVE_DRUGLIKE_ROOT/authoritative_druglike174_cohort.csv\"\n"
        "export AUTHORITATIVE_DRUGLIKE_IDS=\"$AUTHORITATIVE_DRUGLIKE_ROOT/authoritative_druglike174_ids.txt\"\n"
        "export AUTHORITATIVE_DRUGLIKE_N=174\n"
        "test -s \"$AUTHORITATIVE_DRUGLIKE_COHORT_CSV\"\n"
        "test -s \"$AUTHORITATIVE_DRUGLIKE_IDS\"\n"
        "echo \"Authoritative drug-like cohort: $AUTHORITATIVE_DRUGLIKE_N cases\"\n"
        "echo \"CSV: $AUTHORITATIVE_DRUGLIKE_COHORT_CSV\"\n"
        "echo \"IDs: $AUTHORITATIVE_DRUGLIKE_IDS\"\n"
    )

    hash_targets = [
        ids_path,
        cohort_csv_path,
        exclusions_path,
        membership_path,
        counts_path,
        summary_path,
        methods_path,
        environment_path,
        decisions_path,
        Path(__file__).resolve(),
    ]
    workbook_path = output_dir / "authoritative_druglike174_audit.xlsx"
    if workbook_path.exists():
        hash_targets.append(workbook_path)
    (output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {os.path.relpath(path, output_dir)}\n"
            for path in hash_targets
        )
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"AUTHORITATIVE_DRUGLIKE174_OK: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
