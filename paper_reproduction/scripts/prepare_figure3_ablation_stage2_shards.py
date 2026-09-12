#!/usr/bin/env python3
"""Split the unrun portion of the final evaluable 189-case cohort over two GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def read_ids(path: Path, expected: int, label: str) -> list[str]:
    values = [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != expected or len(values) != len(set(values)):
        raise ValueError(f"{label} must contain {expected} unique IDs; found {len(values)}")
    return values


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_order(values: set[str], seed: int, label: str) -> list[str]:
    return sorted(
        values,
        key=lambda name: (
            hashlib.sha256(f"{seed}:{label}:{name}".encode()).hexdigest(),
            name,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full189", required=True, type=Path)
    parser.add_argument("--original171", required=True, type=Path)
    parser.add_argument("--completed60", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args(argv)

    full_path = args.full189.expanduser().resolve()
    original_path = args.original171.expanduser().resolve()
    completed_path = args.completed60.expanduser().resolve()
    full = read_ids(full_path, 189, "full189")
    original = read_ids(original_path, 171, "original171")
    completed = read_ids(completed_path, 60, "completed60")
    full_set, original_set, completed_set = set(full), set(original), set(completed)
    if not completed_set <= original_set:
        raise ValueError("completed60 is not a subset of original171")

    # The final formal/evaluable cohort can differ from the earlier frozen 171 by
    # both additions and removals.  Completed cases outside final189 do not count
    # toward final coverage. Use original171 solely to audit and balance additions.
    added_to_final = full_set - original_set
    removed_from_old = original_set - full_set
    completed_in_final = completed_set & full_set
    completed_outside_final = completed_set - full_set
    if completed_outside_final:
        raise ValueError(
            "Completed ablation cases outside final189: "
            f"{sorted(completed_outside_final)}"
        )
    remaining = full_set - completed_in_final
    old_remaining = remaining - added_to_final
    if len(remaining) != 129:
        raise AssertionError(f"Expected 129 remaining cases; got {len(remaining)}")

    added_order = hash_order(added_to_final, args.seed, "added_to_final")
    original_order = hash_order(old_remaining, args.seed, "original_remaining")
    added_b_n = len(added_order) // 2
    added_b = added_order[:added_b_n]
    added_c = added_order[added_b_n:]
    # Preserve the requested 60-case first new job; the second job absorbs any
    # old completed cases that were later excluded from the formal cohort.
    shard_b_n = 60
    shard_c_n = 69
    original_b_n = shard_b_n - len(added_b)
    if original_b_n < 0 or len(original_order) - original_b_n != shard_c_n - len(added_c):
        raise AssertionError("Could not construct the two stage-2 shards")
    shard_b = added_b + original_order[:original_b_n]
    shard_c = added_c + original_order[original_b_n:]
    if len(shard_b) != shard_b_n or len(shard_c) != shard_c_n:
        raise AssertionError("Stage-2 shard sizes are incorrect")
    if set(shard_b) & set(shard_c) or completed_in_final & (set(shard_b) | set(shard_c)):
        raise AssertionError("Stage-2 shards overlap")
    if completed_in_final | set(shard_b) | set(shard_c) != full_set:
        raise AssertionError("Three shards do not reconstruct full189")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "shard_a_completed60": output / "shard_a_completed60.txt",
        "shard_b60": output / "shard_b60.txt",
        "shard_c69": output / "shard_c69.txt",
        "completed_outside_final": output / "completed_outside_final.txt",
        "added_to_final": output / "added_to_final.txt",
        "removed_from_old": output / "removed_from_old.txt",
        "added_to_final_b": output / "added_to_final_b.txt",
        "added_to_final_c": output / "added_to_final_c.txt",
    }
    paths["shard_a_completed60"].write_text(
        "".join(f"{name}\n" for name in completed if name in full_set)
    )
    paths["shard_b60"].write_text("".join(f"{name}\n" for name in shard_b))
    paths["shard_c69"].write_text("".join(f"{name}\n" for name in shard_c))
    paths["completed_outside_final"].write_text(
        "".join(f"{name}\n" for name in sorted(completed_outside_final))
    )
    paths["added_to_final"].write_text("".join(f"{name}\n" for name in added_order))
    paths["removed_from_old"].write_text(
        "".join(f"{name}\n" for name in sorted(removed_from_old))
    )
    paths["added_to_final_b"].write_text("".join(f"{name}\n" for name in added_b))
    paths["added_to_final_c"].write_text("".join(f"{name}\n" for name in added_c))

    membership = pd.DataFrame({"complex_name": full})
    membership["source_status"] = membership["complex_name"].map(
        lambda name: "added_to_final" if name in added_to_final else "original171"
    )
    assignment = {name: "shard_a_completed60" for name in completed_in_final}
    assignment.update({name: "shard_b60" for name in shard_b})
    assignment.update({name: "shard_c69" for name in shard_c})
    membership["shard"] = membership["complex_name"].map(assignment)
    membership.to_csv(output / "stage2_partition_audit.csv", index=False)

    audit = {
        "policy": "result-blind deterministic split of final189 minus completed-in-final",
        "seed": args.seed,
        "full_n": 189,
        "submitted_completed_n": 60,
        "completed_in_final_n": len(completed_in_final),
        "completed_outside_final_ids": sorted(completed_outside_final),
        "remaining_to_run_n": len(remaining),
        "stage2_shard_sizes": [len(shard_b), len(shard_c)],
        "old_final_overlap_n": len(full_set & original_set),
        "added_to_final_n": len(added_to_final),
        "removed_from_old_n": len(removed_from_old),
        "added_case_distribution": {
            "shard_b60": len(added_b),
            "shard_c69": len(added_c),
        },
        "inputs": {
            "full189": str(full_path),
            "original171": str(original_path),
            "completed60": str(completed_path),
        },
        "input_sha256": {
            "full189": digest(full_path),
            "original171": digest(original_path),
            "completed60": digest(completed_path),
        },
        "outputs_sha256": {name: digest(path) for name, path in paths.items()},
    }
    (output / "stage2_partition_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"FIGURE3_STAGE2_SHARDS_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
