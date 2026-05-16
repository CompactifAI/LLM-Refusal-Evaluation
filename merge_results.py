#!/usr/bin/env python3
"""Merge results from multiple evaluation runs or dataset splits.

Two modes:
  1. Default: merge runs that share the same split structure (same splits across runs)
  2. --flat:  merge different dataset splits into one flat output (e.g. combine
             BeaverTails + general_prompts into a single censor_scores.json)

Handles deduplication by prompt_hash and source tagging.
The merged output is directly consumable by the activation extraction pipeline.

Usage:
    # Merge same-structure runs
    python merge_results.py \
        --input-dirs results/run1 results/run2 \
        --output-dir results/merged

    # Merge different dataset splits into one flat file
    python merge_results.py --flat \
        --input-dirs results/model/30k_test results/model/general_prompts \
        --output-dir results/model/merged
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set


def compute_prompt_hash(prompt: str) -> str:
    """Compute a 16-char hex hash of a prompt for deduplication."""
    if not isinstance(prompt, str):
        prompt = ""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def discover_splits(results_dir: str) -> List[str]:
    """Find all subdirectories containing censor_scores.json.

    Also handles the case where results_dir itself contains censor_scores.json
    directly (i.e., the user passed a split directory, not a parent directory).
    In that case, returns ["."] to indicate the directory itself is a split.
    """
    if not os.path.isdir(results_dir):
        return []

    # Check if this directory itself is a split (has censor_scores.json directly)
    if os.path.exists(os.path.join(results_dir, "censor_scores.json")):
        return ["."]

    # Otherwise look for subdirectories with censor_scores.json
    splits = []
    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if os.path.isdir(entry_path) and os.path.exists(
            os.path.join(entry_path, "censor_scores.json")
        ):
            splits.append(entry)
    return splits


def load_json_safe(path: str) -> Optional[Any]:
    """Load a JSON file, returning None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _resolve_split_dir(input_dir: str, split: str) -> str:
    """Resolve the actual directory path for a split."""
    if split == ".":
        return input_dir
    return os.path.join(input_dir, split)


def _load_and_merge_source(
    split_dir: str,
    source_name: str,
    seen_hashes: Set[str],
    deduplicate: bool,
    merged_censor: List[Dict[str, Any]],
    merged_answers: List[Dict[str, Any]],
    merged_judges: List[Any],
) -> int:
    """Load one source directory and append to merged lists. Returns dedup count."""
    deduped = 0
    censor_data = load_json_safe(os.path.join(split_dir, "censor_scores.json")) or []
    answers_data = load_json_safe(os.path.join(split_dir, "answers.json")) or []
    judges_data = load_json_safe(os.path.join(split_dir, "judge_scores.json")) or []

    # Build prompt_hash lookup for answers/judges to avoid index misalignment
    answers_by_hash: Dict[str, tuple] = {}
    for ans_i, ans in enumerate(answers_data):
        ans_hash = ans.get("prompt_hash") or compute_prompt_hash(ans.get("prompt", ""))
        answers_by_hash[ans_hash] = (ans_i, ans)

    for item in censor_data:
        p_hash = item.get("prompt_hash") or compute_prompt_hash(item.get("prompt", ""))

        if deduplicate and p_hash in seen_hashes:
            deduped += 1
            continue
        seen_hashes.add(p_hash)

        item["source_run"] = source_name
        if "prompt_hash" not in item:
            item["prompt_hash"] = p_hash
        merged_censor.append(item)

        match = answers_by_hash.get(p_hash)
        if match:
            ans_i, ans = match
            ans["source_run"] = source_name
            merged_answers.append(ans)
            if ans_i < len(judges_data):
                merged_judges.append(judges_data[ans_i])

    return deduped


def merge_results(
    input_dirs: List[str],
    output_dir: str,
    deduplicate: bool = True,
) -> None:
    """Merge evaluation results from multiple run directories by split name.

    Each input_dir should have the same split subdirectory structure. Results from
    matching splits are merged together (e.g., run1/jailbreakbench + run2/jailbreakbench).

    Args:
        input_dirs: List of paths to evaluation result directories.
        output_dir: Path to write merged results.
        deduplicate: If True, skip prompts with duplicate prompt_hash.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_entries: List[tuple] = []
    for input_dir in input_dirs:
        source_name = os.path.basename(os.path.normpath(input_dir))
        splits = discover_splits(input_dir)
        if not splits:
            print(f"[WARN] No splits found in {input_dir}")
            continue
        for split in splits:
            split_key = source_name if split == "." else split
            all_entries.append((input_dir, split, split_key, source_name))
        if splits == ["."]:
            print(f"[INFO] {input_dir} is a split directory (has censor_scores.json)")
        else:
            print(f"[INFO] Found {len(splits)} splits in {input_dir}: {splits}")

    if not all_entries:
        print("[ERROR] No data found in any input directory")
        return

    by_split: Dict[str, List[tuple]] = defaultdict(list)
    for input_dir, split, split_key, source_name in all_entries:
        by_split[split_key].append((input_dir, split, source_name))

    total_merged = 0
    total_deduped = 0

    for split_key, sources in sorted(by_split.items()):
        split_out_dir = os.path.join(output_dir, split_key)
        os.makedirs(split_out_dir, exist_ok=True)

        merged_censor: List[Dict[str, Any]] = []
        merged_answers: List[Dict[str, Any]] = []
        merged_judges: List[Any] = []
        seen_hashes: Set[str] = set()

        for input_dir, split, source_name in sources:
            split_dir = _resolve_split_dir(input_dir, split)
            deduped = _load_and_merge_source(
                split_dir,
                source_name,
                seen_hashes,
                deduplicate,
                merged_censor,
                merged_answers,
                merged_judges,
            )
            total_deduped += deduped

        with open(os.path.join(split_out_dir, "censor_scores.json"), "w") as f:
            json.dump(merged_censor, f, indent=2, ensure_ascii=False)
        if merged_answers:
            with open(os.path.join(split_out_dir, "answers.json"), "w") as f:
                json.dump(merged_answers, f, indent=2, ensure_ascii=False)
        if merged_judges:
            with open(os.path.join(split_out_dir, "judge_scores.json"), "w") as f:
                json.dump(merged_judges, f, indent=2, ensure_ascii=False)

        total_merged += len(merged_censor)
        print(
            f"[MERGED] {split_key}: {len(merged_censor)} entries "
            f"from {len(sources)} source(s)"
        )

    print(f"\n[DONE] Merged {total_merged} total entries to {output_dir}")
    if total_deduped > 0:
        print(f"  Deduplicated: {total_deduped} duplicate prompts removed")


def merge_datasets(
    input_dirs: List[str],
    output_dir: str,
    deduplicate: bool = True,
) -> None:
    """Merge different dataset split directories into one flat output.

    Unlike merge_results (which groups by split name), this combines ALL inputs
    into a single censor_scores.json regardless of where they came from. Use this
    to combine results from different datasets (e.g., BeaverTails + general_prompts).

    Args:
        input_dirs: List of paths — can be split directories (containing
            censor_scores.json directly) or parent directories (containing
            split subdirectories).
        output_dir: Path to write merged flat output.
        deduplicate: If True, skip prompts with duplicate prompt_hash.
    """
    os.makedirs(output_dir, exist_ok=True)

    merged_censor: List[Dict[str, Any]] = []
    merged_answers: List[Dict[str, Any]] = []
    merged_judges: List[Any] = []
    seen_hashes: Set[str] = set()
    total_deduped = 0
    source_count = 0

    for input_dir in input_dirs:
        source_name = os.path.basename(os.path.normpath(input_dir))
        splits = discover_splits(input_dir)
        if not splits:
            print(f"[WARN] No data found in {input_dir}")
            continue

        for split in splits:
            split_dir = _resolve_split_dir(input_dir, split)
            label = source_name if split == "." else f"{source_name}/{split}"
            deduped = _load_and_merge_source(
                split_dir,
                label,
                seen_hashes,
                deduplicate,
                merged_censor,
                merged_answers,
                merged_judges,
            )
            total_deduped += deduped
            count = len(merged_censor) - sum(
                1 for c in merged_censor if c.get("source_run") != label
            )
            print(f"  {label}: +{count} entries")
            source_count += 1

    if not merged_censor:
        print("[ERROR] No data found in any input directory")
        return

    # Write flat output (no split subdirectory)
    with open(os.path.join(output_dir, "censor_scores.json"), "w") as f:
        json.dump(merged_censor, f, indent=2, ensure_ascii=False)
    if merged_answers:
        with open(os.path.join(output_dir, "answers.json"), "w") as f:
            json.dump(merged_answers, f, indent=2, ensure_ascii=False)
    if merged_judges:
        with open(os.path.join(output_dir, "judge_scores.json"), "w") as f:
            json.dump(merged_judges, f, indent=2, ensure_ascii=False)

    print(
        f"\n[DONE] Merged {len(merged_censor)} entries from {source_count} source(s) "
        f"to {output_dir}"
    )
    if total_deduped > 0:
        print(f"  Deduplicated: {total_deduped} duplicate prompts removed")


def main():
    parser = argparse.ArgumentParser(
        description="Merge evaluation results from multiple runs or dataset splits."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Paths to result directories (split dirs or parent dirs with split subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to write merged results",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Merge all inputs into a single flat censor_scores.json "
        "(instead of preserving split structure)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication by prompt_hash",
    )
    args = parser.parse_args()

    if args.flat:
        merge_datasets(
            input_dirs=args.input_dirs,
            output_dir=args.output_dir,
            deduplicate=not args.no_dedup,
        )
    else:
        merge_results(
            input_dirs=args.input_dirs,
            output_dir=args.output_dir,
            deduplicate=not args.no_dedup,
        )


if __name__ == "__main__":
    main()
