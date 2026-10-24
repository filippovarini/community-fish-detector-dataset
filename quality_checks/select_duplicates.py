"""
Selects the duplicate images to remove before merging, from the pairs found by quality_checks/deduplicate.py.

Each rule (drop_from, matched_in) removes every image of `drop_from` that has a match with SSCD similarity
>= THRESHOLD in `matched_in` (any match, not only its best one, so that neighbouring frames of a copied
video frame go too). See DATASET_COMPILATION.md for the reasons behind each rule.

Nothing is deleted from data/final: the selected images are written to images_to_remove.csv, which
datasets/merge_all_datasets.py skips.

Usage:
    python -m quality_checks.select_duplicates
"""

import argparse
from pathlib import Path

import pandas as pd

from datasets.settings import Settings

settings = Settings()

DEDUP_DIR = settings.base_dir / "quality" / "dedup"
REMOVE_LIST_PATH = DEDUP_DIR / "images_to_remove.csv"
THRESHOLD = 0.9

# (drop from, matched in)
DROP_COPIES = [
    ("fathomnet", "viame_fishtrack"),  # SEAMAP survey footage uploaded to both
    ("roboflow_fish", "marine_detect"),  # most of roboflow_fish is in marine_detect
    ("uiis10k", "trashcan"),  # uiis10k reuses TrashCan images
    ("ruod_r", "marine_detect"),
    ("uiis10k", "ruod_r"),
]


def load_cross_dataset_pairs(pairs_path: Path) -> pd.DataFrame:
    columns = ["similarity", "kind"] + [f"{c}_{s}" for s in ("a", "b") for c in ("dataset", "split", "file_name")]
    chunks = []
    for chunk in pd.read_csv(pairs_path, usecols=columns, chunksize=5_000_000):
        chunks.append(chunk[(chunk["kind"] == "cross-dataset") & (chunk["similarity"] >= THRESHOLD)])
    return pd.concat(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=DEDUP_DIR / "pairs.csv")
    args = parser.parse_args()

    print(f"Reading cross-dataset pairs >= {THRESHOLD} from {args.pairs}...")
    pairs = load_cross_dataset_pairs(args.pairs)

    # Each pair seen from both sides: (image, its dataset) matched in (other dataset)
    directed = pd.concat([
        pd.DataFrame({"dataset": pairs[f"dataset_{s}"], "split": pairs[f"split_{s}"],
                      "file_name": pairs[f"file_name_{s}"], "matched_in": pairs[f"dataset_{o}"],
                      "matched_file": pairs[f"file_name_{o}"], "similarity": pairs["similarity"]})
        for s, o in (("a", "b"), ("b", "a"))])

    selected = []
    for drop_from, matched_in in DROP_COPIES:
        rule = directed[(directed["dataset"] == drop_from) & (directed["matched_in"] == matched_in)]
        # one row per image, with its best match
        rule = rule.sort_values("similarity", ascending=False).drop_duplicates("file_name")
        print(f"{drop_from:15s} matched in {matched_in:15s}: {len(rule):,} images "
              f"(train {int((rule['split'] == 'train').sum()):,}, val {int((rule['split'] == 'val').sum()):,})")
        selected.append(rule)

    remove = pd.concat(selected).sort_values("similarity", ascending=False).drop_duplicates("file_name")
    remove = remove.sort_values(["dataset", "split", "file_name"])
    remove.to_csv(REMOVE_LIST_PATH, index=False, float_format="%.4f")
    print(f"\n{len(remove):,} images to remove, written to {REMOVE_LIST_PATH}")


if __name__ == "__main__":
    main()
