"""
Merges all processed datasets into a single unified dataset.

Reads from the processed/final directory where each dataset has been split
into train/val sets, and creates a single merged dataset with all images
and annotations combined.

Skips the duplicate images listed in data/quality/dedup/images_to_remove.csv
(created by `python -m quality_checks.select_duplicates`, see DATASET_COMPILATION.md).
"""

import csv
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from datasets.settings import Settings


settings = Settings()

REMOVE_LIST_PATH = settings.base_dir / "quality" / "dedup" / "images_to_remove.csv"


def load_images_to_remove() -> set[str]:
    if not REMOVE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"{REMOVE_LIST_PATH} not found. Run the deduplication first (see DATASET_COMPILATION.md)"
        )
    with open(REMOVE_LIST_PATH) as f:
        return {row["file_name"] for row in csv.DictReader(f)}


def main():
    images_to_remove = load_images_to_remove()
    print(f"Skipping {len(images_to_remove):,} duplicate images listed in {REMOVE_LIST_PATH}")
    n_removed = 0

    final_dataset_dir = settings.processed_dir

    merged_dataset_dir = settings.base_dir / "community_fish_detector"
    merged_dataset_dir.mkdir(parents=True, exist_ok=True)
    merged_images_dir = merged_dataset_dir / settings.images_folder_name
    merged_images_dir.mkdir(parents=True, exist_ok=True)
    merged_annotations_file = merged_dataset_dir / settings.coco_file_name

    # Initialize the merged COCO dataset structure
    merged_coco = {
        "images": [],
        "annotations": [],
        "categories": settings.coco_categories,
    }

    for dataset in final_dataset_dir.glob("*"):
        image_id_old_to_new_map = {}

        if dataset.is_dir():
            is_train_dataset = dataset.name.endswith(settings.train_dataset_suffix)

            print(f"Processing {dataset.name} dataset. Is train: {is_train_dataset}...")

            # Read the COCO annotations file
            coco_file = dataset / settings.coco_file_name
            if not coco_file.exists():
                raise FileNotFoundError(f"No COCO file found in {dataset.name}")

            with open(coco_file, "r") as f:
                dataset_coco = json.load(f)

            # Process images
            for image in tqdm(
                dataset_coco["images"],
                desc="Processing images",
                total=len(dataset_coco["images"]),
            ):
                file_name = Path(image["file_name"]).name
                if file_name in images_to_remove:
                    n_removed += 1
                    continue

                # Copy image to merged directory
                src_image = dataset / settings.images_folder_name / file_name
                dst_image = merged_images_dir / file_name
                shutil.copy2(src_image, dst_image)

                # Update image info
                old_image_id = image["id"]
                image["id"] = image["file_name"]
                image["is_train"] = is_train_dataset
                merged_coco["images"].append(image)
                image_id_old_to_new_map[old_image_id] = image["id"]

            # Process annotations
            for annotation in tqdm(
                dataset_coco["annotations"],
                desc="Processing annotations",
                total=len(dataset_coco["annotations"]),
            ):
                if annotation["image_id"] not in image_id_old_to_new_map:
                    continue  # image removed as a duplicate
                annotation["id"] = len(merged_coco["annotations"])
                annotation["image_id"] = image_id_old_to_new_map[
                    annotation["image_id"]
                ]
                merged_coco["annotations"].append(annotation)

    # Save the merged COCO annotations
    with open(merged_annotations_file, "w") as f:
        json.dump(merged_coco, f, indent=2)

    print(f"Skipped {n_removed:,} duplicate images")
    print(
        f"Successfully merged {len(merged_coco['images'])} images and {len(merged_coco['annotations'])} annotations"
    )
    print(f"Merged dataset saved to {merged_dataset_dir}")


if __name__ == "__main__":
    main()
