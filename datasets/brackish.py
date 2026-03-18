"""
Brackish Dataset
Source: https://public.roboflow.com/ds/vGBLxigwno?key=bhFPGoB3VB
Split logic: By deployment site (train_test_split on unique deployment identifiers)
Categories kept: small_fish, fish
"""

import json
import shutil
from pathlib import Path
from typing import Set

from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    download_file,
    extract_downloaded_file,
    convert_coco_annotations_from_0_indexed_to_1_indexed,
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    save_preview_image,
)


DATASET_SHORTNAME = "brackish_dataset"
CATEGORIES_FILTER = ["small_fish", "fish"]

settings = Settings()


def download_data(data_dir: Path):
    """Download and extract dataset from Roboflow."""
    data_dir.mkdir(exist_ok=True, parents=True)

    data_url = "https://public.roboflow.com/ds/vGBLxigwno?key=bhFPGoB3VB"
    data_path = data_dir / f"{DATASET_SHORTNAME}.zip"

    if data_dir.exists() and len(list(data_dir.glob("*"))) > 0:
        print("Data already downloaded and extracted")
    else:
        print("Downloading data...")
        download_file(data_url, data_path)
        print("Extracting data...")
        extract_downloaded_file(data_path, data_dir)


def join_all_images_and_annotations_into_single_coco_dataset(
    data_dir: Path, coco_images_dir: Path, coco_annotations_path: Path
):
    """
    Merges all train/val/test splits into a single COCO dataset.
    """
    if coco_images_dir.exists() and coco_annotations_path.exists():
        print("Combined dataset already exists")
        return coco_images_dir, coco_annotations_path

    # Define paths to the train, validation, and test directories
    splits = ["train", "valid", "test"]
    split_dirs = [data_dir / split for split in splits]

    # Check if all directories exist
    for split_dir in split_dirs:
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        annotation_file = split_dir / "_annotations.coco.json"
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotation_file}")

    # Create output directory
    coco_images_dir.mkdir(exist_ok=True)

    # Load and combine annotations
    combined_annotations = {"images": [], "annotations": [], "categories": []}
    image_id_offset = 0
    annotation_id_offset = 0

    for split_dir in split_dirs:
        annotation_file = split_dir / "_annotations.coco.json"
        with open(annotation_file) as f:
            split_annotations = json.load(f)

        # Update image IDs and copy images
        for image in split_annotations["images"]:
            image["id"] += image_id_offset

            # Copy image file to combined directory
            src_path = split_dir / image["file_name"]
            dst_path = coco_images_dir / image["file_name"]
            shutil.copy2(src_path, dst_path)

            combined_annotations["images"].append(image)

        # Update annotation IDs and image IDs
        for annotation in split_annotations["annotations"]:
            annotation["id"] += annotation_id_offset
            annotation["image_id"] += image_id_offset
            combined_annotations["annotations"].append(annotation)

        # Only copy categories from first split since they should be the same
        if not combined_annotations["categories"]:
            combined_annotations["categories"] = split_annotations["categories"]

        image_id_offset = max(img["id"] for img in combined_annotations["images"]) + 1
        if combined_annotations["annotations"]:
            annotation_id_offset = (
                max(ann["id"] for ann in combined_annotations["annotations"]) + 1
            )

    # Save combined annotations
    with open(coco_annotations_path, "w") as f:
        json.dump(combined_annotations, f)

    print(f"Combined dataset saved to {coco_images_dir}")
    print(f"Total images: {len(combined_annotations['images'])}")
    print(f"Total annotations: {len(combined_annotations['annotations'])}")

    return coco_images_dir, coco_annotations_path


def get_unique_deployments(image_folder: Path) -> Set:
    deployments = set()
    for image_path in image_folder.glob("*.jpg"):
        image_name_without_dataset_prefix = (
            remove_dataset_shortname_prefix_from_image_filename(
                image_path.stem, DATASET_SHORTNAME
            )
        )
        deployment = "-".join(
            image_name_without_dataset_prefix.split("_jpg")[0].split("-")[:-1]
        )
        deployments.add(deployment)

    return deployments


def get_list_of_cameras_to_include_in_train_set(image_folder: Path) -> list[str]:
    deployments = list(get_unique_deployments(image_folder))
    train_deployments, _ = train_test_split(
        list(deployments),
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return train_deployments


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_images_path.mkdir(parents=True, exist_ok=True)
    coco_annotations_path = processing_dir / settings.coco_file_name

    join_all_images_and_annotations_into_single_coco_dataset(
        raw_download_path, coco_images_path, coco_annotations_path
    )

    coco_annotations_path_1_indexed = processing_dir / "annotations_coco_1_indexed.json"
    convert_coco_annotations_from_0_indexed_to_1_indexed(
        coco_annotations_path, coco_annotations_path_1_indexed
    )

    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    compress_annotations_to_single_category(
        coco_annotations_path_1_indexed, CATEGORIES_FILTER, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        coco_images_path,
        compressed_annotations_path,
        DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(coco_images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_deployments = get_list_of_cameras_to_include_in_train_set(coco_images_path)
    should_the_image_be_included_in_train_set = lambda image_filename: any(
        remove_dataset_shortname_prefix_from_image_filename(
            image_filename, DATASET_SHORTNAME
        ).startswith(deployment)
        for deployment in train_deployments
    )

    train_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    )
    val_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    )
    train_dataset_path.mkdir(parents=True)
    val_dataset_path.mkdir(parents=True)

    split_coco_dataset_into_train_validation(
        coco_images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


if __name__ == "__main__":
    main()
