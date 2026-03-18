"""
KakaduFishAI Dataset
Source: Manual download required
Split logic: Random (no location/camera grouping available)
Categories kept: All (all are fish)

Manual download required: place the dataset in fish-datasets/data/raw/kakadu/
"""

import json
import shutil
from pathlib import Path

from datasets.settings import Settings
from datasets.utils import (
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    copy_images_to_processing,
    add_dataset_shortname_prefix_to_image_names,
    get_train_images_with_random_splitting,
    save_preview_image,
)


DATASET_SHORTNAME = "kakadu"
CATEGORIES_FILTER = None

settings = Settings()

processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
annotations_path = processing_dir / "annotations_coco.json"
coco_images_path = processing_dir / settings.images_folder_name
compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"


def download_data():
    """
    Manual download required. Place the dataset in fish-datasets/data/raw/kakadu/
    """
    download_path = settings.raw_dir / DATASET_SHORTNAME
    if download_path.exists() and any(download_path.iterdir()):
        print("Data already downloaded")
        return
    raise NotImplementedError(
        "Manual download required. Place the dataset in "
        f"{download_path}"
    )


def clean_annotations():
    """Clean annotations by removing entries without bounding boxes."""
    raw_annotations_path = settings.raw_dir / DATASET_SHORTNAME / "KakaduFishAI_boundingbox.json"

    if not raw_annotations_path.exists():
        print(f"Annotations file not found: {raw_annotations_path}")
        return

    with open(raw_annotations_path, "r") as f:
        annotations = json.load(f)

    cleaned_annotations = []
    print(f"Number of annotations: {len(annotations['annotations'])}")

    for annotation in annotations["annotations"]:
        if "bbox" not in annotation or len(annotation["bbox"]) == 0:
            print(f"No bbox found for {annotation['image_id']}")
        else:
            cleaned_annotations.append(annotation)

    annotations["annotations"] = cleaned_annotations

    with open(raw_annotations_path, "w") as f:
        print(f"Number of annotations after cleaning: {len(annotations['annotations'])}")
        json.dump(annotations, f)


def annotations_processing():
    clean_annotations()

    processing_dir.mkdir(parents=True, exist_ok=True)
    coco_images_path.mkdir(parents=True, exist_ok=True)

    copy_images_to_processing(DATASET_SHORTNAME, settings.raw_dir / DATASET_SHORTNAME)

    shutil.move(str(processing_dir / "JPEGImages/KakaduFishAI_boundingbox.json"), str(annotations_path))

    compress_annotations_to_single_category(
        annotations_path, CATEGORIES_FILTER, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        coco_images_path, compressed_annotations_path, DATASET_SHORTNAME
    )


def dataset_splitting():
    # Random splitting
    train_image_names = get_train_images_with_random_splitting(coco_images_path)
    should_the_image_be_included_in_train_set = (
        lambda image_path: image_path in train_image_names
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


def main():
    # 1. DOWNLOAD
    download_data()

    # 2. PROCESS
    annotations_processing()

    # 3. PREVIEW
    save_preview_image(coco_images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
