"""
TORSI (Trawl Observer Recording System Images) Dataset
Source: https://data.csiro.au/collection/64913
Split logic: By date (2019-07-13/14/15 train, 2019-07-16/17 val). Split ratio ~0.19
Categories kept: orange_roughy, eel, misc_fish, orange_roughy_edge, chimera, oreo, shark, whiptail

Manual download required: download the .zip file from the source URL
and place it in fish-datasets/data/raw/torsi/
"""

import json
from pathlib import Path

from datasets.settings import Settings
from datasets.utils import (
    extract_downloaded_file,
    split_coco_dataset_into_train_validation,
    compress_annotations_to_single_category,
    copy_images_to_processing,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)


DATASET_SHORTNAME = "torsi"
CATEGORIES_FILTER = [
    "orange_roughy", "eel", "misc_fish", "orange_roughy_edge",
    "chimera", "oreo", "shark", "whiptail",
]

settings = Settings()

processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
annotations_path = settings.raw_dir / DATASET_SHORTNAME / "data" / "instances.json"
compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
images_path = processing_dir / "JPEGImages"


def download_data():
    """
    The dataset can't be downloaded programmatically.
    Please download it from: https://data.csiro.au/collection/64913
    """
    download_path = settings.raw_dir / DATASET_SHORTNAME
    print(f"Checking if data is already downloaded in {download_path}")
    if download_path.exists():
        print("Data already downloaded")
        return
    else:
        raise NotImplementedError(
            "The dataset can't be downloaded programmatically. "
            "Please download it from: https://data.csiro.au/collection/64913"
        )


def adjust_path():
    """Remove relative path from json leaving only the name of the images."""
    with open(compressed_annotations_path, 'r', encoding='utf-8') as annotations_file:
        annotations_json = json.load(annotations_file)
    for image in annotations_json["images"]:
        old_filename = image["file_name"]
        new_filename = old_filename.split("/")[2]
        image["file_name"] = new_filename

    with open(compressed_annotations_path, 'w', encoding='utf-8') as annotations_file:
        json.dump(annotations_json, annotations_file, indent=2)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    copy_images_to_processing(DATASET_SHORTNAME, settings.raw_dir / DATASET_SHORTNAME / "data/images/port")

    compress_annotations_to_single_category(
        annotations_path, CATEGORIES_FILTER, compressed_annotations_path
    )

    adjust_path()

    add_dataset_shortname_prefix_to_image_names(
        images_path, compressed_annotations_path, DATASET_SHORTNAME
    )


def dataset_splitting():
    # Split by date: 3 days train, 2 days val
    train_set_image_prefix = ["torsi_20190713", "torsi_20190714", "torsi_20190715"]

    should_the_image_be_included_in_train_set = (
        lambda image_path: Path(image_path).stem.split("-")[0] in train_set_image_prefix
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
        images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


def main():
    # 1. DOWNLOAD
    download_data()
    # Extract if needed
    zip_path = settings.raw_dir / DATASET_SHORTNAME / "torsi.zip"
    if zip_path.exists():
        extract_downloaded_file(zip_path, settings.raw_dir / DATASET_SHORTNAME)

    # 2. PROCESS
    processing()

    # 3. PREVIEW
    save_preview_image(images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
