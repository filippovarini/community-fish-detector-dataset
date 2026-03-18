"""
NOAA Puget Sound Nearshore Fish Dataset
Source: https://storage.googleapis.com/public-datasets-lila/noaa-psnf/
Split logic: By camera (train_test_split on unique camera identifiers)
Categories kept: fish
"""

import json
from pathlib import Path
from typing import Set

from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    download_file,
    extract_downloaded_file,
    CompressionType,
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    convert_coco_annotations_from_0_indexed_to_1_indexed,
    remove_dataset_shortname_prefix_from_image_filename,
    save_preview_image,
)


DATASET_SHORTNAME = "noaa_puget"
CATEGORIES_FILTER = ["fish"]

settings = Settings()


def clean_annotations(annotations_path: Path):
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    cleaned_annotations = []
    print(f"Number of annotations: {len(annotations['annotations'])}")

    for annotation in annotations["annotations"]:
        if "bbox" not in annotation or len(annotation["bbox"]) == 0:
            print(f"No bbox found for {annotation['image_id']}")
        else:
            cleaned_annotations.append(annotation)

    annotations["annotations"] = cleaned_annotations

    with open(annotations_path, "w") as f:
        print(f"Number of annotations: {len(annotations['annotations'])}")
        json.dump(annotations, f)


def download_data(data_dir: Path):
    data_dir.mkdir(exist_ok=True, parents=True)

    data_url = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-images.zip"
    annotations_url = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-annotations-2023.08.19.zip"

    data_path_zip = data_dir / "images.zip"
    annotations_path_zip = data_dir / "annotations.zip"

    print("Extracting data...")
    download_file(data_url, data_path_zip)
    download_file(annotations_url, annotations_path_zip)

    extract_downloaded_file(data_path_zip, data_dir, CompressionType.ZIP)
    extract_downloaded_file(annotations_path_zip, data_dir, CompressionType.ZIP)


def get_unique_camera_names(image_folder: Path) -> Set:
    camera_names = set()
    for image_path in image_folder.glob("*.jpg"):
        camera_name = remove_dataset_shortname_prefix_from_image_filename(
            image_path.stem, DATASET_SHORTNAME
        ).split("_")[2]
        camera_names.add(camera_name)
    return camera_names


def get_list_of_cameras_to_include_in_train_set(image_folder: Path) -> list[str]:
    camera_names = list(get_unique_camera_names(image_folder))
    train_camera_names, _ = train_test_split(
        list(camera_names),
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return train_camera_names


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_images_path = raw_download_path / settings.images_folder_name
    raw_annotations_path = raw_download_path / "noaa_estuary_fish-2023.08.19.json"

    if not raw_images_path.exists() or not raw_annotations_path.exists():
        raw_download_path.mkdir(parents=True, exist_ok=True)
        download_data(raw_download_path)

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    annotations_path_1_indexed = (
        processing_dir / "noaa_puget_annotations_1_indexed.json"
    )
    convert_coco_annotations_from_0_indexed_to_1_indexed(
        raw_annotations_path, annotations_path_1_indexed
    )

    compressed_annotations_path = (
        processing_dir / "noaa_puget_compressed_annotations.json"
    )
    compressed_annotations_path = compress_annotations_to_single_category(
        annotations_path_1_indexed, CATEGORIES_FILTER, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=raw_images_path,
        annotations_path=compressed_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(raw_images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_camera_names = get_list_of_cameras_to_include_in_train_set(raw_images_path)
    print(f"Train camera names: {train_camera_names}")
    should_the_image_be_included_in_train_set = (
        lambda image_name: remove_dataset_shortname_prefix_from_image_filename(
            image_name, DATASET_SHORTNAME
        ).split("_")[2]
        in train_camera_names
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
        raw_images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


if __name__ == "__main__":
    main()
