"""
Project Natick Dataset
Source: https://github.com/microsoft/Project_Natick_Analysis/releases/download/annotated_data/data_release.zip
Split logic: Random (all images from same camera, location, and datetime)
Categories kept: Fish, Squid

Pascal VOC format annotations converted to COCO format.
"""

import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import supervision as sv

from datasets.settings import Settings
from datasets.utils import (
    download_file,
    extract_downloaded_file,
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    convert_coco_annotations_from_0_indexed_to_1_indexed,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)


DATASET_SHORTNAME = "project_natick"
DATA_URL = "https://github.com/microsoft/Project_Natick_Analysis/releases/download/annotated_data/data_release.zip"
CATEGORIES_FILTER = ["Fish", "Squid"]

settings = Settings()


def add_extension_to_filename(directory, extension=".jpg"):
    """The XML files have a filename element that does not have an extension.
    This function adds the extension to the filename."""
    print(f"Checking and updating XML annotations in {directory}...")
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            filepath = Path(directory) / filename
            tree = ET.parse(filepath)
            root = tree.getroot()

            filename_element = root.find("filename")
            if filename_element is not None:
                if not filename_element.text.endswith(extension):
                    filename_element.text += extension
                    tree.write(filepath)
                    print(f"Updated {filename}")
                else:
                    print(f"{filename} already has extension, skipped.")


def download_data(data_dir: Path):
    data_dir.mkdir(exist_ok=True, parents=True)

    data_path = data_dir / "data_release.zip"

    if not data_dir.exists() or len(list(data_dir.glob("*"))) == 0:
        print("Downloading data...")
        download_file(DATA_URL, data_path)
        print("Extracting data...")
        extract_downloaded_file(data_path, data_dir)
    else:
        print("Data already downloaded and extracted")


def get_list_of_cameras_to_include_in_train_set(image_folder: Path) -> list[str]:
    # Split the images randomly as all are from the same camera, location and datetime
    all_images = list(image_folder.glob("*.jpg"))
    train_ratio = 1 - settings.train_val_split_ratio
    train_size = int(len(all_images) * train_ratio)
    train_images = random.sample(all_images, train_size)
    return [image.name for image in train_images]


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = raw_download_path / "data_release" / "fish_arrow_worms_annotation"
    images_path = dataset_path / "JPEGImages"
    annotations_path = dataset_path / "Annotations"

    add_extension_to_filename(annotations_path)

    dataset = sv.DetectionDataset.from_pascal_voc(
        images_directory_path=str(images_path),
        annotations_directory_path=str(annotations_path),
    )
    coco_images_path = processing_dir / settings.images_folder_name
    coco_images_path.mkdir(parents=True, exist_ok=True)
    coco_annotations_path = processing_dir / settings.coco_file_name
    dataset.as_coco(str(coco_images_path), str(coco_annotations_path))

    coco_annotations_path_1_indexed = (
        processing_dir / "project_natick_1_indexed_annotations.json"
    )
    convert_coco_annotations_from_0_indexed_to_1_indexed(
        coco_annotations_path, coco_annotations_path_1_indexed
    )

    compressed_annotations_path = (
        processing_dir / "project_natick_compressed_annotations.json"
    )
    compress_annotations_to_single_category(
        coco_annotations_path_1_indexed, CATEGORIES_FILTER, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=compressed_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(coco_images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_image_names = get_list_of_cameras_to_include_in_train_set(coco_images_path)
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


if __name__ == "__main__":
    main()
