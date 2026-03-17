"""
Roboflow Fish Dataset
Source: https://public.roboflow.com/ds/KJiCisn7wU?key=9Qk3A2qMF6
Split logic: Pre-provided split (uses Roboflow's original train/val/test split)
Categories kept: All (all are fish)
"""

import json
import shutil
from pathlib import Path

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


DATASET_SHORTNAME = "roboflow_fish"
CATEGORIES_FILTER = None

settings = Settings()


def download_data(data_dir: Path):
    data_dir.mkdir(exist_ok=True, parents=True)

    data_url = "https://public.roboflow.com/ds/KJiCisn7wU?key=9Qk3A2qMF6"
    data_path = data_dir / "roboflow_fish.zip"

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
    """Merges all train/val/test splits into a single COCO dataset."""
    splits = ["train", "valid", "test"]
    split_dirs = [data_dir / split for split in splits]

    for split_dir in split_dirs:
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        annotation_file = split_dir / "_annotations.coco.json"
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotation_file}")

    coco_images_dir.mkdir(exist_ok=True)

    combined_annotations = {"images": [], "annotations": [], "categories": []}
    image_id_offset = 0
    annotation_id_offset = 0

    for split_dir in split_dirs:
        annotation_file = split_dir / "_annotations.coco.json"
        with open(annotation_file) as f:
            split_annotations = json.load(f)

        for image in split_annotations["images"]:
            image["id"] += image_id_offset
            src_path = split_dir / image["file_name"]
            dst_path = coco_images_dir / image["file_name"]
            shutil.copy2(src_path, dst_path)
            combined_annotations["images"].append(image)

        for annotation in split_annotations["annotations"]:
            annotation["id"] += annotation_id_offset
            annotation["image_id"] += image_id_offset
            combined_annotations["annotations"].append(annotation)

        if not combined_annotations["categories"]:
            combined_annotations["categories"] = split_annotations["categories"]

        image_id_offset = max(img["id"] for img in combined_annotations["images"]) + 1
        if combined_annotations["annotations"]:
            annotation_id_offset = (
                max(ann["id"] for ann in combined_annotations["annotations"]) + 1
            )

    with open(coco_annotations_path, "w") as f:
        json.dump(combined_annotations, f)

    print(f"Combined dataset saved to {coco_images_dir}")
    print(f"Total images: {len(combined_annotations['images'])}")
    print(f"Total annotations: {len(combined_annotations['annotations'])}")

    return coco_images_dir, coco_annotations_path


def get_list_of_cameras_to_include_in_train_set(train_image_folder: Path) -> list[str]:
    """Using same split as provided by Roboflow."""
    train_image_names = list(train_image_folder.glob("*.jpg"))
    train_image_names = [f"{DATASET_SHORTNAME}_{image_name.name}" for image_name in train_image_names]
    return train_image_names


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

    coco_images_path, coco_annotations_path = (
        join_all_images_and_annotations_into_single_coco_dataset(
            raw_download_path, coco_images_path, coco_annotations_path
        )
    )

    coco_annotations_path_1_indexed = processing_dir / "annotations_coco_1_indexed.json"
    coco_annotations_path = convert_coco_annotations_from_0_indexed_to_1_indexed(
        coco_annotations_path, coco_annotations_path_1_indexed
    )

    compressed_annotations_path = (
        processing_dir / "annotations_coco_compressed.json"
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
    raw_train_set = raw_download_path / "train"
    train_image_names = get_list_of_cameras_to_include_in_train_set(raw_train_set)
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
