"""
Orange Chromide (Etroplus maculatus) — Annotated underwater fish detection dataset from pond environments
Source: https://data.mendeley.com/datasets/7w45jx35hd/1
Split logic: Random split (all images from same pond environment, no location metadata)
Categories kept: All (single class — fish)
"""

import json
import shutil
from pathlib import Path

from datasets.settings import Settings
from datasets.utils import (
    download_and_extract,
    CompressionType,
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    get_train_images_with_random_splitting,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)

DATASET_SHORTNAME = "orange_chromide"
DATA_URL = "https://data.mendeley.com/public-api/zip/7w45jx35hd/download/1"
CATEGORIES_FILTER = None

settings = Settings()

RAW_DATA_SUBDIR = (
    "Annotated underwater fish detection dataset from pond environments"
    "/Etroplus_maculatus"
)
IMAGE_SIZE = 640  # All images are 640x640


def download_data(download_path: Path):
    download_and_extract(
        download_path, DATA_URL, DATASET_SHORTNAME, CompressionType.ZIP
    )


def yolo_to_coco(raw_data_dir: Path, coco_images_path: Path, coco_annotations_path: Path):
    """Convert YOLO annotations from all splits into a single COCO dataset."""
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    images = {}
    annotations = []
    image_id = 1
    ann_id = 1

    for split in ["Train", "Valid", "Test"]:
        images_dir = raw_data_dir / split / "Images"
        labels_dir = raw_data_dir / split / "Labels"

        if not images_dir.exists():
            continue

        for label_file in sorted(labels_dir.glob("*.txt")):
            image_name = label_file.stem + ".jpg"
            image_path = images_dir / image_name

            if not image_path.exists():
                continue

            shutil.copy2(image_path, coco_images_path / image_name)

            images[image_name] = {
                "id": image_id,
                "file_name": image_name,
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE,
            }

            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    # YOLO format: class_id cx cy w h (normalized)
                    _, cx, cy, w, h = map(float, parts)

                    # Convert to COCO pixel coords [x, y, width, height]
                    bbox_w = w * IMAGE_SIZE
                    bbox_h = h * IMAGE_SIZE
                    bbox_x = (cx * IMAGE_SIZE) - (bbox_w / 2)
                    bbox_y = (cy * IMAGE_SIZE) - (bbox_h / 2)

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [round(bbox_x, 2), round(bbox_y, 2), round(bbox_w, 2), round(bbox_h, 2)],
                        "area": round(bbox_w * bbox_h, 2),
                        "iscrowd": 0,
                    })
                    ann_id += 1

            image_id += 1

    coco_dict = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [{"id": 1, "name": "fish"}],
    }

    with open(coco_annotations_path, "w") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)}")


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_annotations_path = processing_dir / settings.coco_file_name

    raw_data_dir = raw_download_path / RAW_DATA_SUBDIR
    yolo_to_coco(raw_data_dir, coco_images_path, coco_annotations_path)

    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    compress_annotations_to_single_category(
        coco_annotations_path, CATEGORIES_FILTER, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=compressed_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(coco_images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
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
    train_dataset_path.mkdir(parents=True, exist_ok=True)
    val_dataset_path.mkdir(parents=True, exist_ok=True)

    split_coco_dataset_into_train_validation(
        coco_images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


if __name__ == "__main__":
    main()
