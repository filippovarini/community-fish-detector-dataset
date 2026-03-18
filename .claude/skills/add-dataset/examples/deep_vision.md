# Example: deep_vision (URL download + CSV-to-COCO conversion + deployment-based split)

This is a representative example of a dataset script that:
- Downloads from a direct URL (zip file)
- Converts CSV annotations to COCO format
- Splits by deployment identifier extracted from filenames

```python
"""
Deep Vision Dataset
Source: https://ftp.nmdc.no/nmdc/IMR/MachineLearning/fishDatasetSimulationAlgorithm.zip
Split logic: By deployment (train_test_split on unique deployment identifiers)
Categories kept: All (all are fish)
"""

import csv
import json
import shutil
from pathlib import Path
from typing import Set

from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    download_and_extract,
    CompressionType,
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    save_preview_image,
)


DATASET_SHORTNAME = "deep_vision"
DATA_URL = (
    "https://ftp.nmdc.no/nmdc/IMR/MachineLearning/fishDatasetSimulationAlgorithm.zip"
)
CATEGORIES_FILTER = None

settings = Settings()


def download_data(download_path: Path):
    download_and_extract(
        download_path, DATA_URL, DATASET_SHORTNAME, CompressionType.ZIP
    )


def csvs_to_coco(download_dir: Path, csv_files, images_path, output_json):
    """Converts multiple CSV files with annotations to a COCO-format JSON file."""
    images = {}
    annotations = []
    categories = {}

    ann_id = 1
    image_id = 1

    for csv_file in csv_files:
        with csv_file.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 6:
                    continue
                rel_file_name, xmin, ymin, xmax, ymax, label = row

                rel_file_name = rel_file_name.lstrip("/")
                raw_image_path = download_dir / "fish_dataset" / rel_file_name
                file_name = raw_image_path.name

                try:
                    shutil.copy2(raw_image_path, images_path / file_name)
                except Exception as e:
                    print(f"Error copying image {file_name}: {e}")
                    continue

                try:
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                except ValueError:
                    continue

                width = xmax - xmin
                height = ymax - ymin

                if file_name not in images:
                    images[file_name] = {
                        "id": image_id,
                        "file_name": file_name,
                        "width": None,
                        "height": None,
                    }
                    image_id += 1

                if label not in categories:
                    cat_id = len(categories) + 1
                    categories[label] = {
                        "id": cat_id,
                        "name": label,
                        "supercategory": label,
                    }
                cat_id = categories[label]["id"]

                ann = {
                    "id": ann_id,
                    "image_id": images[file_name]["id"],
                    "category_id": cat_id,
                    "bbox": [xmin, ymin, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
                annotations.append(ann)
                ann_id += 1

    coco_dict = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": list(categories.values()),
    }

    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=4)


def create_coco_dataset(download_dir: Path, images_path: Path, annotations_path: Path):
    csv_files = [
        download_dir / "fish_dataset/2017/train/source-train2017-annotations.csv",
        download_dir / "fish_dataset/2017/test/test_2017_annotations.csv",
        download_dir / "fish_dataset/2018/train/source-train2018-annotations.csv",
        download_dir / "fish_dataset/2018/test/test_2018_annotations.csv",
    ]

    if annotations_path.exists():
        print(f"COCO dataset already exists: {annotations_path}")
    else:
        csvs_to_coco(download_dir, csv_files, images_path, annotations_path)

    return images_path, annotations_path


def get_unique_deployments(image_folder: Path) -> Set:
    deployments = set()
    for image_path in image_folder.glob("*.jpg"):
        deployments.add(image_path.stem.split("_")[0])
    return deployments


def get_list_of_deployments_to_include_in_train_set(image_folder: Path) -> list[str]:
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

    create_coco_dataset(raw_download_path, coco_images_path, coco_annotations_path)

    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    compress_annotations_to_single_category(
        coco_annotations_path, CATEGORIES_FILTER, compressed_annotations_path
    )

    # 3. PREVIEW
    save_preview_image(coco_images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_deployments = get_list_of_deployments_to_include_in_train_set(coco_images_path)
    should_the_image_be_included_in_train_set = (
        lambda image_path: Path(image_path).stem.split("_")[0] in train_deployments
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
```
