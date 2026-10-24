"""
Annotated video footage for automated identification and counting of fish in unconstrained
seagrass habitats (Griffith University, Ditria et al. 2021)
Source: https://doi.pangaea.de/10.1594/PANGAEA.926930
The zip holds COCO annotations (luderick_seagrass_all.json, ids 0-indexed), a per-image metadata
table (dataset_images.csv: site, deployment, date, lat/lon) and an inner luderick-seagrass.zip
with the 4,280 frames (1920x1080, extracted from video at 5 fps). Every frame has at least one box.
Split logic: By estuary. Val = Tallebudgera Creek (the authors' "novel test" site, deployments
    16-18); train = Jack Evans Boat Harbour (the authors' train + test subsets, deployments 1-15,
    which share deployments and one video, so they are not used as separate splits).
Category mapping:
  fish: luderick (Girella tricuspidata), bream (Acanthopagrus australis)
  discard: nothing
"""

import csv
import json
from pathlib import Path

from datasets.settings import Settings
from datasets.utils import (
    CompressionType,
    add_dataset_shortname_prefix_to_image_names,
    copy_images_to_processing,
    download_file_resumable,
    extract_downloaded_file,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "griffith_seagrass"
DATA_URL = "https://download.pangaea.de/dataset/926930/files/Fish_automated_identification_and_counting.zip"
CATEGORIES_FILTER = {
    "luderick": "fish",  # Girella tricuspidata
    "bream": "fish",  # Acanthopagrus australis
}
VAL_SITES = {"Tallebudgera"}

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
raw_data_dir = raw_download_path / "Fish_automated_identification_and_counting"
raw_images_dir = raw_data_dir / "images"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / "griffith_seagrass_annotations.json"
mapped_annotations_path = processing_dir / settings.coco_file_name


def download_data():
    if raw_images_dir.exists():
        print(f"Data already downloaded: {raw_data_dir}")
        return

    raw_download_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_download_path / "Fish_automated_identification_and_counting.zip"
    if not raw_data_dir.exists():
        download_file_resumable(DATA_URL, zip_path)
        extract_downloaded_file(zip_path, raw_download_path, CompressionType.ZIP)
    # The frames sit in an inner zip, flat (no folders)
    extract_downloaded_file(
        raw_data_dir / "luderick-seagrass.zip", raw_images_dir, CompressionType.ZIP
    )


def build_coco():
    """Make the source COCO file 1-indexed (images, annotations, categories) and clip boxes."""
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    with open(raw_data_dir / "luderick_seagrass_all.json") as f:
        source = json.load(f)

    images = []
    image_sizes = {}
    for image in source["images"]:
        image_sizes[image["id"]] = (image["width"], image["height"])
        images.append({
            "id": image["id"] + 1,
            "file_name": image["file_name"],
            "width": image["width"],
            "height": image["height"],
        })

    annotations = []
    n_clipped = 0
    for ann in source["annotations"]:
        width, height = image_sizes[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        # 25 boxes stick out of the frame by 1-2 px
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if (x0, y0, x1, y1) != (x, y, x + w, y + h):
            n_clipped += 1
        annotations.append({
            "id": ann["id"] + 1,
            "image_id": ann["image_id"] + 1,
            "category_id": ann["category_id"] + 1,
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "area": ann["area"],
            "segmentation": ann["segmentation"],
            "iscrowd": 0,
        })

    categories = [{"id": c["id"] + 1, "name": c["name"]} for c in source["categories"]]
    with open(coco_annotations_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)} ({n_clipped} boxes clipped to the frame)")


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    if not coco_images_path.exists():
        copy_images_to_processing(DATASET_SHORTNAME, raw_images_dir)

    build_coco()

    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )


def dataset_splitting():
    with open(raw_data_dir / "dataset_images.csv", newline="") as f:
        site_of_image = {row["file_name"]: row["dataset"] for row in csv.DictReader(f)}
    sites = sorted(set(site_of_image.values()))
    print(f"Sites: {sites}; val = {sorted(VAL_SITES)}")

    def should_the_image_be_included_in_train_set(image_path: str) -> bool:
        name = Path(image_path).name.removeprefix(f"{DATASET_SHORTNAME}_")
        return site_of_image[name] not in VAL_SITES

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
        mapped_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


def main():
    # 1. DOWNLOAD
    download_data()

    # 2. PROCESS
    processing()

    # 3. PREVIEW
    save_preview_image(coco_images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
