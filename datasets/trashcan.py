"""
TrashCan 1.0 — underwater trash, ROVs and marine fauna in JAMSTEC deep-sea ROV footage
Source: https://doi.org/10.13020/g1gx-y834 (Data Repository for the University of Minnesota)
Split logic: By video ID (filenames are vid_<video>_frame<frame>.jpg). The official
    train/val split is by image and shares 113 of its 115 val videos with train, so both
    official sets are merged and re-split 80/20 by video.
Category mapping (instance version):
  fish: animal_fish, animal_eel (eels and eelpouts)
  non-fish: animal_crab, animal_starfish (mostly brittle stars)
  discard: animal_etc (mix of squid, shrimp, crabs, anemones, a fish and unclear blobs),
    animal_shells (mostly dead clam shells and mussel beds), plant, rov, all trash_* classes
"""

import json
import re
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    extract_downloaded_file,
    CompressionType,
    map_annotations_to_fish_and_non_fish,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    split_coco_dataset_into_train_validation,
    save_preview_image,
)

DATASET_SHORTNAME = "trashcan"
# dataset.zip (553 MB); the two *_checkpoints.zip files on the same page are model weights
DATA_URL = (
    "https://conservancy.umn.edu/server/api/core/bitstreams/"
    "fffa78cc-3abf-42b9-b019-bc15a953a8e9/content"
)
CATEGORIES_FILTER = {
    "animal_fish": "fish",
    "animal_eel": "fish",
    "animal_crab": "non-fish",
    "animal_starfish": "non-fish",
}
SOURCE_SPLITS = ["train", "val"]

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
instance_dir = raw_download_path / "dataset" / "instance_version"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "trashcan_mapped_annotations.json"


def download_data():
    if all((instance_dir / f"instances_{s}_trashcan.json").exists() for s in SOURCE_SPLITS):
        print(f"Data already downloaded: {raw_download_path}")
        return

    raw_download_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_download_path / "dataset.zip"
    download_file_resumable(DATA_URL, zip_path)
    # Extracts to dataset/{instance_version,material_version,original_data,scripts}/
    extract_downloaded_file(zip_path, raw_download_path, CompressionType.ZIP)


def merge_source_splits_to_coco():
    """Merge the official train and val COCO files into one, dropping segmentation masks."""
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)
    images, annotations, categories = [], [], None
    for split in SOURCE_SPLITS:
        with open(instance_dir / f"instances_{split}_trashcan.json") as f:
            data = json.load(f)
        if categories is None:
            categories = [{"id": c["id"], "name": c["name"]} for c in data["categories"]]
        assert [c["name"] for c in data["categories"]] == [c["name"] for c in categories]

        new_image_ids = {}
        for image in tqdm(data["images"], desc=split):
            shutil.copy2(instance_dir / split / image["file_name"], coco_images_path)
            new_image_ids[image["id"]] = len(images) + 1
            images.append(
                {
                    "id": len(images) + 1,
                    "file_name": image["file_name"],
                    "width": image["width"],
                    "height": image["height"],
                }
            )
        for annotation in data["annotations"]:
            x, y, w, h = annotation["bbox"]
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": new_image_ids[annotation["image_id"]],
                    "category_id": annotation["category_id"],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )

    with open(coco_annotations_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    print(f"COCO dataset: {len(images)} images, {len(annotations)} annotations")


def get_video_id(image_name: str) -> str:
    """trashcan_vid_000074_frame0000016.jpg -> 000074"""
    name = remove_dataset_shortname_prefix_from_image_filename(
        Path(image_name).name, DATASET_SHORTNAME
    )
    return re.match(r"vid_(\d+)_frame\d+", name).group(1)


def get_train_videos() -> set[str]:
    with open(mapped_annotations_path) as f:
        videos = sorted({get_video_id(image["file_name"]) for image in json.load(f)["images"]})
    train_videos, _ = train_test_split(
        videos, test_size=settings.train_val_split_ratio, random_state=settings.random_state
    )
    return set(train_videos)


def main():
    # 1. DOWNLOAD
    download_data()

    # 2. PROCESS
    processing_dir.mkdir(parents=True, exist_ok=True)
    merge_source_splits_to_coco()
    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )
    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(coco_images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_videos = get_train_videos()
    should_the_image_be_included_in_train_set = (
        lambda image_name: get_video_id(image_name) in train_videos
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
        mapped_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


if __name__ == "__main__":
    main()
