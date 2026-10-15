"""
PomerFish Dataset
Source: https://zenodo.org/records/17432128
Split logic: By video. The video is the part of the filename before "_frame_"
    (e.g. dataset_116, dataset_121-2; 37 videos)
Category mapping:
  fish: all 10 classes (freshwater fish species)
  non-fish: none
  discard: none

Only PomerFishObj (images + COCO boxes) is used. PomerFishSeg is skipped: its filenames
have no video id, so it cannot be split by video. Frames without boxes (fish out of view)
are kept.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    save_preview_image,
)

DATASET_SHORTNAME = "pomerfish"
DATA_URL = "https://zenodo.org/records/17432128/files/PomerFish.rar?download=1"
RAR_FILENAME = "PomerFish.rar"
# Some source names use non-breaking spaces (\xa0); keys must match them exactly
CATEGORIES_FILTER = {
    "Perca fluviatilis": "fish",
    "Thymallus\xa0thymallus": "fish",
    "Salmo trutta\xa0morpha\xa0trutta (male)": "fish",
    "Salmo trutta\xa0morpha\xa0trutta (female)": "fish",
    "Salmoninae (Juvenile)": "fish",
    "Salmo trutta morpha fario (Adult)": "fish",
    "Salmo trutta morpha fario (juvenile)": "fish",
    "Rutilus rutilus": "fish",
    "Oncorhynchus mykiss": "fish",
    "Leuciscus idus": "fish",
}

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
raw_obj_dir = raw_data_dir / Path(RAR_FILENAME).stem / "PomerFishObj"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
prefixed_annotations_path = processing_dir / "pomerfish_coco.json"
mapped_annotations_path = processing_dir / "pomerfish_mapped_annotations.json"


def download_data():
    """Download (resuming broken downloads) and extract PomerFishObj from the RAR archive."""
    extracted_marker = raw_data_dir / ".pomerfishobj_extracted"
    if extracted_marker.exists():
        print(f"Dataset already extracted at {raw_obj_dir}")
        return

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    rar_path = raw_data_dir / RAR_FILENAME
    download_file_resumable(DATA_URL, rar_path)

    # unrar, not unar: unar truncates files larger than 1.5 MB in this archive.
    # unrar checks each file's CRC and exits non-zero on any error
    print("Extracting PomerFishObj with unrar...")
    subprocess.run(
        ["unrar", "x", "-o+", "-idq", str(rar_path), "PomerFish/PomerFishObj/*", f"{raw_data_dir}/"],
        check=True,
    )
    extracted_marker.touch()


def get_video_id(image_name: str) -> str:
    """dataset_121-2_frame_000003.PNG -> dataset_121-2 (works with or without the prefix)."""
    name = Path(image_name).name.removeprefix(f"{DATASET_SHORTNAME}_")
    return re.match(r"(.+)_frame_\d+\.", name).group(1)


def copy_images_and_prefix_annotations():
    """Copy the images with the dataset prefix and write the matching COCO file."""
    if prefixed_annotations_path.exists():
        print(f"Prefixed annotations already exist: {prefixed_annotations_path}")
        return

    images_path.mkdir(parents=True, exist_ok=True)
    with open(raw_obj_dir / "combined_standard.json") as f:
        coco = json.load(f)

    for image in tqdm(coco["images"], desc="Copying images"):
        file_name = f"{DATASET_SHORTNAME}_{image['file_name']}"
        target = images_path / file_name
        if not target.exists():
            shutil.copy2(raw_obj_dir / "images" / image["file_name"], target)
        image["file_name"] = file_name

    partial = prefixed_annotations_path.with_name(prefixed_annotations_path.name + ".part")
    with open(partial, "w") as f:
        json.dump(coco, f)
    partial.rename(prefixed_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    copy_images_and_prefix_annotations()
    map_annotations_to_fish_and_non_fish(
        prefixed_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )


def dataset_splitting():
    with open(mapped_annotations_path) as f:
        video_ids = {get_video_id(image["file_name"]) for image in json.load(f)["images"]}

    train_video_ids, _ = train_test_split(
        sorted(video_ids),
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    train_video_ids = set(train_video_ids)

    should_the_image_be_included_in_train_set = (
        lambda image_path: get_video_id(image_path) in train_video_ids
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
    save_preview_image(images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
