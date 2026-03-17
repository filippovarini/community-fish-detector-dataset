"""
Marine Detect Dataset (FishInv + Megafauna)
Source: Roboflow (two separate datasets merged)
Split logic: By original dataset split suffix in filename (train/valid/test)
Categories kept: turtle, ray, shark, bolbometopon_muricatum, chaetodontidae,
    cheilinus_undulatus, cromileptes_altivelis, fish, haemulidae, lutjanidae,
    muraenidae, scaridae, serranidae

This script downloads, merges and processes two datasets:
Marine Detect FishInv and Marine Detect Megafauna.
These were merged as many common images had annotations split between the two.

Manual download required: download both datasets from Roboflow and place them
in fish-datasets/data/raw/marine_detect/
"""

import json
import os
import shutil
from pathlib import Path

from PIL import Image

from datasets.settings import Settings
from datasets.utils import (
    split_coco_dataset_into_train_validation,
    compress_annotations_to_single_category,
    copy_images_to_processing,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)


DATASET_SHORTNAME = "marine_detect"
CATEGORIES_FILTER = [
    "turtle", "ray", "shark", "bolbometopon_muricatum", "chaetodontidae",
    "cheilinus_undulatus", "cromileptes_altivelis", "fish", "haemulidae",
    "lutjanidae", "muraenidae", "scaridae", "serranidae",
]

settings = Settings()

processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
annotations_path = processing_dir / "annotations.json"
compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
images_path = processing_dir / "JPEGImages"


def download_data():
    """
    Manual download required. Download both Marine Detect FishInv and
    Marine Detect Megafauna from Roboflow and place in:
    fish-datasets/data/raw/marine_detect/
    """
    download_path = settings.raw_dir / DATASET_SHORTNAME
    if download_path.exists() and any(download_path.iterdir()):
        print("Data already downloaded")
        return
    raise NotImplementedError(
        "Manual download required. Download both Marine Detect FishInv and "
        "Marine Detect Megafauna from Roboflow and place in: "
        f"{download_path}"
    )


def merge_files(root_dir, dataset_name, images_dir, labels_dir):
    """
    Merges images and labels from train, valid, and test directories,
    adding dataset-specific suffixes. Skips OzFish images.
    """
    datasets = [("train", "train"), ("valid", "valid"), ("test", "test")]

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    ignored_count = 0
    empty_annotation_count = 0
    skip_patterns = ["_L.MP4.", "_R.MP4.", "_L.avi.", "_R.avi."]

    for dataset, suffix in datasets:
        img_src = os.path.join(root_dir, dataset, "images")
        lbl_src = os.path.join(root_dir, dataset, "labels")

        if os.path.exists(img_src):
            for file in os.listdir(img_src):
                if any(pattern in file for pattern in skip_patterns):
                    ignored_count += 1
                    continue
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name, ext = os.path.splitext(file)
                    new_filename = f"{dataset_name}_{name}_{suffix}{ext}"
                    shutil.copy2(os.path.join(img_src, file), os.path.join(images_dir, new_filename))

        if os.path.exists(lbl_src):
            for file in os.listdir(lbl_src):
                if any(pattern in file for pattern in skip_patterns):
                    ignored_count += 1
                    continue
                if file.lower().endswith('.txt'):
                    label_path = os.path.join(lbl_src, file)
                    name, ext = os.path.splitext(file)
                    new_filename = f"{dataset_name}_{name}_{suffix}{ext}"

                    with open(label_path, 'r') as label_file:
                        if not label_file.read().strip():
                            empty_annotation_count += 1
                            continue

                    shutil.copy2(label_path, os.path.join(labels_dir, new_filename))

    print(f"Ignored {ignored_count} OzFish files, excluded {empty_annotation_count} empty annotations")


def convert_to_coco(image_dir, label_dir, output_json, categories):
    """Converts annotation text files into a COCO format JSON."""
    coco_data = {
        "info": {"description": "Dataset in COCO format", "version": "1.0", "year": 2025},
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    image_id = 0
    annotation_id = 0

    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_name = os.path.basename(image_file)
        image_path = os.path.join(image_dir, image_name)
        annotation_file = os.path.splitext(image_name)[0] + ".txt"
        annotation_path = os.path.join(label_dir, annotation_file)

        with Image.open(image_path) as img:
            width, height = img.size

        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height,
        })

        if os.path.exists(annotation_path) and os.path.getsize(annotation_path) > 0:
            with open(annotation_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, x_min, y_min, x_max, y_max = map(float, parts)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    })
                    annotation_id += 1

        image_id += 1

    with open(output_json, "w") as json_file:
        json.dump(coco_data, json_file, indent=4)

    print(f"COCO annotations saved to {output_json}")


def create_coco_datasets():
    """Create COCO datasets from both FishInv and Megafauna."""
    raw_dir = settings.raw_dir / DATASET_SHORTNAME
    images_dir = str(raw_dir / "images")
    labels_dir = str(raw_dir / "labels")

    fishinv_dir = raw_dir / "fishinv"
    megafauna_dir = raw_dir / "megafauna"

    if Path(images_dir).exists() and any(Path(images_dir).iterdir()):
        print("Images already merged")
        return

    fishinv_categories = [
        {"id": 0, "name": "bolbometopon_muricatum"},
        {"id": 1, "name": "chaetodontidae"},
        {"id": 2, "name": "cheilinus_undulatus"},
        {"id": 3, "name": "cromileptes_altivelis"},
        {"id": 4, "name": "fish"},
        {"id": 5, "name": "haemulidae"},
        {"id": 6, "name": "lutjanidae"},
        {"id": 7, "name": "muraenidae"},
        {"id": 8, "name": "scaridae"},
        {"id": 9, "name": "serranidae"},
        {"id": 10, "name": "urchin"},
        {"id": 11, "name": "giant_clam"},
        {"id": 12, "name": "sea_cucumber"},
        {"id": 13, "name": "crown_of_thorns"},
        {"id": 14, "name": "lobster"},
    ]

    megafauna_categories = [
        {"id": 0, "name": "ray"},
        {"id": 1, "name": "shark"},
        {"id": 2, "name": "turtle"},
    ]

    if fishinv_dir.exists():
        merge_files(str(fishinv_dir), "fishinv", images_dir, labels_dir)
    if megafauna_dir.exists():
        merge_files(str(megafauna_dir), "megafauna", images_dir, labels_dir)

    fishinv_json = raw_dir / "fishinv_coco.json"
    megafauna_json = raw_dir / "megafauna_coco.json"

    convert_to_coco(images_dir, labels_dir, str(fishinv_json), fishinv_categories)
    convert_to_coco(images_dir, labels_dir, str(megafauna_json), megafauna_categories)


def merge_datasets():
    """Merge FishInv and Megafauna COCO datasets into one."""
    raw_dir = settings.raw_dir / DATASET_SHORTNAME
    fishinv_json = raw_dir / "fishinv_coco.json"
    megafauna_json = raw_dir / "megafauna_coco.json"
    merged_json = raw_dir / "annotations.json"

    if merged_json.exists():
        print("Merged annotations already exist")
        return

    with open(fishinv_json, "r") as f:
        fishinv_data = json.load(f)
    with open(megafauna_json, "r") as f:
        megafauna_data = json.load(f)

    # Offset megafauna IDs
    max_image_id = max(img["id"] for img in fishinv_data["images"]) + 1 if fishinv_data["images"] else 0
    max_ann_id = max(ann["id"] for ann in fishinv_data["annotations"]) + 1 if fishinv_data["annotations"] else 0
    max_cat_id = max(cat["id"] for cat in fishinv_data["categories"]) + 1 if fishinv_data["categories"] else 0

    for img in megafauna_data["images"]:
        img["id"] += max_image_id
    for ann in megafauna_data["annotations"]:
        ann["image_id"] += max_image_id
        ann["id"] += max_ann_id
        ann["category_id"] += max_cat_id
    for cat in megafauna_data["categories"]:
        cat["id"] += max_cat_id

    merged = {
        "images": fishinv_data["images"] + megafauna_data["images"],
        "annotations": fishinv_data["annotations"] + megafauna_data["annotations"],
        "categories": fishinv_data["categories"] + megafauna_data["categories"],
    }

    with open(merged_json, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(merged['images'])} images, {len(merged['annotations'])} annotations")


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    create_coco_datasets()
    merge_datasets()

    images_path.mkdir(parents=True, exist_ok=True)
    copy_images_to_processing(DATASET_SHORTNAME, settings.raw_dir / DATASET_SHORTNAME / "images")

    shutil.copy(
        str(settings.raw_dir / DATASET_SHORTNAME / "annotations.json"),
        str(annotations_path),
    )

    compress_annotations_to_single_category(
        annotations_path, CATEGORIES_FILTER, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path, compressed_annotations_path, DATASET_SHORTNAME
    )


def dataset_splitting():
    # Keep original dataset split from filename suffix
    should_the_image_be_included_in_train_set = (
        lambda image_path: Path(image_path).stem.split("_")[-1] == "train"
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

    # 2. PROCESS
    processing()

    # 3. PREVIEW
    save_preview_image(images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
