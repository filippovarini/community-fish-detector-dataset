"""
Seaclear Marine Debris Dataset
Source: https://doi.org/10.4121/4f1dff25-e157-4399-a5d4-478055461689 (4TU.ResearchData)
Download: Kaggle mirror https://www.kaggle.com/datasets/jocelyndumlao/seaclear-marine-debris-detection-and-segmentation
    (4TU serves a single .rar and was unreachable when this script was written)
8610 ROV images (1920x1080) of shallow-water marine debris from 5 sites (Bistrina, Jakljan,
Lokrum, Slano in Croatia; Marseille in France) and 3 cameras. One COCO file (dataset.json) with
polygon masks for 40 categories; images live in <site>/<camera>/ folders but the COCO file names
have no folder, so folders are recovered by file name (names are unique across folders).
Split logic: By site. Fish are almost all at Lokrum, so Lokrum stays in train and the small
    sites Jakljan + Slano (545 images) form the val set.
Category mapping:
  fish: animal_fish
  non-fish: animal_urchin (sea urchins), animal_etc (sample crops are sea cucumbers), animal_starfish
  discard: animal_shells (empty shells, oysters, snails), animal_sponge, plant, unknown_instance
    (debris), all debris categories (can_metal, bottle_glass, tire_rubber, ...) and robot parts
    (rov_cable, rov_tortuga, rov_vehicle_leg, rov_bluerov)
Marseille images have no animal labels and are kept as background images.
"""

import json
import shutil

import kagglehub
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)

DATASET_SHORTNAME = "seaclear"
KAGGLE_DATASET = "jocelyndumlao/seaclear-marine-debris-detection-and-segmentation"
CATEGORIES_FILTER = {
    "animal_fish": "fish",
    "animal_urchin": "non-fish",
    "animal_etc": "non-fish",
    "animal_starfish": "non-fish",
    # discarded: animal_shells, animal_sponge, plant, unknown_instance, debris and rov_* categories
}
VAL_SITES = {"Jakljan", "Slano"}

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
raw_data_dir = raw_download_path / "Seaclear Marine Debris Dataset"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "seaclear_mapped_annotations.json"
image_sites_path = processing_dir / "image_sites.json"


def download_data():
    if raw_data_dir.exists():
        print(f"Data already downloaded: {raw_data_dir}")
        return

    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    raw_download_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(path, str(raw_download_path))


def build_coco():
    """Copy images into one flat folder and write a 1-indexed, bbox-only COCO file."""
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    # file name -> path; images sit in <site>/<camera>/ folders
    image_paths = {}
    for image_path in raw_data_dir.glob("*/*/*.jpg"):
        if image_path.name in image_paths:
            raise ValueError(f"Duplicate image name across folders: {image_path.name}")
        image_paths[image_path.name] = image_path

    with open(raw_data_dir / "dataset.json") as f:
        source = json.load(f)

    # Source image and annotation ids start at 0; renumber from 1
    image_id_map = {}
    images = []
    image_sites = {}
    for image in tqdm(source["images"]):
        name = image["file_name"]
        image_path = image_paths[name]
        shutil.copy2(image_path, coco_images_path / name)
        image_id_map[image["id"]] = len(images) + 1
        images.append({
            "id": len(images) + 1,
            "file_name": name,
            "width": image["width"],
            "height": image["height"],
        })
        # Folder layout is <site>/<camera>/<image>
        image_sites[name] = image_path.parent.parent.name

    annotations = []
    for ann in source["annotations"]:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        annotations.append({
            "id": len(annotations) + 1,
            "image_id": image_id_map[ann["image_id"]],
            "category_id": ann["category_id"],
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
        })

    categories = [{"id": c["id"], "name": c["name"]} for c in source["categories"]]
    assert min(c["id"] for c in categories) == 1, "Expected 1-indexed source categories"

    with open(coco_annotations_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    with open(image_sites_path, "w") as f:
        json.dump(image_sites, f)

    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)}")


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

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
    with open(image_sites_path) as f:
        image_sites = json.load(f)

    def should_the_image_be_included_in_train_set(image_path: str) -> bool:
        name = image_path.split("/")[-1].removeprefix(f"{DATASET_SHORTNAME}_")
        return image_sites[name] not in VAL_SITES

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
