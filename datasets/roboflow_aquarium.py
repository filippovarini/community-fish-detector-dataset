"""
Roboflow Aquarium Dataset ("Aquarium Combined")
Source: https://universe.roboflow.com/brad-dwyer/aquarium-combined
Download: Kaggle mirror https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots
    (Roboflow YOLOv5 export; Roboflow Universe needs an API key and blocks scripts)
638 photos taken through the glass at two US aquariums (Henry Doorly Zoo, Omaha and
National Aquarium, Baltimore), late 2020. Images are resized to fit 1024x1024.
Roboflow appends "_jpg.rf.<hash>" to every name: IMG_2274_jpeg_jpg.rf.<hash>.jpg -> IMG_2274.jpg
Split logic: By camera. The provided train/valid/test split is random per image, so it is
    ignored. Filenames show two cameras: IMG_2xxx-3xxx (437 images, saved as .jpeg) go to train;
    IMG_83xx-85xx (201 images, stills and MOV video frames) go to val.
Category mapping:
  fish: fish, shark, stingray
  non-fish: jellyfish, starfish
  discard: penguin, puffin (birds, almost always photographed on rocks out of the water)
"""

import json
import re
import shutil
from pathlib import Path

import kagglehub
from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)

DATASET_SHORTNAME = "roboflow_aquarium"
KAGGLE_DATASET = "slavkoprytula/aquarium-data-cots"
# YOLO class ids 0-6 in data.yaml order
CLASS_NAMES = ["fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"]
CATEGORIES_FILTER = {
    "fish": "fish",
    "shark": "fish",
    "stingray": "fish",
    "jellyfish": "non-fish",
    "starfish": "non-fish",
    # "penguin": discarded (bird, mostly out of the water)
    # "puffin": discarded (bird, mostly out of the water)
}
# Images numbered IMG_8000 and above come from the second camera and form the val set
VAL_MIN_IMAGE_NUMBER = 8000

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
raw_data_dir = raw_download_path / "aquarium_pretrain"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "roboflow_aquarium_mapped_annotations.json"


def download_data():
    if raw_data_dir.exists():
        print(f"Data already downloaded: {raw_data_dir}")
        return

    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    raw_download_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(path, str(raw_download_path))


def original_name(roboflow_path: Path) -> str:
    """IMG_2274_jpeg_jpg.rf.<hash>.jpg -> IMG_2274.jpg, IMG_8590_MOV-4_jpg.rf.<hash>.jpg -> IMG_8590_MOV-4.jpg"""
    stem = roboflow_path.name.split(".rf.")[0]
    return re.sub(r"(_jpe?g)+$", "", stem) + ".jpg"


def yolo_to_coco():
    """Convert the YOLO labels of all source folders (train, valid, test) into one COCO dataset."""
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    image_paths = {}
    for image_path in sorted(raw_data_dir.glob("*/images/*.jpg")):
        name = original_name(image_path)
        if name in image_paths:
            raise ValueError(f"Duplicate source image: {name}")
        image_paths[name] = image_path

    images = []
    annotations = []
    for name, image_path in tqdm(sorted(image_paths.items())):
        with Image.open(image_path) as im:
            width, height = im.size
        shutil.copy2(image_path, coco_images_path / name)
        image_id = len(images) + 1
        images.append({
            "id": image_id,
            "file_name": name,
            "width": width,
            "height": height,
        })

        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            # YOLO format: class_id cx cy w h (normalized)
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            bbox_w, bbox_h = w * width, h * height
            bbox_x, bbox_y = cx * width - bbox_w / 2, cy * height - bbox_h / 2
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [round(bbox_x, 2), round(bbox_y, 2), round(bbox_w, 2), round(bbox_h, 2)],
                "area": round(bbox_w * bbox_h, 2),
                "iscrowd": 0,
            })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES, start=1)],
    }
    with open(coco_annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)}")


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    yolo_to_coco()

    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )


def dataset_splitting():
    def should_the_image_be_included_in_train_set(image_path: str) -> bool:
        name = Path(image_path).name.removeprefix(f"{DATASET_SHORTNAME}_")
        image_number = int(re.match(r"IMG_(\d+)", name).group(1))
        return image_number < VAL_MIN_IMAGE_NUMBER

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
