"""
OBSEA Dataset
Source: OBSEA underwater fish detection dataset
Split logic: Use source split; train -> train, valid/test -> validation
Categories kept: fish and fish-like classes; Diver and Octopus vulgaris filtered out
"""

import json
import os
import shutil
from pathlib import Path

from PIL import Image
import yaml

from datasets.settings import Settings
from datasets.utils.coco import compress_annotations_to_single_category
from datasets.utils.split import split_coco_dataset_into_train_validation
from datasets.utils.images import add_dataset_shortname_prefix_to_image_names


DATASET_SHORTNAME = "obsea"

# Keep fish and fish-like categories. Non-fish classes are filtered out.
# Muraena helena (eel) and Myliobatidae (rays) are intentionally retained as
# fish-like organisms under the repo's dataset criteria.
CATEGORIES_FILTER = [
    "Chromis chromis",
    "Coris julis",
    "Dactylopterus volitans",
    "Dentex dentex",
    "Diplodus cervinus",
    "Diplodus puntazzo",
    "Diplodus sargus",
    "Diplodus vulgaris",
    "Epinephelus costae",
    "Epinephelus marginatus",
    "Mullus surmuletus",
    "Muraena helena",
    "Myliobatidae",
    "Oblada melanura",
    "Parablennius gattorugine",
    "Sarpa salpa",
    "Sciaena umbra",
    "Seriola dumerili",
    "Serranus cabrilla",
    "Sparus aurata",
    "Symphodus mediterraneus",
]

settings = Settings()


def load_class_names(data_yaml_path: Path) -> list[str]:
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    names = data["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return names


def yolo_bbox_to_coco_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    bbox_width = width * image_width
    bbox_height = height * image_height
    x_min = (x_center * image_width) - (bbox_width / 2)
    y_min = (y_center * image_height) - (bbox_height / 2)
    return [x_min, y_min, bbox_width, bbox_height]


def convert_obsea_yolo_to_coco(
    raw_data_path: Path,
    coco_images_path: Path,
    coco_annotations_path: Path,
) -> Path:
    """
    Converts OBSEA YOLO train/valid/test folders into one COCO annotation file.
    Source split is stored in each image record as source_split so we can split later.
    """
    if coco_images_path.exists() and coco_annotations_path.exists():
        print("COCO dataset already exists")
        return coco_annotations_path

    coco_images_path.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(raw_data_path / "data.yaml")
    categories = [{"id": i + 1, "name": class_name} for i, class_name in enumerate(class_names)]

    coco_data = {"images": [], "annotations": [], "categories": categories}

    image_id = 1
    annotation_id = 1

    for split_name in ["train", "valid", "test"]:
        images_dir = raw_data_path / split_name / "images"
        labels_dir = raw_data_path / split_name / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        image_paths = sorted([
            *images_dir.glob("*.jpg"),
            *images_dir.glob("*.jpeg"),
            *images_dir.glob("*.png"),
        ])

        for image_path in image_paths:
            target_image_path = coco_images_path / image_path.name
            if not target_image_path.exists():
                shutil.copy2(image_path, target_image_path)

            with Image.open(image_path) as img:
                image_width, image_height = img.size

            coco_data["images"].append({
                "id": image_id,
                "file_name": image_path.name,
                "width": image_width,
                "height": image_height,
                "source_split": split_name,
            })

            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue

                        if len(parts) != 5:
                            raise ValueError(f"Unexpected YOLO label format in {label_path}: {line}")

                        class_id_0_indexed = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])

                        category_id = class_id_0_indexed + 1
                        bbox = yolo_bbox_to_coco_bbox(
                            x_center, y_center, width, height, image_width, image_height
                        )

                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                        })
                        annotation_id += 1

            image_id += 1

    with open(coco_annotations_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"Saved COCO annotations to {coco_annotations_path}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Total categories: {len(coco_data['categories'])}")

    return coco_annotations_path


def main():
    raw_data_path = settings.raw_dir / DATASET_SHORTNAME
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"OBSEA raw data not found at {raw_data_path}. "
            "Expected train/valid/test folders under data/raw/obsea."
        )

    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_annotations_path = processing_dir / settings.coco_file_name

    convert_obsea_yolo_to_coco(raw_data_path, coco_images_path, coco_annotations_path)

    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    compress_annotations_to_single_category(
        coco_annotations_path,
        CATEGORIES_FILTER,
        compressed_annotations_path,
    )

    add_dataset_shortname_prefix_to_image_names(
        coco_images_path,
        compressed_annotations_path,
        DATASET_SHORTNAME,
    )

    os.environ.setdefault("MPLBACKEND", "Agg")
    from datasets.utils.visualization import save_preview_image

    save_preview_image(coco_images_path, compressed_annotations_path, DATASET_SHORTNAME)

    with open(compressed_annotations_path, "r") as f:
        coco_data = json.load(f)

    source_split_by_filename = {
        image["file_name"]: image.get("source_split") for image in coco_data["images"]
    }

    def should_the_image_be_included_in_train_set(image_filename: str) -> bool:
        return source_split_by_filename[image_filename] == "train"

    train_dataset_path = settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    val_dataset_path = settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"

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
