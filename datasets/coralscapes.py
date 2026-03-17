"""
Coralscapes Dataset
Source: https://josauder.github.io/coralscapes/
Split logic: By site (sites 1-24 train, sites 25-35 val)
Categories kept: fish (extracted from segmentation masks)

Manual download required: download the Parquet files from the source URL
and place them in fish-datasets/data/raw/coralscapes/

This script reads Parquet files, extracts images, converts segmentation masks
for the "fish" category to bounding boxes, and writes COCO annotations.
Requires: pyarrow, megadetector (for create_coco_dataset)
"""

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    split_coco_dataset_into_train_validation,
    copy_images_to_processing,
    add_dataset_shortname_prefix_to_image_names,
    save_preview_image,
)


DATASET_SHORTNAME = "coralscapes"
CATEGORIES_FILTER = None

settings = Settings()

processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
annotations_path = settings.raw_dir / DATASET_SHORTNAME / "coco/coralscapes-coco/coralscapes.json"
compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
images_path = processing_dir / "JPEGImages"


def get_bounding_boxes(mask_image, image_id, include_category_ids=None, exclude_category_id=None):
    """
    Extract bounding boxes for specified categories from a segmentation mask.
    Returns COCO-formatted annotation dicts.
    """
    mask = np.array(mask_image)
    unique_categories = np.unique(mask)
    if exclude_category_id is not None:
        unique_categories = unique_categories[unique_categories != exclude_category_id]

    annotations = []
    annotation_index = 0

    for category_id in unique_categories:
        if include_category_ids is not None and category_id not in include_category_ids:
            continue

        mask_binary = (mask == category_id).astype(np.uint8)
        labeled_mask, num_labels = measure.label(mask_binary, return_num=True, connectivity=2)

        for label_id in range(1, num_labels + 1):
            y_indices, x_indices = np.where(labeled_mask == label_id)

            if len(y_indices) > 0:
                x_min, y_min = np.min(x_indices), np.min(y_indices)
                x_max, y_max = np.max(x_indices), np.max(y_indices)
                width, height = x_max - x_min + 1, y_max - y_min + 1

                annotation_id = image_id + '_ann_' + str(annotation_index).zfill(4)

                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': int(category_id),
                    'bbox': [int(x_min), int(y_min), int(width), int(height)]
                }
                annotations.append(annotation)
                annotation_index += 1

    return annotations


def create_coco_dataset():
    """
    Read Parquet files, extract images and fish bounding boxes from segmentation masks,
    and write COCO annotations.
    """
    input_folder = settings.raw_dir / DATASET_SHORTNAME
    data_folder = input_folder / "data"

    label_mapping_file = input_folder / "id2label.json"
    with open(label_mapping_file, 'r') as f:
        category_id_to_name = json.load(f)

    category_name_to_id = {v: k for k, v in category_id_to_name.items()}
    fish_category_id = category_name_to_id['fish']
    categories_to_include = [int(fish_category_id)]

    output_image_folder = input_folder / "coco" / "coralscapes-coco"
    output_image_folder.mkdir(parents=True, exist_ok=True)
    output_coco_file = output_image_folder / "coralscapes.json"

    if output_coco_file.exists():
        print(f"COCO dataset already exists: {output_coco_file}")
        return

    parquet_files_relative = [fn for fn in os.listdir(data_folder) if fn.endswith('.parquet')]
    print(f"Found {len(parquet_files_relative)} parquet files")

    output_dict = {
        'info': {'version': '2025.03.28', 'description': 'Coralscapes dataset, fish only, converted to boxes'},
        'images': [],
        'annotations': [],
        'categories': [{'id': int(fish_category_id), 'name': 'fish'}],
    }

    for i_file, fn_relative in enumerate(parquet_files_relative):
        fn_abs = data_folder / fn_relative
        df = pd.read_parquet(fn_abs)
        print(f"Read {len(df)} rows from {fn_abs}")

        for i_row, row in tqdm(df.iterrows(), total=len(df)):
            image_fn_relative = row['image']['path']
            image_bytes = row['image']['bytes']
            img = Image.open(io.BytesIO(image_bytes))

            label_bytes = row['label']['bytes']
            label = Image.open(io.BytesIO(label_bytes))

            image_tokens = image_fn_relative.split('_')
            site = image_tokens[0]

            boxes_this_image = get_bounding_boxes(
                mask_image=label,
                image_id=image_fn_relative,
                include_category_ids=categories_to_include,
                exclude_category_id=0,
            )

            image_fn_output_abs = output_image_folder / image_fn_relative
            img.save(str(image_fn_output_abs))

            output_dict['images'].append({
                'file_name': image_fn_relative,
                'id': image_fn_relative,
                'width': img.size[0],
                'height': img.size[1],
                'location': site,
            })
            output_dict['annotations'].extend(boxes_this_image)

    with open(output_coco_file, 'w') as f:
        json.dump(output_dict, f, indent=1)

    print(f"COCO dataset saved to {output_coco_file}")


def processing():
    """Process: copy images, fix IDs, compress categories, prefix filenames."""
    processing_dir.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    copy_images_to_processing(DATASET_SHORTNAME, settings.raw_dir / DATASET_SHORTNAME / "coco/coralscapes-coco")

    # Fix issues: fish category has ID=9 (must be 1), IDs are strings, missing area/iscrowd
    with open(annotations_path, 'r', encoding='utf-8') as annotations_file:
        annotations_json = json.load(annotations_file)

    image_id = 1
    annotation_id = 1

    for image in annotations_json["images"]:
        old_id = image["id"]
        image["id"] = image_id

        for annotation in annotations_json["annotations"]:
            if annotation["image_id"] == old_id:
                bbox = annotation["bbox"]
                annotation["id"] = annotation_id
                annotation["image_id"] = image_id
                annotation["category_id"] = 1
                annotation["area"] = bbox[2] * bbox[3]
                annotation["iscrowd"] = 0
                annotation_id += 1

        image_id += 1

    annotations_json["categories"][0]["id"] = 1

    with open(compressed_annotations_path, 'w', encoding='utf-8') as annotations_file:
        json.dump(annotations_json, annotations_file, indent=2)

    add_dataset_shortname_prefix_to_image_names(
        images_path, compressed_annotations_path, DATASET_SHORTNAME
    )


def dataset_splitting():
    # Split by site: sites 1-24 train, sites 25-35 val
    should_the_image_be_included_in_train_set = (
        lambda image_path: int(Path(image_path).stem.split("_")[1].split("e")[1]) < 25
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
    # 1. DOWNLOAD + COCO CREATION
    create_coco_dataset()

    # 2. PROCESS
    processing()

    # 3. PREVIEW
    save_preview_image(images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
