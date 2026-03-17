"""
DeepFish Dataset
Source: http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar
Split logic: By deployment (train_test_split on unique deployment identifiers)
Categories kept: All (only fish)

This dataset includes count, classification, and segmentation labels; we only use
the segmentation labels, reducing them to bounding boxes. Segmentation labels are
stored as mask images, so we parse connected components from the images.
"""

import os
import json
from pathlib import Path
from typing import Set

import cv2
from skimage import measure
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    download_and_extract,
    CompressionType,
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    save_preview_image,
)


DATASET_SHORTNAME = "deepfish"
SOURCE_URL = "http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar"
CATEGORIES_FILTER = None

settings = Settings()


def download_data(download_path: Path):
    """Download and extract the DeepFish dataset."""
    download_and_extract(
        download_path, SOURCE_URL, DATASET_SHORTNAME, CompressionType.TAR
    )


def get_boxes_from_mask_image(mask_file):
    """
    Load a binary image, find connected components, and convert to COCO-formatted bounding boxes.
    """
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    image_id = os.path.splitext(os.path.basename(mask_file))[0]

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    labels = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labels)

    annotations = []
    for idx, region in enumerate(regions):
        bbox = region.bbox
        coco_bbox = [
            bbox[1],  # x
            bbox[0],  # y
            bbox[3] - bbox[1],  # width
            bbox[2] - bbox[0],  # height
        ]
        area = coco_bbox[2] * coco_bbox[3]

        annotation = {
            "id": f"{image_id}_{str(idx).zfill(3)}",
            "image_id": image_id,
            "category_id": 1,
            "bbox": coco_bbox,
            "area": area,
            "iscrowd": 0,
        }
        annotations.append(annotation)

    return annotations


def create_coco_dataset(download_path: Path, coco_dataset_file: Path):
    """Process mask images to create a COCO format dataset."""
    segmentation_base = download_path / "DeepFish" / "Segmentation"
    segmentation_mask_base = segmentation_base / "masks" / "valid"
    segmentation_image_base = segmentation_base / "images" / "valid"

    if coco_dataset_file.exists():
        print(f"COCO dataset already exists: {coco_dataset_file}")
        return segmentation_image_base, coco_dataset_file

    if not segmentation_mask_base.exists():
        raise FileNotFoundError(f"Mask directory not found: {segmentation_mask_base}")
    if not segmentation_image_base.exists():
        raise FileNotFoundError(f"Image directory not found: {segmentation_image_base}")

    valid_masks = list(segmentation_mask_base.glob("*"))
    print(f"Found {len(valid_masks)} mask files")
    valid_images = list(segmentation_image_base.glob("*"))
    print(f"Found {len(valid_images)} image files")

    assert len(valid_images) == len(valid_masks), "Number of images and masks should match"

    annotation_records = []
    for mask_file in tqdm(valid_masks, total=len(valid_masks)):
        coco_formatted_annotations = get_boxes_from_mask_image(mask_file)
        annotation_records.extend(coco_formatted_annotations)

    print(f"Created {len(annotation_records)} annotations")

    coco_data = {
        "info": {},
        "categories": [{"name": "fish", "id": 1}],
        "annotations": annotation_records,
        "images": [],
    }

    for image_file_abs in tqdm(valid_images):
        im_cv = cv2.imread(str(image_file_abs))
        image_id = os.path.splitext(os.path.basename(image_file_abs))[0]
        coco_data["images"].append({
            "id": image_id,
            "file_name": str(image_file_abs.relative_to(segmentation_image_base)),
            "height": im_cv.shape[0],
            "width": im_cv.shape[1],
        })

    with open(coco_dataset_file, "w") as f:
        json.dump(coco_data, f, indent=1)

    print(f"COCO dataset saved to {coco_dataset_file}")
    return segmentation_image_base, coco_dataset_file


def get_unique_deployments(image_folder: Path) -> Set:
    deployments = set()
    for image_path in image_folder.glob("*.jpg"):
        image_filename_without_prefix = remove_dataset_shortname_prefix_from_image_filename(
            image_path.stem, DATASET_SHORTNAME
        )
        deployments.add(image_filename_without_prefix.split("_")[0])
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

    coco_annotations_path = processing_dir / settings.coco_file_name
    images_path, annotations_path = create_coco_dataset(
        raw_download_path, coco_annotations_path
    )

    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    compress_annotations_to_single_category(
        annotations_path, CATEGORIES_FILTER, compressed_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=images_path,
        annotations_path=compressed_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_deployments = get_list_of_deployments_to_include_in_train_set(images_path)
    should_the_image_be_included_in_train_set = (
        lambda image_filename: remove_dataset_shortname_prefix_from_image_filename(
            image_filename, DATASET_SHORTNAME
        ).split("_")[0] in train_deployments
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


if __name__ == "__main__":
    main()
